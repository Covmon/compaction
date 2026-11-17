"""Indexed Retention: instead of a prose summary, a small model maintains a
hierarchical index (sections / subsections -> transcript message ids). At
compaction the head of the context is indexed and dropped; the agent sees the
index and can bring any past part back verbatim with `fetch_messages`.

Invariants:
- Every message that leaves the live context is covered by the index (indexer ops
  first, then harness auto-coverage), except harness-synthetic messages and
  messages that only carry `fetch_messages` results (copies of already-indexed
  messages: indexing them would create duplicate copies).
- The indexer never sees fetched content: those tool_result bodies are replaced by
  the transcript's RETRIEVED_OMITTED placeholder (also stated in the prompt).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..context import LiveContext
from ..tokens import CHARS_PER_TOKEN, estimate_tokens
from ..tools import ToolSpec
from ..transcript import RETRIEVED_OMITTED, Message
from .base import CompactionEvent, CompactionStrategy, split_tail
from .index import (
    OPS_SCHEMA,
    Index,
    IndexLookupError,
    fmt_range,
    fmt_ranges,
    mid,
    mnum,
    parse_ops,
    runs,
)

FETCH_TOOL = "fetch_messages"
READ_INDEX_TOOL = "read_index"
INDEX_LISTING_OMITTED = "[index listing omitted]"

INDEXER_SYSTEM = """\
You maintain the INDEX of an AI agent's work transcript. The agent has a small context window: \
when it fills up, older messages are removed from the agent's context and the agent keeps only \
this index. Later, when the agent needs an exact detail (a value, id, name, timestamp, file \
content, a decision), it reads the index to find WHERE that information lives and fetches those \
original messages verbatim by section/subsection id. The index is a retrieval map: its job is \
to let the agent find the right messages quickly and precisely. The original messages are \
always available, so the index must point, not paraphrase.

Structure:
- Sections ("N") group work by topic or task phase (e.g. "Task brief and plan", "Auth service \
investigation", "Database records"). Subsections ("N.M") each point at 1-6 consecutive messages \
via message-id ranges like "m0012-m0015" (or a single "m0012").
- Write SPECIFIC, CONTENT-BEARING summaries that name the concrete entities and identifiers the \
messages contain, so the agent can tell where a fact lives. Good: "Read config/payments.yaml: \
port, timeout, retry policy, feature flags (fraud_check)". Bad: "Read a config file". Good: \
"search('ERR-4471') -> 3 hits in auth-service logs, 2 tickets". Name files, services, tools, \
ticket/record ids, people and key terms. You may include short exact values, but the agent \
will fetch the originals for anything it must rely on.
- Granularity: roughly one subsection per tool call + result pair (one per file read is good), \
or per 1-6 messages of discussion. Never one giant subsection for everything.
- The first user message usually holds the task instructions: give it its own subsection whose \
summary restates the task concisely.

Continuity (you see the current index with its ids):
- Put new subsections under an existing section when they continue that topic; create a new \
section for a new topic or phase.
- Use "edit" to keep titles accurate as scope grows (e.g. rename section "2" from "Auth logs" \
to "Auth + gateway logs") or to sharpen a subsection summary.
- Use "extend" when new messages directly continue an earlier subsection (e.g. the rest of the \
same file or the same search), instead of creating a near-duplicate.

Rules:
- Index ONLY the messages listed as "to index". Every one of them must end up in some range \
(ranges may also span skipped ids). Do not reference ids that are not listed.
- Results of the agent's earlier fetch_messages calls appear as \
"[retrieved copy of earlier messages omitted]". They are copies of messages that are ALREADY \
indexed: never index their content and never create subsections for them.
- Ranges are inclusive, start <= end.

Output ONLY JSON: {"ops": [op, ...]}. Each op has the fields op, id, section, title, summary, \
ranges (set unused fields to null):
- {"op": "add_section", "title": "..."}  -> creates the next section id
- {"op": "add_subsection", "section": "<existing section id, or the exact title of a section \
you added earlier in this same ops list>", "summary": "...", "ranges": ["m0005-m0011"]}
- {"op": "edit", "id": "<N or N.M>", "title": "<new section title>" or "summary": "<new \
subsection summary>"}  (edits cannot change ranges)
- {"op": "extend", "id": "<N.M>", "ranges": ["m0040-m0041"]}
"""

RESUME_NOTICE = """\
[Context was reset to free space. Earlier messages were removed from your context but are \
kept verbatim in a permanent transcript. The index below maps that earlier work to message \
ids (mNNNN): sections "N" and subsections "N.M". The index summaries are only POINTERS to \
where information lives, not facts to rely on: whenever you need exact values (ids, numbers, \
names, timestamps, file contents) or details of earlier work, call fetch_messages with the \
relevant subsection/section ids (e.g. {"ids": ["2.1", "3"]}) to retrieve the original \
messages. Prefer narrow subsection ids. {task_hint}Your most recent messages follow the index.]
"""

SYSTEM_ADDENDUM = """

## Context management (indexed retention)
Your context window is small. When it fills up it is reset: older messages leave your \
context but are kept verbatim in a permanent transcript, organized by an index of sections \
("N") and subsections ("N.M") that map to message ids (mNNNN). After a reset you are shown the \
index.
- fetch_messages(ids): returns the exact original messages for section/subsection ids (e.g. \
["2.1", "3"]) or message ids/ranges (e.g. ["m0012-m0015"]). Use it whenever you need an exact \
value or detail from earlier work instead of guessing from index summaries, and prefer it over \
redoing work the index shows you already did. Fetch narrowly (subsections rather than whole \
sections): large fetches are truncated.
- read_index(): shows the current index.
"""


class IndexedRetentionStrategy(CompactionStrategy):
    name = "indexed"

    def __init__(
        self,
        indexer_model: str = "claude-haiku-5-5",
        fetch_budget_frac: float = 0.35,
        max_indexer_input_tokens: int = 60_000,
        indexer_max_tokens: int = 8192,
        max_index_frac: float = 0.30,
        structured_output: bool = True,
        keep_last: int | None = None,
    ) -> None:
        self.indexer_model = indexer_model
        self.fetch_budget_frac = fetch_budget_frac
        self.max_indexer_input_tokens = max_indexer_input_tokens
        self.indexer_max_tokens = indexer_max_tokens
        self.max_index_frac = max_index_frac  # max share of the cap the notice's index may use
        self.structured_output = structured_output
        self.keep_last = keep_last  # None -> AgentConfig.keep_last
        self.index = Index()
        self.fetch_calls = 0
        self.fetched_tokens = 0

    # ---- binding / helpers ------------------------------------------------
    def bind(self, llm, transcript, config) -> None:  # type: ignore[override]
        super().bind(llm, transcript, config)
        self.index = Index()

    @property
    def fetch_budget(self) -> int:
        return max(200, int(self.config.context_cap * self.fetch_budget_frac))

    def _tool_names(self) -> dict[str, str]:
        """tool_use_id -> tool name, over the whole transcript."""
        return {
            b["id"]: b.get("name", "")
            for m in self.transcript.messages
            if m.role == "assistant"
            for b in m.content
            if b.get("type") == "tool_use" and "id" in b
        }

    @staticmethod
    def _only_retrieved(m: Message, names: dict[str, str]) -> bool:
        """True if the message carries nothing but fetch_messages results."""
        if not m.retrieved:
            return False
        for b in m.content:
            t = b.get("type")
            if t == "tool_result" and names.get(b.get("tool_use_id", "")) == FETCH_TOOL:
                continue
            if t == "text" and not b.get("text", "").strip():
                continue
            if t in ("thinking", "redacted_thinking"):
                continue
            return False
        return True

    def _masked(self, m: Message, names: dict[str, str]) -> Message:
        """Copy of m with fetch_messages result bodies replaced by the transcript's
        RETRIEVED_OMITTED placeholder (and read_index listings by a short note).
        For a message holding only fetch results this equals
        Transcript.render(omit_retrieved=True); unlike it, results of OTHER tools
        issued in parallel with a fetch keep their bodies."""
        content = []
        for b in m.content:
            if b.get("type") == "tool_result":
                name = names.get(b.get("tool_use_id", ""))
                if m.retrieved and name == FETCH_TOOL:
                    b = {**b, "content": RETRIEVED_OMITTED}
                elif name == READ_INDEX_TOOL:
                    b = {**b, "content": INDEX_LISTING_OMITTED}
            content.append(b)
        return Message(m.id, m.role, content, m.session, m.retrieved, m.synthetic)

    def _render(self, msgs: list[Message], names: dict[str, str], max_chars: int | None = None) -> str:
        parts = []
        for m in msgs:
            s = self.transcript.render([self._masked(m, names)])
            if max_chars and len(s) > max_chars:
                s = s[:max_chars] + f"\n...[message truncated for indexing; {len(s) - max_chars} more chars]"
            parts.append(s)
        return "\n".join(parts)

    def _describe(self, nums: list[int]) -> str:
        """Heuristic summary for auto-covered messages: the tool calls / text they hold."""
        bits: list[str] = []
        for n in nums:
            try:
                m = self.transcript.get(mid(n))
            except KeyError:
                continue
            for b in m.content:
                if b.get("type") == "tool_use":
                    args = json.dumps(b.get("input", {}), ensure_ascii=False)
                    bits.append(f"{b.get('name')}({args[:80]})")
                elif b.get("type") == "text" and m.role == "user" and not bits:
                    bits.append("user: " + b.get("text", "").strip().replace("\n", " ")[:80])
        desc = "; ".join(bits)[:240] if bits else "messages"
        return f"{desc}  ({fmt_range((nums[0], nums[-1]))})"

    # ---- tools ------------------------------------------------------------------
    def system_addendum(self) -> str:
        return SYSTEM_ADDENDUM

    def tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(
                READ_INDEX_TOOL,
                "Show the current index of your earlier work (sections N / subsections N.M "
                "mapped to transcript message ids).",
                {"type": "object", "properties": {}},
                self._read_index,
            ),
            ToolSpec(
                FETCH_TOOL,
                "Retrieve the exact original messages of earlier work, by index section ids "
                '("2"), subsection ids ("2.1") or message ids/ranges ("m0012", "m0012-m0015"). '
                "Use whenever you need exact values or details that are no longer in your "
                "context. Prefer narrow subsection ids; large results are truncated.",
                {
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": 'e.g. ["2.1", "3"] or ["m0012-m0015"]',
                        }
                    },
                    "required": ["ids"],
                },
                self._fetch,
            ),
        ]

    def _read_index(self, _args: dict) -> str:
        if not self.index:
            return "(index is empty: no context reset has happened yet, so all your work is still in context)"
        return self.index.render()

    def _fetch(self, args: dict) -> str:
        raw = args.get("ids", [])
        if isinstance(raw, str):
            raw = [p for p in raw.replace(",", " ").split() if p]
        if not isinstance(raw, list) or not raw:
            return 'Error: pass {"ids": [...]} with section ("2"), subsection ("2.1") or message ("m0012-m0015") ids.'
        self.fetch_calls += 1
        groups, notes = [], []
        last_err: IndexLookupError | None = None
        for i in raw:
            try:
                groups.extend(self.index.expand([str(i)]))
            except IndexLookupError as e:
                notes.append(str(i))
                last_err = e
        if last_err is not None and not groups:
            return f"Error: {last_err}"
        names = self._tool_names()
        n_msgs = len(self.transcript)
        budget = self.fetch_budget
        out = [f"Original messages for {', '.join(str(i) for i in raw)} (verbatim; fetched copies and harness notes excluded):"]
        if notes:
            out.append(f"(unknown ids ignored: {', '.join(notes)}; call read_index for valid ids)")
        used = estimate_tokens("\n".join(out))
        seen: set[int] = set()
        truncated_at: str | None = None
        remaining_labels: list[str] = []
        shown = 0
        for label, nums in groups:
            if truncated_at is not None:
                remaining_labels.append(label.split(" ", 1)[0])
                continue
            msgs = []
            for n in nums:
                if n in seen or not (1 <= n <= n_msgs):
                    continue
                seen.add(n)
                m = self.transcript.get(mid(n))
                if m.synthetic or self._only_retrieved(m, names):
                    continue
                msgs.append(m)
            if not msgs:
                continue
            header = f"=== {label} ==="
            block_parts = [header]
            for m in msgs:
                s = self._render([m], names)
                cost = estimate_tokens("\n".join(block_parts + [s]))
                if used + cost > budget:
                    room = int((budget - used - estimate_tokens("\n".join(block_parts))) * CHARS_PER_TOKEN) - 200
                    if room > 300:
                        block_parts.append(s[:room] + "\n...[message cut]")
                        shown += 1
                    truncated_at = m.id
                    break
                block_parts.append(s)
                shown += 1
            out.append("\n".join(block_parts))
            used += estimate_tokens(out[-1])
        if truncated_at is not None:
            rest = f" Not shown: from {truncated_at} on" + (
                f" (incl. {', '.join(remaining_labels[:20])})" if remaining_labels else ""
            )
            out.append(
                f"[TRUNCATED: output exceeded the fetch budget of {budget} tokens.{rest}. "
                "Fetch narrower ids (single subsections or message ranges like m0012-m0013) "
                "to see the rest.]"
            )
        if shown == 0 and truncated_at is None:
            out.append("(no original messages in that range: it holds only harness notes or fetched copies)")
        text = "\n".join(out)
        self.fetched_tokens += estimate_tokens(text)
        return text

    # ---- compaction ----------------------------------------------------------
    def compact(self, ctx: LiveContext, session: int, step: int) -> tuple[LiveContext, CompactionEvent]:
        T = self.transcript
        before = estimate_tokens(ctx.to_api_messages(T))
        keep = self.config.keep_last if self.keep_last is None else self.keep_last
        head, tail = split_tail(ctx, T, keep)
        names = self._tool_names()
        tail_set = set(tail)
        # Everything outside the new tail that is not yet indexed -- including any
        # message the agent's hard-cap fallback dropped earlier without compaction.
        limit = mnum(tail[0]) - 1 if tail else len(T)
        covered = self.index.covered()
        todo: list[Message] = []
        skipped_retrieved = 0
        for m in T.messages[:limit]:
            if m.id in tail_set or m.synthetic or mnum(m.id) in covered:
                continue
            if self._only_retrieved(m, names):
                skipped_retrieved += 1
                continue
            todo.append(m)
        if not todo and all(T.get(i).synthetic for i in head):
            return ctx, CompactionEvent(step, before, before, detail={"noop": True})

        detail: dict = {
            "indexed_messages": len(todo),
            "skipped_retrieved": skipped_retrieved,
            "indexer_calls": 0,
            "sections_added": 0,
            "subsections_added": 0,
            "edits": 0,
            "extends": 0,
            "op_errors": 0,
            "op_error_examples": [],
            "indexer_failures": [],
        }
        cin = cout = 0
        for chunk in self._chunks(todo, names):
            allowed = {mnum(m.id) for m in chunk}
            lo, hi = min(allowed), max(allowed)
            # Ranges must stay within this chunk's span (they may cross skipped ids).
            rep_ops, (i_in, i_out), failure = self._run_indexer(chunk, names)
            cin, cout = cin + i_in, cout + i_out
            detail["indexer_calls"] += 1
            if failure:
                detail["indexer_failures"].append(failure)
            if rep_ops:
                rep = self.index.apply(rep_ops, valid=lambda n, lo=lo, hi=hi: lo <= n <= hi)
                for k, v in rep.as_dict().items():
                    if k == "op_error_examples":
                        detail[k] = (detail[k] + v)[:5]
                    else:
                        detail[k] += v
        _, uncovered = self.index.coverage([m.id for m in todo])
        detail["auto_covered"] = self.index.auto_cover([mnum(i) for i in uncovered], self._describe)
        detail["indexer_input_tokens"], detail["indexer_output_tokens"] = cin, cout

        notice = self._notice()
        note = T.append("user", notice, session=session + 1, synthetic=True)
        new = LiveContext([note.id] + tail)
        after = estimate_tokens(new.to_api_messages(T))
        detail["index_sections"] = len(self.index.sections)
        detail["index_subsections"] = sum(len(s.subsections) for s in self.index.sections)
        detail["index_tokens"] = estimate_tokens(self.index.render())
        detail["fetch_calls_so_far"] = self.fetch_calls
        detail["tail_start"] = tail[0] if tail else None
        self._persist()
        return new, CompactionEvent(step, before, after, cin, cout, detail)

    def _indexer_msg_budget(self) -> int:
        """Tokens available for rendered messages in one indexer request (prompt,
        current index and headroom for its growth subtracted)."""
        overhead = estimate_tokens(INDEXER_SYSTEM) + estimate_tokens(self.index.render()) + 2000
        return max(2000, self.max_indexer_input_tokens - overhead)

    def _chunks(self, todo: list[Message], names: dict[str, str]) -> list[list[Message]]:
        """Split messages so each indexer request stays under max_indexer_input_tokens
        (the prompt, the current index and some headroom for index growth included)."""
        if not todo:
            return []
        budget = self._indexer_msg_budget()
        per_msg_chars = int(budget * CHARS_PER_TOKEN * 0.5)
        chunks: list[list[Message]] = [[]]
        used = 0
        for m in todo:
            cost = estimate_tokens(self._render([m], names, per_msg_chars))
            if chunks[-1] and used + cost > budget:
                chunks.append([])
                used = 0
            chunks[-1].append(m)
            used += cost
        return chunks

    def _run_indexer(
        self, chunk: list[Message], names: dict[str, str]
    ) -> tuple[list[dict] | None, tuple[int, int], str | None]:
        """One indexer call over `chunk` (sees the current, already-updated index).
        Returns (ops or None, (input_tokens, output_tokens), failure note)."""
        per_msg_chars = int(self._indexer_msg_budget() * CHARS_PER_TOKEN * 0.5)
        ids = [m.id for m in chunk]
        prompt = (
            "<current_index>\n" + self.index.render() + "\n</current_index>\n\n"
            "<new_messages>\n" + self._render(chunk, names, per_msg_chars) + "\n</new_messages>\n\n"
            f"Messages to index ({len(ids)}): {fmt_ranges(runs(mnum(i) for i in ids))}\n"
            f"Ids to index: {', '.join(ids)}\n"
            "Update the index so that every one of these messages is covered. "
            'Return only JSON: {"ops": [...]}.'
        )
        kw: dict = {
            "model": self.indexer_model,
            "system": INDEXER_SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.indexer_max_tokens,
            "purpose": "compactor",
        }
        try:
            if self.structured_output:
                try:
                    resp = self.llm.create(**kw, output_format=OPS_SCHEMA)
                except TypeError:  # an LLM client without structured-output support
                    resp = self.llm.create(**kw)
            else:
                resp = self.llm.create(**kw)
        except Exception as e:  # noqa: BLE001 -- the strategy must never fail
            return None, (0, 0), f"indexer call failed: {type(e).__name__}: {e}"[:300]
        usage = (resp.total_input_tokens, resp.output_tokens)
        ops = parse_ops(resp.text)
        if ops is None:
            return None, usage, f"unparseable indexer output (stop_reason={resp.stop_reason})"
        return ops, usage, None

    def _notice(self) -> str:
        names = self._tool_names()
        first_task = next(
            (
                m.id
                for m in self.transcript.messages
                if m.role == "user" and not m.synthetic and not self._only_retrieved(m, names)
            ),
            None,
        )
        hint = (
            f"Your original task instructions are in message {first_task} "
            f'(fetch_messages {{"ids": ["{first_task}"]}} if unsure about requirements). '
            if first_task
            else ""
        )
        rendered = self.index.render()
        max_tok = int(self.config.context_cap * self.max_index_frac)
        compact_note = ""
        if estimate_tokens(rendered) > max_tok:
            for cut in (160, 100, 60):
                rendered = self.index.render(max_summary_chars=cut)
                if estimate_tokens(rendered) <= max_tok:
                    break
            else:
                rendered = self.index.render(sections_only=True)
            compact_note = "(Index shown abbreviated to save space; call read_index for the full version.)\n"
        return RESUME_NOTICE.replace("{task_hint}", hint) + "\n" + compact_note + rendered

    def _persist(self) -> None:
        p = getattr(self, "transcript_path", None)
        if not p:
            return
        p = Path(p)
        prefix = "" if p.name == "transcript.jsonl" else p.stem + "."
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            (p.parent / f"{prefix}index.md").write_text(self.index.render() + "\n", encoding="utf-8")
            (p.parent / f"{prefix}index.json").write_text(
                json.dumps(self.index.to_json(), indent=1, ensure_ascii=False), encoding="utf-8"
            )
        except OSError:
            pass
