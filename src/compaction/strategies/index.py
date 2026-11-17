"""Hierarchical index of the permanent transcript (the core data structure of
Indexed Retention).

An index has sections ("N") holding subsections ("N.M"); each subsection carries a
short content-bearing summary and one or more inclusive ranges of transcript
message ids. The index is edited only through `Index.apply(ops)` (ops produced by
the indexer model, validated here) and `Index.auto_cover(...)` (harness fallback).

Message ids are "m0001", "m0002", ... (see transcript.py); internally ranges are
stored as integer pairs and rendered back with `mid()`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

DASH = "–"  # en dash used in rendered ranges
AUTO_SECTION_TITLE = "(auto) Unindexed messages"

_MID_RE = re.compile(r"^\s*m?(\d+)\s*$", re.IGNORECASE)
_RANGE_SPLIT_RE = re.compile(r"\s*(?:-|–|—|\.\.|:|\bto\b)\s*")
_SECTION_ID_RE = re.compile(r"^\d+$")
_SUB_ID_RE = re.compile(r"^\d+\.\d+$")

# JSON schema for structured outputs (output_config.format). Structured-output
# schemas require additionalProperties=false and every property in `required`, so
# the op is one flat object whose unused fields are null.
_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}
OPS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "ops": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["add_section", "add_subsection", "edit", "extend"],
                    },
                    "id": _NULLABLE_STR,
                    "section": _NULLABLE_STR,
                    "title": _NULLABLE_STR,
                    "summary": _NULLABLE_STR,
                    "ranges": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["op", "id", "section", "title", "summary", "ranges"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ops"],
    "additionalProperties": False,
}


class IndexLookupError(ValueError):
    """Raised by `Index.resolve/expand` for ids that are not in the index."""

    def __init__(self, unknown: list[str], valid: list[str]) -> None:
        self.unknown, self.valid = unknown, valid
        shown = ", ".join(valid[:60]) + (" ..." if len(valid) > 60 else "")
        super().__init__(
            f"unknown index id(s): {', '.join(unknown)}. "
            f"Valid ids: {shown or '(index is empty)'}. Use section ids like \"2\", "
            "subsection ids like \"2.1\", or message ids/ranges like \"m0012\" / "
            "\"m0012-m0015\"."
        )


# ---------------------------------------------------------------------------
# message-id helpers
# ---------------------------------------------------------------------------
def mid(n: int) -> str:
    """Integer -> transcript message id (same format as Transcript.append)."""
    return f"m{n:04d}"


def mnum(msg_id: str | int) -> int:
    """Message id ("m0012", "12", 12) -> 12. Raises ValueError if malformed."""
    if isinstance(msg_id, bool):
        raise TypeError(f"bad message id {msg_id!r}")
    if isinstance(msg_id, int):
        return msg_id
    m = _MID_RE.match(str(msg_id))
    if not m:
        raise ValueError(f"bad message id {msg_id!r}")
    return int(m.group(1))


def parse_range(r: object) -> tuple[int, int]:
    """Accepts "m0005-m0011", "m0005–m0011", "m0005..m0011", "m0007", ["m0005","m0011"],
    {"start": .., "end": ..}. Returns (start, end) ints (not validated for order)."""
    if isinstance(r, dict):
        return mnum(r.get("start")), mnum(r.get("end", r.get("start")))  # type: ignore[arg-type]
    if isinstance(r, list | tuple):
        if len(r) == 1:
            return parse_range(r[0])
        if len(r) == 2:
            return mnum(r[0]), mnum(r[1])
        raise ValueError(f"bad range {r!r}")
    if isinstance(r, int) and not isinstance(r, bool):
        return r, r
    if isinstance(r, str):
        s = r.strip().strip("[]()")
        parts = [p for p in _RANGE_SPLIT_RE.split(s) if p]
        if len(parts) == 1:
            n = mnum(parts[0])
            return n, n
        if len(parts) == 2:
            return mnum(parts[0]), mnum(parts[1])
    raise ValueError(f"bad range {r!r}")


def merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and merge overlapping/adjacent integer ranges."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(ranges):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def runs(nums: Iterable[int]) -> list[tuple[int, int]]:
    """Consecutive runs of integers -> ranges."""
    return merge_ranges((n, n) for n in nums)


def fmt_range(r: tuple[int, int]) -> str:
    a, b = r
    return mid(a) if a == b else f"{mid(a)}{DASH}{mid(b)}"


def fmt_ranges(ranges: Iterable[tuple[int, int]]) -> str:
    return ", ".join(fmt_range(r) for r in ranges)


# ---------------------------------------------------------------------------
# data model
# ---------------------------------------------------------------------------
@dataclass
class Subsection:
    id: str  # "N.M"
    summary: str
    ranges: list[tuple[int, int]] = field(default_factory=list)  # merged, inclusive

    def message_nums(self) -> set[int]:
        return {n for a, b in self.ranges for n in range(a, b + 1)}


@dataclass
class Section:
    id: str  # "N"
    title: str
    subsections: list[Subsection] = field(default_factory=list)

    def ranges(self) -> list[tuple[int, int]]:
        return merge_ranges(r for s in self.subsections for r in s.ranges)

    def span(self) -> tuple[int, int] | None:
        rs = self.ranges()
        return (rs[0][0], rs[-1][1]) if rs else None


@dataclass
class ApplyReport:
    sections_added: int = 0
    subsections_added: int = 0
    edits: int = 0
    extends: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "sections_added": self.sections_added,
            "subsections_added": self.subsections_added,
            "edits": self.edits,
            "extends": self.extends,
            "op_errors": len(self.errors),
            "op_error_examples": self.errors[:5],
        }


class Index:
    def __init__(self, sections: list[Section] | None = None) -> None:
        self.sections: list[Section] = sections or []

    # ---- lookup -------------------------------------------------------
    def __bool__(self) -> bool:  # an index is truthy iff it has any subsection
        return any(s.subsections for s in self.sections)

    def section(self, sid: str) -> Section | None:
        return next((s for s in self.sections if s.id == sid), None)

    def subsection(self, ssid: str) -> Subsection | None:
        sec = self.section(ssid.split(".", 1)[0]) if "." in ssid else None
        return next((x for x in sec.subsections if x.id == ssid), None) if sec else None

    def all_ids(self) -> list[str]:
        out: list[str] = []
        for s in self.sections:
            out.append(s.id)
            out.extend(x.id for x in s.subsections)
        return out

    def covered(self) -> set[int]:
        """All message numbers covered by any subsection range."""
        return {n for s in self.sections for x in s.subsections for n in x.message_nums()}

    def coverage(self, msg_ids: Iterable[str]) -> tuple[list[str], list[str]]:
        """Split msg_ids into (covered, uncovered), preserving order."""
        cov = self.covered()
        c, u = [], []
        for m in msg_ids:
            (c if mnum(m) in cov else u).append(m)
        return c, u

    # ---- mutation -----------------------------------------------------
    def _next_section_id(self) -> str:
        return str(max((int(s.id) for s in self.sections), default=0) + 1)

    def add_section(self, title: str) -> Section:
        sec = Section(self._next_section_id(), title.strip())
        self.sections.append(sec)
        return sec

    def add_subsection(
        self, sec: Section, summary: str, ranges: list[tuple[int, int]]
    ) -> Subsection:
        n = max((int(x.id.split(".")[1]) for x in sec.subsections), default=0) + 1
        sub = Subsection(f"{sec.id}.{n}", summary.strip(), merge_ranges(ranges))
        sec.subsections.append(sub)
        return sub

    def apply(
        self,
        ops: list[dict],
        *,
        valid: Callable[[int], bool] | None = None,
    ) -> ApplyReport:
        """Apply indexer ops; invalid ops (or invalid ranges inside an op) are skipped
        and reported, never raised. `valid(n)` decides whether message number n may be
        referenced by a range endpoint (default: any n >= 1).

        Ops (flat dicts; unused fields may be absent or null):
          add_section    {title}
          add_subsection {section: existing id | exact title of a section (e.g. one
                          added earlier in the same batch) | null = last section added
                          in this batch, summary, ranges}
          edit           {id: "N" (title) | "N.M" (summary), title/summary}
          extend         {id: "N.M", ranges}
        """
        rep = ApplyReport()
        valid = valid or (lambda n: n >= 1)
        added: list[Section] = []
        for k, op in enumerate(ops if isinstance(ops, list) else []):
            where = f"op {k}"
            try:
                if not isinstance(op, dict):
                    raise TypeError("op is not an object")
                kind = str(op.get("op") or "").strip().lower()
                if kind == "add_section":
                    title = _text(op.get("title")) or _text(op.get("summary"))
                    if not title:
                        raise ValueError("add_section needs a title")
                    added.append(self.add_section(title))
                    rep.sections_added += 1
                elif kind == "add_subsection":
                    sec = self._target_section(op.get("section"), added)
                    summary = _text(op.get("summary")) or _text(op.get("title"))
                    if not summary:
                        raise ValueError("add_subsection needs a summary")
                    rs = self._ranges(op.get("ranges"), valid, rep, where)
                    if not rs:
                        raise ValueError("add_subsection has no valid ranges")
                    self.add_subsection(sec, summary, rs)
                    rep.subsections_added += 1
                elif kind == "edit":
                    oid = _text(op.get("id"))
                    if _SUB_ID_RE.match(oid):
                        sub = self.subsection(oid)
                        new = _text(op.get("summary")) or _text(op.get("title"))
                        if sub is None or not new:
                            raise ValueError(f"edit: unknown subsection {oid!r} or empty summary")
                        sub.summary = new
                    else:
                        sec = self.section(oid)
                        new = _text(op.get("title")) or _text(op.get("summary"))
                        if sec is None or not new:
                            raise ValueError(f"edit: unknown section {oid!r} or empty title")
                        sec.title = new
                    rep.edits += 1
                elif kind == "extend":
                    oid = _text(op.get("id"))
                    sub = self.subsection(oid)
                    if sub is None:
                        raise ValueError(f"extend: unknown subsection {oid!r}")
                    rs = self._ranges(op.get("ranges"), valid, rep, where)
                    if not rs:
                        raise ValueError("extend has no valid ranges")
                    sub.ranges = merge_ranges(sub.ranges + rs)
                    rep.extends += 1
                else:
                    raise ValueError(f"unknown op {kind!r}")
            except (ValueError, TypeError) as e:
                rep.errors.append(f"{where}: {e}")
        # Sections added in this batch that never received a subsection are dropped.
        for sec in added:
            if not sec.subsections and sec in self.sections:
                self.sections.remove(sec)
                rep.sections_added -= 1
                rep.errors.append(f"section {sec.title!r} had no subsections; dropped")
        return rep

    def _target_section(self, ref: object, added: list[Section]) -> Section:
        r = _text(ref)
        if not r:
            if added:
                return added[-1]
            raise ValueError("add_subsection needs a section")
        r = r.lstrip("#§ ").rstrip(".")
        if _SECTION_ID_RE.match(r) and (sec := self.section(r)):
            return sec
        low = r.strip().lower()
        for sec in list(added)[::-1] + self.sections[::-1]:
            if sec.title.strip().lower() == low:
                return sec
        raise ValueError(f"unknown section {ref!r}")

    @staticmethod
    def _ranges(
        raw: object, valid: Callable[[int], bool], rep: ApplyReport, where: str
    ) -> list[tuple[int, int]]:
        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        # Each item is one range ("m0005-m0011", ["m0005","m0011"], "m0007"). A flat
        # ["m0005", "m0011"] list is read as two single messages, not a range.
        out = []
        for r in items:
            try:
                a, b = parse_range(r)
            except (ValueError, TypeError) as e:
                rep.errors.append(f"{where}: {e}")
                continue
            if a > b:
                rep.errors.append(f"{where}: range {fmt_range((b, a))} reversed (start > end)")
                continue
            if not (valid(a) and valid(b)):
                rep.errors.append(f"{where}: range {mid(a)}-{mid(b)} references messages outside the allowed set")
                continue
            out.append((a, b))
        return out

    def auto_cover(
        self, nums: list[int], describe: Callable[[list[int]], str], max_run: int = 6
    ) -> int:
        """Cover leftover message numbers with "(auto)" subsections (one per run of
        nearby messages, at most `max_run` each) under the auto section. Returns
        the number of messages covered."""
        nums = sorted(set(nums) - self.covered())
        if not nums:
            return 0
        sec = next((s for s in self.sections if s.title == AUTO_SECTION_TITLE), None)
        if sec is None:
            sec = self.add_section(AUTO_SECTION_TITLE)
        group: list[int] = []
        groups: list[list[int]] = []
        for n in nums:
            if group and (n - group[-1] > 2 or len(group) >= max_run):
                groups.append(group)
                group = []
            group.append(n)
        groups.append(group)
        for g in groups:
            self.add_subsection(sec, "(auto) " + describe(g), [(g[0], g[-1])])
        return len(nums)

    # ---- resolution ---------------------------------------------------
    def expand(self, ids: list[str]) -> list[tuple[str, list[int]]]:
        """Requested ids -> ordered groups (label, message numbers). Sections expand
        to their subsections; raw message ids/ranges become their own group. Message
        numbers are not deduplicated across groups here. Raises IndexLookupError."""
        groups: list[tuple[str, list[int]]] = []
        unknown: list[str] = []
        for raw in ids:
            i = str(raw).strip().lstrip("#§ ").rstrip(".")
            if _SUB_ID_RE.match(i) and (sub := self.subsection(i)):
                groups.append((self._sub_label(sub), sorted(sub.message_nums())))
            elif _SECTION_ID_RE.match(i) and (sec := self.section(i)):
                for sub in sec.subsections:
                    groups.append((self._sub_label(sub), sorted(sub.message_nums())))
            elif i.lower().startswith("m"):
                try:
                    a, b = parse_range(i)
                except (ValueError, TypeError):
                    unknown.append(str(raw))
                    continue
                a, b = min(a, b), max(a, b)
                groups.append((f"messages {fmt_range((a, b))}", list(range(a, b + 1))))
            else:
                unknown.append(str(raw))
        if unknown:
            raise IndexLookupError(unknown, self.all_ids())
        return groups

    def resolve(self, ids: list[str]) -> list[str]:
        """Section / subsection / message ids -> sorted, de-duplicated message ids."""
        nums = {n for _, g in self.expand(ids) for n in g}
        return [mid(n) for n in sorted(nums)]

    def _sub_label(self, sub: Subsection) -> str:
        sec = self.section(sub.id.split(".")[0])
        title = f" ({sec.title})" if sec else ""
        return f"{sub.id} {sub.summary}{title}  [{fmt_ranges(sub.ranges)}]"

    # ---- rendering / serialization -------------------------------------
    def render(self, *, max_summary_chars: int | None = None, sections_only: bool = False) -> str:
        if not self.sections:
            return "# Index\n(empty: nothing has been indexed yet)"
        lines = ["# Index"]
        for s in self.sections:
            span = s.span()
            lines.append(f"## {s.id}. {s.title}" + (f"  [{fmt_range(span)}]" if span else ""))
            if sections_only:
                ids = ", ".join(x.id for x in s.subsections)
                lines.append(f"(subsections: {ids})")
                continue
            for x in s.subsections:
                summ = x.summary
                if max_summary_chars and len(summ) > max_summary_chars:
                    summ = summ[: max_summary_chars - 1].rstrip() + "…"
                lines.append(f"### {x.id} {summ}  [{fmt_ranges(x.ranges)}]")
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "sections": [
                {
                    "id": s.id,
                    "title": s.title,
                    "subsections": [
                        {
                            "id": x.id,
                            "summary": x.summary,
                            "ranges": [[mid(a), mid(b)] for a, b in x.ranges],
                        }
                        for x in s.subsections
                    ],
                }
                for s in self.sections
            ]
        }

    @classmethod
    def from_json(cls, data: dict | str) -> Index:
        if isinstance(data, str):
            data = json.loads(data)
        return cls(
            [
                Section(
                    s["id"],
                    s["title"],
                    [
                        Subsection(
                            x["id"],
                            x["summary"],
                            [(mnum(a), mnum(b)) for a, b in x["ranges"]],
                        )
                        for x in s.get("subsections", [])
                    ],
                )
                for s in data.get("sections", [])
            ]
        )


def _text(v: object) -> str:
    return v.strip() if isinstance(v, str) else ("" if v is None else str(v).strip())


# ---------------------------------------------------------------------------
# robust extraction of {"ops": [...]} from model text
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_ops(text: str) -> list[dict] | None:
    """Extract the ops list from model output. Accepts pure JSON, fenced JSON, or
    JSON embedded in prose; either {"ops": [...]} or a bare [...] list of ops.
    Returns None if nothing usable is found."""
    if not text:
        return None
    candidates = [text.strip()] + [m.strip() for m in _FENCE_RE.findall(text)]
    for c in candidates:
        try:
            got = _ops_from(json.loads(c))
        except (json.JSONDecodeError, ValueError):
            continue
        if got is not None:
            return got
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        got = _ops_from(obj)
        if got is not None:
            return got
    return None


def _ops_from(obj: object) -> list[dict] | None:
    if isinstance(obj, dict) and isinstance(obj.get("ops"), list):
        ops = obj["ops"]
    elif isinstance(obj, list) and obj and all(isinstance(o, dict) and "op" in o for o in obj):
        ops = obj
    else:
        return None
    ops = [o for o in ops if isinstance(o, dict)]
    return ops
