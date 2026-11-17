"""Deterministic offline LLM for plumbing tests (``--mock``).

MOCK RESULTS ARE PLUMBING TESTS, NOT FINDINGS. The "agent" is a scripted policy
(:class:`~compaction.eval.tasks.MockAgentPolicy`) that plays the investigation
perfectly and cannot actually remember anything; per-strategy "forgetfulness" is
a fixed knob below, chosen only so reports are not all identical.
"""

from __future__ import annotations

import json
import re

from compaction.eval.tasks import MockAgentPolicy, Task
from compaction.llm import LLMResponse, ScriptedLLM, text_response, tool_response

# strategy name -> MockAgentPolicy kwargs (arbitrary; NOT a model of real behaviour)
MOCK_PROFILES: dict[str, dict] = {
    "truncate": {"forget_early": True, "error_rate": 0.1},
    "summary": {"error_rate": 0.3},
    "indexed": {},
}

_IDS_RE = re.compile(r"^Ids to index: (.*)$", re.MULTILINE)
_SECTION_RE = re.compile(r"^## \d+\.", re.MULTILINE)


def _runs(nums: list[int]) -> list[list[str]]:
    out: list[list[int]] = []
    for n in sorted(nums):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return [[f"m{a:04d}", f"m{b:04d}"] for a, b in out]


def _last_user_text(messages: list[dict]) -> str:
    if not messages:
        return ""
    c = messages[-1].get("content")
    if isinstance(c, str):
        return c
    parts = []
    for b in c or []:
        if b.get("type") == "text":
            parts.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            parts.append(str(b.get("content", "")))
    return "\n".join(parts)


class MockLLM(ScriptedLLM):
    """ScriptedLLM driven by a MockAgentPolicy for ``purpose == "agent"`` and by
    canned text/JSON for compactor calls."""

    def __init__(self, task: Task, strategy: str, **policy_kw) -> None:
        kw = {**MOCK_PROFILES.get(strategy, {}), **policy_kw}
        self.policy = MockAgentPolicy(task, **kw)
        self.strategy = strategy
        self._indexer_calls = 0
        self._fetched = False
        super().__init__(self._respond)

    def _respond(self, messages: list[dict], tools: list[dict] | None, purpose: str) -> LLMResponse:
        if purpose == "agent":
            return self._agent(messages, tools or [])
        text = _last_user_text(messages)
        if "Ids to index:" in text:
            return self._indexer(text)
        return text_response(
            "MOCK SUMMARY. Task: investigation of operations files. Read files in order; "
            "key identifiers retained verbatim: (mock). Open questions: none."
        )

    def _agent(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        obs = _last_user_text(messages)
        if "The investigation phase is over." in obs:  # questions revealed (maybe forced): go answer
            self.policy._i = len(self.policy._script)
        act = self.policy.next_action(obs)
        if act is None:
            return text_response("Nothing left to do.")
        name, inp = act
        names = {t["name"] for t in tools}
        if name == "submit_answer" and "fetch_messages" in names and not self._fetched:
            # exercise the retrieval path once, then submit on the next step
            self._fetched = True
            self.policy._submitted = False
            return tool_response("fetch_messages", {"ids": ["1"]}, text="Let me re-check my index.")
        return tool_response(name, inp)

    def _indexer(self, text: str) -> LLMResponse:
        self._indexer_calls += 1
        m = _IDS_RE.search(text)
        nums = [int(x[1:]) for x in re.findall(r"m\d{4}", m.group(1))] if m else []
        cur = text.split("</current_index>", 1)[0]
        sec = len(_SECTION_RE.findall(cur)) + 1
        if self._indexer_calls % 2 == 0 and len(nums) > 3:
            nums = nums[:-2]  # leave some uncovered -> exercises the auto-coverage path
        if not nums:
            return text_response('{"ops": []}')
        ops = [
            {"op": "add_section", "title": f"Mock topic {sec}"},
            {
                "op": "add_subsection",
                "section": str(sec),
                "summary": f"Mock work on messages {nums[0]}-{nums[-1]}",
                "ranges": _runs(nums),
            },
        ]
        return text_response(json.dumps({"ops": ops}))
