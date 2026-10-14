"""Planner: Claude Haiku 5.5, called at episode start and on replan only (PLAN.md §2, §5).

It never rewrites the executor's instruction: RLDX-1-FT-GR1 was fine-tuned on the benchmark's own
instruction strings. At start it lists the expected hand events (their text becomes the boundary
`subtask` shown to the judge). On replan it sees the judge's reason and the episode history and
chooses: resume | recover_then_resume (with a hand) | abort.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from condex.cost import Ledger

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {"type": "array", "items": {
            "type": "object",
            "properties": {"hand": {"type": "string", "enum": ["left", "right"]},
                           "type": {"type": "string", "enum": ["grasp", "release"]},
                           "subtask": {"type": "string"}},
            "required": ["hand", "type", "subtask"], "additionalProperties": False}},
    },
    "required": ["steps"],
    "additionalProperties": False,
}

REPLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["resume", "recover_then_resume", "abort"]},
        "hand": {"type": "string", "enum": ["left", "right", "none"]},
        "why": {"type": "string"},
    },
    "required": ["decision", "hand", "why"],
    "additionalProperties": False,
}

SYSTEM = ("You plan for a humanoid robot with two dexterous hands doing a tabletop pick-and-place task. "
          "A learned policy executes the task instruction; you cannot change the instruction. "
          "A judge checks every grasp and release.")


@dataclass
class Plan:
    steps: list[dict] = field(default_factory=list)

    def subtask_for(self, hand: str, btype: str, done: list[dict]) -> str:
        """The first planned step of this hand/type not yet matched by a successful boundary."""
        used = sum(1 for d in done if d["hand"] == hand and d["type"] == btype and d.get("outcome") == "success")
        matches = [s for s in self.steps if s["hand"] == hand and s["type"] == btype]
        return matches[min(used, len(matches) - 1)]["subtask"] if matches else ""


class Planner:
    def __init__(self, ledger: Ledger, model: str = "claude-haiku-5-5", effort: str = "low"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.ledger, self.model, self.effort = ledger, model, effort

    def _call(self, purpose: str, prompt: str, schema: dict) -> dict:
        self.ledger.ensure()
        r = self.client.messages.create(
            model=self.model, max_tokens=4096, system=SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
        )
        self.ledger.charge(purpose, self.model, r.usage.input_tokens, r.usage.output_tokens)
        if r.stop_reason == "refusal":
            return {}
        return json.loads(next((b.text for b in r.content if b.type == "text"), "{}"))

    def plan(self, instruction: str) -> Plan:
        d = self._call("plan", f"Instruction: {instruction}\nList the hand events (grasp/release, which hand) "
                       "the robot should go through, each with a short subtask description.", PLAN_SCHEMA)
        return Plan(d.get("steps", []))

    def replan(self, instruction: str, reason: str, history: list[dict]) -> dict:
        hist = "\n".join(f"- {h['hand']} {h['type']}: {h.get('outcome')} ({h.get('reason', '')})" for h in history[-8:])
        d = self._call("replan", f"Instruction: {instruction}\nChecks so far:\n{hist}\nThe last check triggered a replan: "
                       f"{reason}\nChoose: resume the policy, recover_then_resume (open the given hand and back off "
                       "first), or abort if the task can no longer succeed.", REPLAN_SCHEMA)
        return d or {"decision": "abort", "hand": "none", "why": "planner refused"}
