"""Judge output (PLAN.md §5): one JSON object."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass

OUTCOMES = ("success", "fail", "unclear")
VALID = ("yes", "no")

SCHEMA = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": list(OUTCOMES)},
        "plan_still_valid": {"type": "string", "enum": list(VALID)},
        "reason": {"type": "string"},
    },
    "required": ["outcome", "plan_still_valid", "reason"],
    "additionalProperties": False,
}


@dataclass
class Verdict:
    outcome: str
    plan_still_valid: str
    reason: str
    source: str = ""  # judge name
    parse_ok: bool = True

    def to_json(self) -> str:
        """The training target and the logged form: exactly the three schema fields, fixed order."""
        return json.dumps({"outcome": self.outcome, "plan_still_valid": self.plan_still_valid, "reason": self.reason})

    def as_dict(self) -> dict:
        return asdict(self)


def parse(text: str, source: str = "") -> Verdict:
    """Lenient: take the first {...} block; anything invalid becomes `unclear` with parse_ok False."""
    m = re.search(r"\{.*\}", text or "", re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        d = {}
    o, v = d.get("outcome"), d.get("plan_still_valid")
    if o in OUTCOMES and v in VALID:
        return Verdict(o, v, str(d.get("reason", "")).strip(), source)
    return Verdict("unclear", "yes", f"unparseable judge output: {(text or '')[:120]!r}", source, parse_ok=False)
