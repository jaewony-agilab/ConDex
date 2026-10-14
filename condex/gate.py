"""Gate rule (PLAN.md §5) - code, not prompt.

| judge                                   | gate                                             |
|-----------------------------------------|--------------------------------------------------|
| success                                 | continue (resets that hand's retries)            |
| fail, valid                             | scripted recovery, at most `max_retries` per hand |
| fail, invalid / unclear twice / cap hit | replan; the planner gets the reason              |
| budget cap                              | abort (BudgetExceeded, handled by the harness)   |
A single unclear continues. Ablation `recovery_enabled=False`: fail-valid aborts.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

from condex.judge.schema import Verdict


class Action(str, Enum):
    CONTINUE = "continue"
    RECOVER = "recover"
    REPLAN = "replan"
    ABORT = "abort"


class Gate:
    def __init__(self, max_retries: int = 2, recovery_enabled: bool = True):
        self.max_retries = max_retries
        self.recovery_enabled = recovery_enabled
        self.reset()

    def reset(self) -> None:
        self.retries: dict[str, int] = defaultdict(int)
        self.unclear = 0

    def decide(self, hand: str, v: Verdict) -> tuple[Action, str]:
        if v.outcome == "success":
            self.unclear = 0
            self.retries[hand] = 0
            return Action.CONTINUE, ""
        if v.outcome == "unclear":
            self.unclear += 1
            if self.unclear >= 2:
                self.unclear = 0
                return Action.REPLAN, f"judge unclear twice: {v.reason}"
            return Action.CONTINUE, ""
        self.unclear = 0
        if v.plan_still_valid == "no":
            return Action.REPLAN, v.reason
        if not self.recovery_enabled:
            return Action.ABORT, f"fail with recovery disabled: {v.reason}"
        if self.retries[hand] >= self.max_retries:
            return Action.REPLAN, f"retry cap {self.max_retries} hit on the {hand} hand: {v.reason}"
        self.retries[hand] += 1
        return Action.RECOVER, v.reason
