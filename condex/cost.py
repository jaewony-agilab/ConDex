"""Dollar caps for every paid API call (planner, reference judge). PLAN.md §7.

Caps file (configs/costcaps.yaml, not committed; see configs/costcaps.example.yaml):
  run_cap_usd: 2.0            per process
  global_cap_usd: 50.0        across runs, summed from the global ledger
  global_ledger: ~/condex_ledger.jsonl
  prices: {<model>: {input_per_mtok: ..., output_per_mtok: ...}}
Every call is appended to the run's calls.jsonl and to the global ledger with its purpose.
Callers check `ensure()` before a call; exceeding a cap raises BudgetExceeded (gate: abort).
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, caps_path: str | Path | None = None, calls_path: str | Path | None = None):
        caps_path = Path(caps_path or os.environ.get("CONDEX_COSTCAPS", "configs/costcaps.yaml"))
        if not caps_path.exists():
            raise FileNotFoundError(f"cost caps file {caps_path} missing: copy configs/costcaps.example.yaml and set caps")
        caps = yaml.safe_load(caps_path.read_text())
        self.run_cap = float(caps["run_cap_usd"])
        self.global_cap = float(caps["global_cap_usd"])
        self.prices = caps["prices"]
        self.global_path = Path(os.path.expanduser(caps["global_ledger"]))
        self.calls_path = Path(calls_path) if calls_path else None
        self.run_spent = 0.0
        self.by_purpose: dict[str, float] = {}
        self._lock = threading.Lock()  # eval scripts call the judge from a thread pool

    def global_spent(self) -> float:
        if not self.global_path.exists():
            return 0.0
        return sum(json.loads(line)["usd"] for line in self.global_path.read_text().splitlines() if line.strip())

    def ensure(self) -> None:
        if self.run_spent >= self.run_cap:
            raise BudgetExceeded(f"run cap ${self.run_cap:.2f} reached (${self.run_spent:.4f})")
        if self.global_spent() >= self.global_cap:
            raise BudgetExceeded(f"global cap ${self.global_cap:.2f} reached")

    def price(self, model: str, input_tokens: int, output_tokens: int) -> float:
        if model not in self.prices:
            raise KeyError(f"no price for {model} in the cost caps file")
        p = self.prices[model]
        return input_tokens / 1e6 * p["input_per_mtok"] + output_tokens / 1e6 * p["output_per_mtok"]

    def charge(self, purpose: str, model: str, input_tokens: int, output_tokens: int, **extra) -> float:
        usd = self.price(model, input_tokens, output_tokens)
        with self._lock:
            return self._charge(purpose, model, input_tokens, output_tokens, usd, extra)

    def _charge(self, purpose, model, input_tokens, output_tokens, usd, extra) -> float:
        self.run_spent += usd
        self.by_purpose[purpose] = self.by_purpose.get(purpose, 0.0) + usd
        rec = {"t": time.time(), "purpose": purpose, "model": model, "input_tokens": input_tokens,
               "output_tokens": output_tokens, "usd": usd, "job": os.environ.get("SLURM_JOB_ID", "local"), **extra}
        self.global_path.parent.mkdir(parents=True, exist_ok=True)
        with self.global_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        if self.calls_path:
            self.calls_path.parent.mkdir(parents=True, exist_ok=True)
            with self.calls_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        return usd

    def summary(self) -> dict:
        return {"run_spent_usd": round(self.run_spent, 6), "by_purpose": {k: round(v, 6) for k, v in self.by_purpose.items()}}
