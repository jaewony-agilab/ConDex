"""Isolation (PLAN.md §7, Phase 0 step 3): privileged state only in the recorder and the oracle.

* Only scripts/ and condex/privileged/ import condex.privileged.
* Importing the executor client, harness, judge, renderer or dataset code loads no privileged module.
* Outside condex/privileged, nothing reads the record's privileged `events`.
* The executor itself is RLDX-1's server in its own venv/process; condex is not installed there
  (scripts/install_rldx.sh installs condex into the sim venv only), so it cannot import the recorder.
"""

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "condex"


def imports(path: Path) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
    return out


def test_only_scripts_and_privileged_import_privileged():
    offenders = [str(p.relative_to(REPO)) for p in PKG.rglob("*.py")
                 if "privileged" not in p.parts and any(m.startswith("condex.privileged") for m in imports(p))]
    assert not offenders, offenders


def test_non_privileged_code_never_reads_events():
    offenders = []
    for p in PKG.rglob("*.py"):
        if "privileged" in p.parts or p.name == "episode.py":
            continue
        for node in ast.walk(ast.parse(p.read_text())):
            if isinstance(node, ast.Attribute) and node.attr == "events":
                offenders.append(f"{p.relative_to(REPO)}:{node.lineno}")
    assert not offenders, offenders


def test_executor_and_judge_paths_load_no_privileged_module():
    mods = ["condex.sim.executor", "condex.harness", "condex.judge.prompt", "condex.judge.backends", "condex.render",
            "condex.survival", "condex.train.dataset", "condex.train.lora", "condex.planner", "condex.gate", "condex.recovery"]
    code = ("import sys; " + "; ".join(f"import {m}" for m in mods) +
            "; bad=[m for m in sys.modules if m.startswith('condex.privileged')]; print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
