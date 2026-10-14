"""Benchmark config (configs/gr1.yaml) and run provenance."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import yaml

from condex.boundary import ClosureConfig

REPO = Path(__file__).resolve().parents[1]


def load(path: str | Path | None = None) -> dict:
    path = Path(path or os.environ.get("CONDEX_CONFIG", REPO / "configs" / "gr1.yaml"))
    cfg = yaml.safe_load(path.read_text())
    cfg["_path"] = str(path)
    return cfg


def task_list(cfg: dict, which: str = "tasks") -> list[str]:
    """tasks | sweep_tasks | repro_tasks | selected_tasks (sweep_tasks defaults to all tasks)."""
    if which == "sweep_tasks" and not cfg.get("sweep_tasks"):
        return list(cfg["tasks"])
    return list(cfg.get(which) or [])


def env_name(cfg: dict, task: str) -> str:
    return f"{cfg['env_prefix']}/{task}"


def closure_configs(cfg: dict) -> dict[str, ClosureConfig]:
    c = dict(cfg["closure"])
    c["flexion_dims"] = tuple(c["flexion_dims"])
    return {h: ClosureConfig(**c) for h in cfg["hands"]}


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def run_info(config_path: str | None = None, **extra) -> dict:
    """Job id, commit, config path: written into every output (PLAN.md §8)."""
    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True).stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, dirty = "unknown", None
    job = os.environ.get("SLURM_JOB_ID", "local")
    if os.environ.get("SLURM_ARRAY_TASK_ID"):
        job += "_" + os.environ["SLURM_ARRAY_TASK_ID"]
    info = {"job_id": job, "commit": commit, "dirty": dirty, "config_path": config_path, "host": os.uname().nodename}
    if config_path and Path(config_path).exists():
        info["config_hash"] = file_hash(config_path)
    info.update(extra)
    return info
