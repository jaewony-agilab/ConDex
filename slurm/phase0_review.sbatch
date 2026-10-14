#!/usr/bin/env bash
#SBATCH -J condex_review
#SBATCH -p cpu -t 2:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%j.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Phase 0 steps 4-5 (CPU): boundary sheets for 10 episodes, failure GIFs + CSV to label, SR tables.
source slurm/common.sh
"$PY" scripts/check_boundaries.py "$CONDEX_RAW/phase0" --episodes 10 --out "results/phase0/boundaries/$JOB"
"$PY" scripts/failure_review.py "$CONDEX_RAW/phase0" --out "results/phase0/failures/$JOB"
"$PY" scripts/summarize.py select results/phase0/sweep | tee "results/phase0/failures/$JOB/select.txt"
