#!/usr/bin/env bash
#SBATCH -J condex_sweep
#SBATCH --array=0-23 --gres=gpu:1 -t 8:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%A_%a.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Phase 0 steps 1+5: executor alone, every task, 20 episodes, 1 seed, recorded (Section 3 format).
# Gives the reproduction numbers (vs published SR) and the task-selection SR in one pass.
source slurm/common.sh
TASK="$(task_at sweep_tasks "${SLURM_ARRAY_TASK_ID:-0}")"
start_policy_server
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --episodes "${EPISODES:-20}" --condition alone \
  --raw "$CONDEX_RAW/phase0" --results "results/phase0/sweep/$TASK"
