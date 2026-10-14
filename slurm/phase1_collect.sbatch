#!/usr/bin/env bash
#SBATCH -J condex_collect
#SBATCH --array=0-15 --gres=gpu:1 -t 24:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%A_%a.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Phase 1 step 1: executor-alone rollouts on the selected tasks, fixed seed list (episode ids from START).
source slurm/common.sh
TASK="$(task_at selected_tasks "${SLURM_ARRAY_TASK_ID:-0}")"
start_policy_server
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --episodes "${EPISODES:-150}" --start "${START:-1000}" --condition alone \
  --raw "$CONDEX_RAW/phase1" --results "results/phase1/collect/$TASK"
