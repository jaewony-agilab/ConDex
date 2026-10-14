#!/usr/bin/env bash
#SBATCH -J condex_collect
#SBATCH --array=0-15 --gres=gpu:1 -t 24:00:00
#SBATCH -o slurm-%x-%A_%a.out
# Submit: slurm/submit.sh gpu slurm/phase1_collect.sbatch  (partition/wckey/CPUs come from env.local.sh)
# Phase 1 step 1: executor-alone rollouts on the selected tasks, fixed seed list (episode ids from START).
source slurm/common.sh
TASK="$(task_at selected_tasks "${SLURM_ARRAY_TASK_ID:-0}")"
start_policy_server
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --episodes "${EPISODES:-150}" --start "${START:-1000}" --condition alone \
  --raw "$CONDEX_RAW/phase1" --results "results/phase1/collect/$TASK"
