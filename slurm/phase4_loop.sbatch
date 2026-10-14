#!/usr/bin/env bash
#SBATCH -J condex_loop
#SBATCH --array=0-15 --gres=gpu:1 -t 24:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%A_%a.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Phase 4: CONDITION=alone|plain|overlay|oracle|text JUDGE=vlm|claude SEED=42 EPISODES=20 (screening) / 50 (full)
# ADAPTER=<adapter dir> to judge with a LoRA. EXTRA="--no-recovery" | "--no-prev-frames" for ablations.
source slurm/common.sh
TASK="$(task_at selected_tasks "${SLURM_ARRAY_TASK_ID:-0}")"
start_policy_server
MODEL=""
if [[ "${CONDITION:-alone}" =~ ^(plain|text|overlay)$ && "${JUDGE:-vlm}" = vlm ]]; then
  start_vlm_server "${ADAPTER:-}" judge_lora
  [ -n "${ADAPTER:-}" ] && MODEL=judge_lora
fi
STAGE="${STAGE:-screen}"
ABL="$(echo "${EXTRA:-}" | tr -d " -")"   # e.g. norecovery, noprevframes
"$SIM_PY" scripts/run_episodes.py --task "$TASK" --episodes "${EPISODES:-20}" --seed "${SEED:-42}" \
  --condition "${CONDITION:-alone}" --judge "${JUDGE:-vlm}" ${MODEL:+--vlm-model "$MODEL"} ${EXTRA:-} \
  --raw "$CONDEX_RAW/phase4_$STAGE" --results "results/phase4/$STAGE/${CONDITION:-alone}${ADAPTER:+_lora}${ABL:+_$ABL}_s${SEED:-42}/$TASK"
