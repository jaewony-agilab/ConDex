#!/usr/bin/env bash
#SBATCH -J condex_eval
#SBATCH --gres=gpu:1 -t 6:00:00
#SBATCH -o slurm-%x-%j.out
# Submit: slurm/submit.sh gpu slurm/phase3_eval.sbatch  (partition/wckey/CPUs come from env.local.sh)
# Phase 3: score a judge. DATASET=<name> CONDITION=plain|text|overlay JUDGE=vlm|claude SPLIT=heldout
# ADAPTER=<adapter dir> (optional: serve a LoRA instead of the frozen model).
source slurm/common.sh
D="$CONDEX_SCRATCH/datasets/$DATASET/$CONDITION"
MODEL=""
if [ "${JUDGE:-vlm}" = vlm ]; then
  VLM_GPU_FRAC=0.9 start_vlm_server "${ADAPTER:-}" judge_lora
  [ -n "${ADAPTER:-}" ] && MODEL=judge_lora
fi
TAG="${JUDGE:-vlm}${ADAPTER:+_lora}"
"$VLM_PY" scripts/eval_judge.py "$D" --split "${SPLIT:-heldout}" --judge "${JUDGE:-vlm}" ${MODEL:+--vlm-model "$MODEL"} \
  --out "results/phase3/eval/$DATASET/${CONDITION}_${TAG}_$JOB"
