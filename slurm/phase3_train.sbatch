#!/usr/bin/env bash
#SBATCH -J condex_lora
#SBATCH --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G -t 24:00:00
#SBATCH -o slurm-%x-%j.out
# Phase 3 step 2: DATASET=<name> CONDITION=plain|text|overlay SEED=0|1. Adapter -> $CONDEX_SCRATCH/adapters.
source slurm/common.sh
OUT="$CONDEX_SCRATCH/adapters/$DATASET/${CONDITION}_s${SEED:-0}"
"$VLM_PY" scripts/train_judge.py "$CONDEX_SCRATCH/datasets/$DATASET/$CONDITION" --seed "${SEED:-0}" --out "$OUT"
mkdir -p "results/phase3/train/$DATASET"; cp "$OUT/train_summary.json" "results/phase3/train/$DATASET/${CONDITION}_s${SEED:-0}.json"
