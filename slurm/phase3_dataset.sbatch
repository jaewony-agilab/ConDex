#!/usr/bin/env bash
#SBATCH -J condex_dataset
#SBATCH -t 6:00:00
#SBATCH -o slurm-%x-%j.out
# Submit: slurm/submit.sh cpu slurm/phase3_dataset.sbatch  (partition/wckey/CPUs come from env.local.sh)
# Phase 3 inputs (CPU): LABELS=<labels.jsonl> -> per-condition JSONL + images in $CONDEX_SCRATCH/datasets/<name>.
source slurm/common.sh
NAME="${NAME:-$JOB}"
"$PY" scripts/build_judge_dataset.py "$LABELS" --out "$CONDEX_SCRATCH/datasets/$NAME" ${STYLE:+--style "$STYLE"}
mkdir -p "results/phase3/datasets/$NAME"
for c in plain text overlay; do cp "$CONDEX_SCRATCH/datasets/$NAME/$c/dataset.json" "results/phase3/datasets/$NAME/$c.json"; done
