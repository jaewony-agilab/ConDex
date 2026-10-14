#!/usr/bin/env bash
#SBATCH -J condex_labels
#SBATCH -t 2:00:00
#SBATCH -o slurm-%x-%j.out
# Submit: slurm/submit.sh cpu slurm/phase2_labels.sbatch  (partition/wckey/CPUs come from env.local.sh)
# Phase 2 (CPU): labels from the recorder output; labels.jsonl stays in scratch, the summary is committed.
source slurm/common.sh
"$PY" scripts/build_labels.py "$CONDEX_RAW/phase1" --out "$CONDEX_SCRATCH/labels/$JOB"
mkdir -p "results/phase2/$JOB"; cp "$CONDEX_SCRATCH/labels/$JOB/labels_summary.json" "results/phase2/$JOB/"
echo "labels: $CONDEX_SCRATCH/labels/$JOB/labels.jsonl"
