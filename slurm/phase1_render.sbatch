#!/usr/bin/env bash
#SBATCH -J condex_render
#SBATCH --cpus-per-task=8 --mem=32G -t 4:00:00
#SBATCH -o slurm-%x-%j.out
# Phase 1 steps 2-4 (CPU): render 20 boundaries/task with STYLE, survival check, dataset statistics.
source slurm/common.sh
STYLE="${STYLE:-configs/style/default.yaml}"
"$PY" scripts/render_check.py "$CONDEX_RAW/phase1" --style "$STYLE" --out "results/phase1/render/$JOB"
"$PY" scripts/dataset_stats.py "$CONDEX_RAW/phase1" --out "results/phase1/stats/$JOB/stats.json" ${LABELS:+--labels "$LABELS"}
