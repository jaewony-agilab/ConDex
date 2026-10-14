#!/usr/bin/env bash
#SBATCH -J condex_probe
#SBATCH --gres=gpu:1 -t 2:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%j.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Phase 0 step 2 + calibration: sensors, step-time overhead, command/force percentiles, overlay frames.
source slurm/common.sh
start_policy_server
"$SIM_PY" scripts/probe_sensors.py --task "${TASK:-PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}" \
  --results "results/phase0/probe/$JOB" --raw "$CONDEX_RAW/probe/$JOB"
