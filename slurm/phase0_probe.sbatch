#!/usr/bin/env bash
#SBATCH -J condex_probe
#SBATCH --gres=gpu:1 -t 2:00:00
#SBATCH -o slurm-%x-%j.out
# Submit: slurm/submit.sh gpu slurm/phase0_probe.sbatch  (partition/wckey/CPUs come from env.local.sh)
# Phase 0 step 2 + calibration: sensors, step-time overhead, command/force percentiles, overlay frames.
source slurm/common.sh
start_policy_server
"$SIM_PY" scripts/probe_sensors.py --task "${TASK:-PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env}" \
  --results "results/phase0/probe/$JOB" --raw "$CONDEX_RAW/probe/$JOB"
