#!/usr/bin/env bash
#SBATCH -J condex_install
#SBATCH -p cpu -t 6:00:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%j.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# One-time install from a clean clone, without srun: sbatch slurm/install.sbatch
# Needs internet on the node (GitHub, PyPI, Hugging Face) and env.local.sh for the big-disk paths.
# WHAT=rldx | judge | all (default all).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
case "${WHAT:-all}" in
  rldx)  bash scripts/install_rldx.sh ;;
  judge) bash scripts/install_judge.sh ;;
  all)   bash scripts/install_rldx.sh && bash scripts/install_judge.sh ;;
  *) echo "WHAT must be rldx | judge | all"; exit 1 ;;
esac
