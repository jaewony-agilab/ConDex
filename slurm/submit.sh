#!/usr/bin/env bash
# Submit a job with this cluster's flags: slurm/submit.sh cpu|gpu <job.sbatch> [sbatch args]
# The #SBATCH headers carry only portable flags (name, time, --gres, --array, log path). Partition,
# wckey/account and CPU/memory requests come from CONDEX_SBATCH_CPU / CONDEX_SBATCH_GPU in
# <repo>/env.local.sh (examples in env.example.sh). Extra sbatch args go last and win.
#   slurm/submit.sh cpu slurm/install.sbatch --export=ALL,WHAT=rldx
#   TASK=... slurm/submit.sh gpu slurm/phase0_probe.sbatch
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
usage() { echo "usage: slurm/submit.sh cpu|gpu <job.sbatch> [sbatch args]" >&2; exit 2; }
[ $# -ge 2 ] || usage
kind="$1"; job="$(realpath "$2")"; shift 2
[ -f "$job" ] || { echo "no such job script: $job" >&2; exit 2; }

source "$REPO/scripts/load_env.sh" "$REPO"
case "$kind" in
  cpu) var=CONDEX_SBATCH_CPU ;;
  gpu) var=CONDEX_SBATCH_GPU ;;
  *) usage ;;
esac
[ -n "${!var+x}" ] || { echo "$var is not set: add it to $REPO/env.local.sh (see env.example.sh)" >&2; exit 1; }
read -r -a flags <<< "${!var}"

# GPU jobs request their GPU in the header; refuse a mismatch rather than queue the wrong job.
if grep -qE '^#SBATCH .*--gres=gpu' "$job"; then has_gpu=gpu; else has_gpu=cpu; fi
[ "$has_gpu" = "$kind" ] || { echo "$(basename "$job") is a $has_gpu job (by its #SBATCH --gres); submit it as $has_gpu" >&2; exit 2; }

cd "$REPO"   # jobs use SLURM_SUBMIT_DIR as the repo root; logs land here
echo "+ sbatch ${flags[*]} $* $job" >&2
exec sbatch "${flags[@]}" "$@" "$job"
