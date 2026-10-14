# Sourced by scripts/install_*.sh and slurm/common.sh: loads <repo>/env.local.sh if present.
# Usage: source scripts/load_env.sh <repo>
_condex_env="$1/env.local.sh"
if [ -f "$_condex_env" ]; then
  # shellcheck disable=SC1090
  source "$_condex_env"
  case "${CONDEX_RAW:-}${CONDEX_SCRATCH:-}${HF_HOME:-}" in
    */path/to/*) echo "$_condex_env still has the /path/to placeholder: set BIG to your big disk"; exit 1 ;;
  esac
  mkdir -p ${HF_HOME:+"$HF_HOME"} ${UV_CACHE_DIR:+"$UV_CACHE_DIR"} ${CONDEX_RAW:+"$CONDEX_RAW"} ${CONDEX_SCRATCH:+"$CONDEX_SCRATCH"}
else
  echo "[i] no env.local.sh: big-disk paths default to \$HOME (cp env.example.sh env.local.sh to change)"
fi
unset _condex_env
