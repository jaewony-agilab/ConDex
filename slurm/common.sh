# Sourced by every sbatch script. Starts services on this node and cleans them up on exit.
# Headers carry only portable flags; submit with slurm/submit.sh cpu|gpu, which adds this cluster's
# partition/wckey/CPU flags from env.local.sh (CONDEX_SBATCH_CPU / CONDEX_SBATCH_GPU).
set -euo pipefail

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
source "$REPO/scripts/load_env.sh" "$REPO"   # big-disk paths from env.local.sh (gitignored)
# shellcheck disable=SC1091
source "$REPO/.condex_paths"   # RLDX, SIM_PY (install_rldx.sh), VLM_PY (install_judge.sh)
export CONDEX_CONFIG="${CONDEX_CONFIG:-$REPO/configs/gr1.yaml}"
export CONDEX_COSTCAPS="${CONDEX_COSTCAPS:-$REPO/configs/costcaps.yaml}"
export CONDEX_RAW="${CONDEX_RAW:-$HOME/condex_raw}"          # raw episode dirs (set in env.local.sh; never committed)
export CONDEX_SCRATCH="${CONDEX_SCRATCH:-$HOME/condex_scratch}" # datasets, adapters, server logs
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NO_ALBUMENTATIONS_UPDATE=1
JOB="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
LOGS="$CONDEX_SCRATCH/logs/$JOB"
mkdir -p "$LOGS" "$CONDEX_RAW"
PY="${SIM_PY:-python}"

PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

# Fail fast if this node cannot render headless (MuJoCo EGL); only when a GPU is allocated.
check_render() {
  "$SIM_PY" -c '
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>")
r = mujoco.Renderer(m, 8, 8); r.update_scene(mujoco.MjData(m)); r.render(); r.close()' 2>"$LOGS/render_check.log" || {
    echo "MuJoCo EGL rendering fails on $(hostname); see $LOGS/render_check.log (try MUJOCO_EGL_DEVICE_ID=0)"; exit 1; }
}

free_port() { "$PY" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'; }

# start_policy_server -> exports RLDX_PORT
start_policy_server() {
  check_render
  local model; model="$(cfg_get model_path)"
  export RLDX_PORT; RLDX_PORT="$(free_port)"
  ( cd "$RLDX" && uv run python rldx/eval/run_rldx_server.py --model-path "$model" \
      --embodiment-tag GENERAL_EMBODIMENT --use-sim-policy-wrapper --host 127.0.0.1 --port "$RLDX_PORT" ) \
      > "$LOGS/policy_server.log" 2>&1 &
  PIDS+=($!)
  for _ in $(seq 1 120); do
    if "$SIM_PY" -c "
from rldx.policy.server_client import PolicyClient
import sys; sys.exit(0 if PolicyClient(host='127.0.0.1', port=$RLDX_PORT, timeout_ms=2000).ping() else 1)" 2>/dev/null; then
      echo "[i] policy server up on $RLDX_PORT"; return 0; fi
    sleep 5
  done
  echo "policy server did not come up; see $LOGS/policy_server.log"; exit 1
}

# start_vlm_server [ADAPTER_DIR [ADAPTER_NAME]] -> exports CONDEX_VLM_URL (served model: base id, or ADAPTER_NAME)
start_vlm_server() {
  local base="${CONDEX_VLM_MODEL:-Qwen/Qwen3-VL-8B-Instruct}" port lora=()
  [ -n "${1:-}" ] && lora=(--enable-lora --max-lora-rank 64 --lora-modules "${2:-judge_lora}=$1")
  port="$(free_port)"
  "$VLM_PY" -m vllm.entrypoints.openai.api_server --model "$base" --port "$port" --max-model-len 16384 \
      --gpu-memory-utilization "${VLM_GPU_FRAC:-0.45}" --limit-mm-per-prompt '{"image": 8}' "${lora[@]}" \
      > "$LOGS/vlm_server.log" 2>&1 &
  PIDS+=($!)
  export CONDEX_VLM_URL="http://127.0.0.1:$port/v1"
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$port/v1/models" >/dev/null && { echo "[i] vlm up on $port"; return 0; }
    sleep 5
  done
  echo "vlm server did not come up; see $LOGS/vlm_server.log"; exit 1
}

cfg_get() { "$PY" -c "from condex import config as C; print(C.load('$CONDEX_CONFIG')['$1'])"; }
# cfg_list tasks|sweep_tasks|selected_tasks -> space-separated task names
cfg_list() { "$PY" -c "from condex import config as C; print(' '.join(C.task_list(C.load('$CONDEX_CONFIG'), '$1')))"; }
# task_at LIST INDEX -> the task, or exit 0 when the array index is past the end of the list
task_at() {
  local -a tasks; read -r -a tasks <<< "$(cfg_list "$1")"
  [ ${#tasks[@]} -gt 0 ] || { echo "configs: $1 is empty" >&2; exit 1; }
  if [ "$2" -ge ${#tasks[@]} ]; then echo "index $2 >= ${#tasks[@]} tasks in $1: nothing to do" >&2; exit 0; fi
  echo "${tasks[$2]}"
}
