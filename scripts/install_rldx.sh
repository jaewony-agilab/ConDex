#!/usr/bin/env bash
# Unattended install from a clean clone (Linux, CUDA 12.x, uv >= 0.8.4). CPU node is fine:
#   sbatch slurm/install.sbatch   (or: bash scripts/install_rldx.sh on any node with internet)
# 1. RLDX-1 at a pinned commit + its model venv (inference server; condex is NOT installed there)
# 2. GR-1 Tabletop sim venv via RLDX-1's own setup script (robosuite 1.5.1, robocasa-gr1-tabletop-tasks)
# 3. condex[llm] into the sim venv (harness, sensors, recorder, judge/planner clients)
# 4. RLWRLD/RLDX-1-FT-GR1 into the HF cache
# Rendering needs EGL and is checked at the start of every GPU job (slurm/common.sh), not here.
# Paths (HF_HOME, UV_CACHE_DIR, CONDEX_THIRD_PARTY, ...) come from <repo>/env.local.sh (see env.example.sh).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/load_env.sh" "$REPO"
TP="${CONDEX_THIRD_PARTY:-$REPO/third_party}"
RLDX_COMMIT="cf67c31f7f168ebbca6d06e8c3c23914cb94435c"   # RLWRLD/RLDX-1 main, 2026-07-31
RLDX="$TP/RLDX-1"
command -v uv >/dev/null || { echo "uv not found (need >= 0.8.4)"; exit 1; }
mkdir -p "$TP"

[ -d "$RLDX/.git" ] || git clone https://github.com/RLWRLD/RLDX-1.git "$RLDX"
git -C "$RLDX" fetch --quiet origin "$RLDX_COMMIT" || true
git -C "$RLDX" checkout --quiet "$RLDX_COMMIT"
echo "[i] RLDX-1 at $(git -C "$RLDX" rev-parse HEAD)"

# 1. model venv
( cd "$RLDX" && uv sync --python 3.10 && uv pip install -e . )
( cd "$RLDX" && uv run python -c "import rldx; print('rldx', getattr(rldx, '__version__', '?'))" )

# 2. sim venv: RLDX-1's script (it rebuilds the venv). Its last step makes an env with rendering
#    (EGL); on a CPU node only that step may fail, which we accept after checking the env builds.
SIM_DIR="rldx/eval/sim/robocasa-gr1-tabletop-tasks"
SIM_PY="$RLDX/$SIM_DIR/robocasa_uv/.venv/bin/python"
SUB="$RLDX/external_dependencies/robocasa-gr1-tabletop-tasks"
if ( cd "$RLDX" && INSTALL_FLASH_ATTN=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl bash "$SIM_DIR/setup_RoboCasaGR1TabletopTasks.sh" ); then
  echo "[i] sim setup OK including its render check"
else
  [ -x "$SIM_PY" ] || { echo "sim venv missing at $SIM_PY: install failed before the render check"; exit 1; }
  "$SIM_PY" - <<'PY' || { echo "env does not build without a renderer: real install failure"; exit 1; }
import gymnasium as gym, robocasa, robosuite  # noqa: F401
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401
env = gym.make("gr1_unified/PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_Env", enable_render=False)
env.reset(seed=0)
print("env OK without renderer; robosuite", robosuite.__version__)
PY
  echo "[i] only the render check failed (expected on a CPU node)"
fi
for patch in "$RLDX"/run_scripts/eval/gr1_tabletop/patches/*.patch; do
  [ -f "$patch" ] || continue
  if git -C "$SUB" apply --check "$patch" 2>/dev/null; then git -C "$SUB" apply "$patch"; echo "[i] applied $(basename "$patch")"
  elif git -C "$SUB" apply --reverse --check "$patch" 2>/dev/null; then echo "[i] already applied $(basename "$patch")"
  else echo "patch $(basename "$patch") neither applies nor is applied"; exit 1; fi
done

# 3. condex into the sim venv (not [dev]: it would pull a newer mujoco than the sim pins)
uv pip install --python "$SIM_PY" -e "$REPO[llm]" pytest
"$SIM_PY" -c "import condex, anthropic, openai, mujoco, pyarrow; print('condex OK; mujoco', mujoco.__version__)"
uv pip freeze --python "$SIM_PY" > "$TP/sim_gr1_freeze.txt"

# 4. checkpoint
( cd "$RLDX" && uv run python -c "from huggingface_hub import snapshot_download as s; print(s('RLWRLD/RLDX-1-FT-GR1'))" )

P="$REPO/.condex_paths"; touch "$P"
upsert() { grep -v "^$1=" "$P" > "$P.tmp" || true; echo "$1=$2" >> "$P.tmp"; mv "$P.tmp" "$P"; }
upsert RLDX "$RLDX"
upsert SIM_PY "$SIM_PY"
echo "[i] done. $P:"; cat "$P"
