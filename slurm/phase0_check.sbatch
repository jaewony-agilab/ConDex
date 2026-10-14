#!/usr/bin/env bash
#SBATCH -J condex_check
#SBATCH --gres=gpu:1 -t 0:30:00
#SBATCH --wckey=project-short-name:sub_4dpdata
#SBATCH -o slurm-%x-%j.out
# Cluster rules: --wckey is required; no CPU/memory requests (per-GPU defaults apply).
# Post-install sanity on a GPU node: MuJoCo EGL rendering + the test suite in the sim venv
# (the projection-vs-render test only runs where EGL works). Output: results/phase0/check/<job>/.
source slurm/common.sh
OUT="results/phase0/check/$JOB"; mkdir -p "$OUT"
check_render && echo "render: ok" | tee "$OUT/render.txt"
"$SIM_PY" -m pytest -q -rs 2>&1 | tee "$OUT/pytest.txt"
cp .condex_paths "$OUT/condex_paths.txt"
"$SIM_PY" -c "import mujoco, numpy, pyarrow; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'pyarrow', pyarrow.__version__)" | tee "$OUT/versions.txt"
