# Big-disk locations for this clone. Copy to env.local.sh (gitignored) and edit:
#   cp env.example.sh env.local.sh
# Sourced automatically by scripts/install_*.sh and slurm/common.sh (every sbatch job).
BIG=/path/to/big/disk/condex   # e.g. a project/scratch filesystem, not $HOME

export HF_HOME="$BIG/hf"                 # RLDX-1 + Qwen3-VL checkpoints (~35 GB)
export UV_CACHE_DIR="$BIG/uv_cache"      # uv package cache (venv builds)
export CONDEX_RAW="$BIG/raw"             # raw episode directories (frames, depth, parquet)
export CONDEX_SCRATCH="$BIG/scratch"     # judge datasets, LoRA adapters, server logs
# export CONDEX_THIRD_PARTY="$BIG/third_party"   # RLDX-1 clone + venvs (default: <repo>/third_party)
# export ANTHROPIC_API_KEY=...                   # planner / Claude judge; this file is never committed
# export UV_PYTHON_PREFERENCE=only-managed       # if the system Python has no headers (sdist builds fail on Python.h)

# Cluster-specific sbatch flags, added by slurm/submit.sh cpu|gpu (the #SBATCH headers carry none).
# Example: the original cluster (CPU partition "cpu", required --wckey, no CPU/memory requests):
#   CONDEX_SBATCH_CPU="-p cpu --wckey=project-short-name:sub_4dpdata"
#   CONDEX_SBATCH_GPU="-p <gpu partition> --wckey=project-short-name:sub_4dpdata"
# Example: a cluster with no CPU partition, default QOS/account, 7 GB RAM per requested CPU:
#   CONDEX_SBATCH_CPU="-p all -c 8"
#   CONDEX_SBATCH_GPU="-p a6000"
CONDEX_SBATCH_CPU="-p cpu --wckey=project-short-name:sub_4dpdata"
CONDEX_SBATCH_GPU="--wckey=project-short-name:sub_4dpdata"   # add -p <gpu partition>
