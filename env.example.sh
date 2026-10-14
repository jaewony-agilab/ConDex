# Big-disk locations for this clone. Copy to env.local.sh (gitignored) and edit:
#   cp env.example.sh env.local.sh
# Sourced automatically by scripts/install_*.sh and slurm/common.sh (every sbatch job).
BIG=/path/to/big/disk/condex   # e.g. a project/scratch filesystem, not $HOME

export HF_HOME="$BIG/hf"                 # RLDX-1 + Qwen3-VL checkpoints (~20 GB+)
export UV_CACHE_DIR="$BIG/uv_cache"      # uv package cache (venv builds)
export CONDEX_RAW="$BIG/raw"             # raw episode directories (frames, depth, parquet)
export CONDEX_SCRATCH="$BIG/scratch"     # judge datasets, LoRA adapters, server logs
# export CONDEX_THIRD_PARTY="$BIG/third_party"   # RLDX-1 clone + venvs (default: <repo>/third_party)
# export ANTHROPIC_API_KEY=...                   # planner / Claude judge; this file is never committed
