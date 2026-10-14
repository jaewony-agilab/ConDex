#!/usr/bin/env bash
# Judge venv (cluster): vLLM for serving Qwen3-VL-8B (frozen and LoRA), plus peft for the LoRA
# fine-tune, plus condex. CPU node is fine:  bash scripts/install_judge.sh
# Pins: vllm 0.11.0 is the first version pin for Qwen3-VL (not yet verified on our cluster).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${CONDEX_THIRD_PARTY:-$REPO/third_party}"
VENV="$TP/judge_venv"
uv venv "$VENV" --python 3.10 --allow-existing
uv pip install --python "$VENV/bin/python" "vllm==0.11.0" "peft>=0.17" "accelerate>=1.0" -e "$REPO[llm]"
"$VENV/bin/python" -c "import vllm, transformers, peft, torch; from transformers import Qwen3VLForConditionalGeneration; print('vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__, 'torch', torch.__version__)"
"$VENV/bin/python" -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-VL-8B-Instruct'))"
uv pip freeze --python "$VENV/bin/python" > "$TP/judge_freeze.txt"
P="$REPO/.condex_paths"; touch "$P"
grep -v "^VLM_PY=" "$P" > "$P.tmp" || true; echo "VLM_PY=$VENV/bin/python" >> "$P.tmp"; mv "$P.tmp" "$P"
cat "$P"
