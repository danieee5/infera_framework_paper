#!/bin/bash
# setup_runpod.sh — One-time setup for RunPod RTX 4090 dedicated instance.
# Run from the project root after cloning the repo.
#
# Requires: HF_TOKEN environment variable set to your HuggingFace token.
# LLaMA 3.1 models require an account with approved access at huggingface.co/meta-llama
#
# Usage:
#   export HF_TOKEN=hf_your_token_here
#   bash scripts/setup_runpod.sh
#
# What this script does:
#   1. Verifies the GPU is RTX 4090 (hard requirement — NVML measures total GPU power)
#   2. Checks HF_TOKEN and logs into HuggingFace
#   3. Installs all Python dependencies at pinned versions (requirements.txt)
#   4. Downloads LLaMA 3.1 8B Instruct base model (~16 GB) → for FP16 and INT8 runs
#   5. Downloads LLaMA 3.1 8B Instruct AWQ INT4 model (~4.5 GB) → for INT4 AWQ runs
#   6. Generates results/reproducibility.json (environment snapshot)
#
# Expected duration: 25–45 minutes (model download dominates)

set -e  # Exit immediately on any error

echo ""
echo "======================================================================"
echo "  LLM Inference Energy Benchmark — RunPod Setup"
echo "======================================================================"
echo ""

# ── 1. Verify GPU ─────────────────────────────────────────────────────────────
echo "[1/6] Verifying GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unavailable")
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 || echo "unavailable")
echo "  GPU:  $GPU_NAME"
echo "  VRAM: $VRAM"

if [[ "$GPU_NAME" != *"RTX 4090"* ]]; then
    echo ""
    echo "  WARNING: Expected NVIDIA RTX 4090 but found: $GPU_NAME"
    echo "  NVML measures total GPU power draw — on a shared instance, other"
    echo "  processes contaminate energy measurements. Use a dedicated pod."
    echo "  Proceeding anyway — verify your instance type before running benchmark."
    echo ""
else
    echo "  ✓ RTX 4090 confirmed"
fi
echo ""

# ── 2. Check HuggingFace token ────────────────────────────────────────────────
echo "[2/6] Checking HuggingFace token..."
if [ -z "$HF_TOKEN" ]; then
    echo ""
    echo "  FATAL: HF_TOKEN environment variable is not set."
    echo "  LLaMA 3.1 requires a HuggingFace account with approved access."
    echo "  Request access at: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct"
    echo "  Then set your token: export HF_TOKEN=hf_your_token_here"
    echo ""
    exit 1
fi
echo "  Token present (length: ${#HF_TOKEN})"
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || \
    python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>/dev/null || true
echo "  ✓ HuggingFace login done"
echo ""

# ── 3. Install Python dependencies ────────────────────────────────────────────
echo "[3/6] Installing Python dependencies (pinned versions from requirements.txt)..."
pip install -r requirements.txt --quiet
echo "  ✓ All dependencies installed"
echo ""

# ── 4. Download base model (FP16 / INT8) ──────────────────────────────────────
echo "[4/6] Downloading LLaMA 3.1 8B Instruct base model (FP16/INT8, ~16 GB)..."
mkdir -p /workspace/models

if [ -d "/workspace/models/llama3.1-8b-instruct" ] && \
   [ "$(ls -A /workspace/models/llama3.1-8b-instruct 2>/dev/null)" ]; then
    echo "  Model directory already exists and is non-empty — skipping download."
    echo "  (Delete /workspace/models/llama3.1-8b-instruct to force re-download)"
else
    python - <<PYEOF
import sys
from huggingface_hub import snapshot_download
print("  Downloading meta-llama/Meta-Llama-3.1-8B-Instruct ...")
print("  This takes 15-30 minutes depending on network speed.")
try:
    snapshot_download(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        local_dir="/workspace/models/llama3.1-8b-instruct",
        token="$HF_TOKEN",
        ignore_patterns=["*.pt", "original/*"],
    )
    print("  ✓ Base model downloaded to /workspace/models/llama3.1-8b-instruct")
except Exception as e:
    print(f"\n  FATAL: Could not download model: {e}", file=sys.stderr)
    print("  Check that your HF token has approved access to meta-llama/Meta-Llama-3.1-8B-Instruct", file=sys.stderr)
    sys.exit(1)
PYEOF
fi
echo ""

# ── 5. Download AWQ INT4 model ────────────────────────────────────────────────
echo "[5/6] Downloading LLaMA 3.1 8B Instruct AWQ INT4 (~4.5 GB)..."

if [ -d "/workspace/models/llama3.1-8b-instruct-awq" ] && \
   [ "$(ls -A /workspace/models/llama3.1-8b-instruct-awq 2>/dev/null)" ]; then
    echo "  AWQ model directory already exists and is non-empty — skipping download."
    echo "  (Delete /workspace/models/llama3.1-8b-instruct-awq to force re-download)"
else
    python - <<PYEOF
import sys
from huggingface_hub import snapshot_download
print("  Downloading hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 ...")
try:
    snapshot_download(
        repo_id="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
        local_dir="/workspace/models/llama3.1-8b-instruct-awq",
        token="$HF_TOKEN",
    )
    print("  ✓ AWQ INT4 model downloaded to /workspace/models/llama3.1-8b-instruct-awq")
except Exception as e:
    print(f"\n  FATAL: Could not download AWQ model: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
fi
echo ""

# ── 6. Generate reproducibility snapshot ──────────────────────────────────────
echo "[6/6] Generating reproducibility snapshot..."
mkdir -p results
python scripts/generate_reproducibility_info.py
echo ""

# ── Summary ───────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "  SETUP COMPLETE"
echo "======================================================================"
echo ""
echo "Next steps:"
echo ""
echo "  Step 1 — Build the prompt corpus:"
echo "    python scripts/build_prompt_dataset.py --verify-only  # dry-run first"
echo "    python scripts/build_prompt_dataset.py"
echo ""
echo "  Step 2 — Run the pilot (strongly recommended before full benchmark):"
echo "    Terminal 2: bash scripts/start_vllm_fp16.sh"
echo "    Wait for:   INFO: Application startup complete"
echo "    Terminal 1: python scripts/benchmark_runner.py --quantization fp16 --pilot"
echo ""
echo "  Step 3 — Full benchmark runs (in order):"
echo "    python scripts/benchmark_runner.py --quantization fp16"
echo "    python scripts/benchmark_runner.py --quantization int8_w8a16"
echo "    python scripts/benchmark_runner.py --quantization int4_awq"
echo ""
echo "  See docs/runpod_guide.md for complete instructions."
echo ""
