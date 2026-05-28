#!/bin/bash
# start_vllm_awq.sh — Start vLLM server with INT4 AWQ
# Uses a pre-quantized AWQ checkpoint. Frees ~12 GB VRAM vs FP16,
# enabling Case C (4096 tokens) + batch=8 configurations.

echo "Starting vLLM server: INT4 AWQ"
echo "Model: /workspace/models/llama3.1-8b-instruct-awq"
echo "Expected VRAM usage: ~4-5 GB (model weights)"
echo "KV-cache budget: ~19 GB — enables Case C + batch=8"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/llama3.1-8b-instruct-awq \
    --quantization awq \
    --dtype float16 \
    --max-model-len 8192 \
    --port 8000 \
    --disable-log-requests
