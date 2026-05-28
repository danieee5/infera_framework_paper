#!/bin/bash
# start_vllm_int8.sh — Start vLLM server with INT8 W8A16 (bitsandbytes)
# Weights quantized to INT8, activations remain FP16.
# Expected VRAM usage: ~10–12 GB (model weights)

echo "Starting vLLM server: INT8 W8A16 (bitsandbytes)"
echo "Model: /workspace/models/llama3.1-8b-instruct"
echo "Expected VRAM usage: ~10-12 GB (model weights)"
echo "Note: W8A16 may be slower than FP16 at batch=1 due to dequantization overhead."
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/llama3.1-8b-instruct \
    --quantization bitsandbytes \
    --load-format bitsandbytes \
    --dtype float16 \
    --max-model-len 8192 \
    --port 8000 \
    --disable-log-requests
