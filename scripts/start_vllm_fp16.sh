#!/bin/bash
# start_vllm_fp16.sh — Start vLLM server with FP16 (baseline)
# Run in a SEPARATE terminal before executing the benchmark.
# Keep this terminal open during the entire fp16 benchmark run.

echo "Starting vLLM server: FP16 (baseline)"
echo "Model: /workspace/models/llama3.1-8b-instruct"
echo "Expected VRAM usage: ~16 GB (model weights)"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/llama3.1-8b-instruct \
    --served-model-name "meta-llama/Meta-Llama-3.1-8B-Instruct" \
    --dtype float16 \
    --max-model-len 8192 \
    --port 8000 \
    --disable-log-requests
