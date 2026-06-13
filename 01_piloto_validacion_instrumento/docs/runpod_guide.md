# RunPod Execution Guide

Step-by-step instructions to run the full benchmark on a RunPod RTX 4090 instance.

---

## Before you start

**Required:**
- RunPod account with billing enabled
- HuggingFace account with access to Meta-LLaMA models (request at huggingface.co/meta-llama)
- Your HuggingFace token (`hf_...`)

**Important:** Use a **dedicated** GPU instance, not a spot instance. NVML measures total
GPU power — a shared instance where other processes use the GPU will contaminate your
energy measurements. This is a hard requirement for valid results.

---

## Step 1 — Create the RunPod instance

1. Log in to runpod.io
2. Click **Deploy** → **GPU Cloud**
3. Filter by: RTX 4090 | 24 GB VRAM
4. Select a **Secure Cloud** pod (not Community) for dedicated GPU
5. Select template: **RunPod PyTorch 2.x** (includes CUDA 12.x pre-installed)
6. Set storage: minimum **80 GB** (model weights require ~40 GB)
7. Click **Deploy**
8. Wait for the instance to start (~2–3 minutes)

---

## Step 2 — Connect and upload files

Option A — via RunPod web terminal:
1. Click **Connect** → **Start Web Terminal**
2. Upload your repo as a zip via the file browser, or clone from GitHub

Option B — via SSH (recommended):
```bash
# Add SSH key in RunPod settings first, then:
ssh root@<pod-ip> -p <port> -i ~/.ssh/your_key
```

Option C — via GitHub clone (simplest):
```bash
git clone https://github.com/<your-org>/llm-inference-energy-benchmark
cd llm-inference-energy-benchmark
```

---

## Step 3 — Set environment and run setup

```bash
export HF_TOKEN=hf_your_token_here

bash scripts/setup_runpod.sh
```

This script:
- Verifies GPU is RTX 4090
- Installs all Python dependencies (pinned versions)
- Creates directory structure
- Downloads LLaMA 3.1 8B Instruct (FP16/INT8 base, ~16 GB)
- Downloads LLaMA 3.1 8B AWQ INT4 pre-quantized (~4 GB)
- Generates `results/reproducibility.json`

**Expected duration:** 25–40 minutes (model download dominates)

---

## Step 4 — Build the prompt corpus

```bash
# Verify token counts without saving
python scripts/build_prompt_dataset.py --verify-only

# Expected output:
# Case A: n=30 | mean=256 | target=256 ±15%=[218,294] | valid=30/30
# Case B: n=30 | mean=1024 | target=1024 ±15%=[870,1178] | valid=30/30
# Case C: n=30 | mean=4096 | target=4096 ±15%=[3482,4710] | valid=30/30

# Build and save corpus
python scripts/build_prompt_dataset.py
```

If valid count is not 30/30, check your context files and refer to `docs/context_guide.md`.

---

## Step 5 — Run the pilot (strongly recommended)

The pilot runs 9 configurations (1 repetition each, Case A only) to verify everything
works before committing to the full 243-run benchmark.

**Terminal 2** — Start vLLM server:
```bash
bash scripts/start_vllm_fp16.sh
```

Wait until you see: `INFO: Application startup complete` (takes 2–4 minutes to load model)

**Terminal 1** — Verify server, then run pilot:
```bash
curl http://localhost:8000/health
# Should return: {"status":"ok"}

python scripts/benchmark_runner.py --quantization fp16 --pilot
```

**Expected pilot output:**
```
PILOT MODE: 9 configurations
[1/9] [fp16_b1_o64_caseA_rep1_...] ✓ 2.847J | 45.2 tok/s | VRAM peak: 16234MB
[2/9] [fp16_b1_o256_caseA_rep1_...] ✓ 8.123J | 43.8 tok/s | VRAM peak: 16289MB
...
```

If you see OOM at batch=1 or other errors, do NOT proceed to full benchmark. Check logs.

---

## Step 6 — Full FP16 benchmark

Keep the vLLM server from Step 5 running.

```bash
python scripts/benchmark_runner.py --quantization fp16
```

**Expected duration:** 6–10 hours
**Results saved to:** `results/fp16_<timestamp>/`

Do not close the terminal. You can check progress at any time:
```bash
cat results/fp16_*/summary.json
```

---

## Step 7 — Switch to INT8 W8A16

Stop the FP16 server (Ctrl+C in Terminal 2), then:

**Terminal 2:**
```bash
bash scripts/start_vllm_int8.sh
```

Wait for startup. Then:

**Terminal 1:**
```bash
python scripts/benchmark_runner.py --quantization int8_w8a16
```

**Expected duration:** 6–10 hours (may be slower than FP16 at low batch)

---

## Step 8 — Switch to INT4 AWQ

Stop the INT8 server, then:

**Terminal 2:**
```bash
bash scripts/start_vllm_awq.sh
```

**Terminal 1:**
```bash
python scripts/benchmark_runner.py --quantization int4_awq
```

**Note:** INT4 AWQ uses only ~4–5 GB for model weights, leaving ~19 GB for KV cache.
Case C (4096 tokens) + batch=8 configurations that OOM in FP16 may succeed here.
This is an expected and scientifically interesting result.

---

## Step 9 — Download results and stop the instance

```bash
# Check final summaries
cat results/fp16_*/summary.json
cat results/int8_*/summary.json
cat results/int4_*/summary.json
```

Download the entire `results/` folder to your local machine before stopping the instance.

**Via SCP:**
```bash
scp -r -P <port> root@<pod-ip>:/workspace/llm-inference-energy-benchmark/results/ ./results/
```

**Via RunPod file browser:** use the web interface to download results as zip.

Once downloaded, stop (not just pause) the RunPod instance to avoid charges.

---

## Estimated costs

| Phase | Duration | RTX 4090 cost (~$0.74/hr) |
|-------|----------|---------------------------|
| Setup + model download | ~40 min | ~$0.50 |
| Pilot | ~20 min | ~$0.25 |
| FP16 full benchmark | ~8 hr | ~$5.90 |
| INT8 full benchmark | ~9 hr | ~$6.65 |
| INT4 AWQ full benchmark | ~7 hr | ~$5.20 |
| **Total** | **~25 hr** | **~$18–20** |

Prices as of 2025. Check current RunPod pricing before deploying.

---

## Troubleshooting

**vLLM server won't start:**
- Check GPU memory: `nvidia-smi`
- Kill any existing processes: `pkill -f vllm`
- Check logs for CUDA version mismatch

**OOM errors in benchmark:**
- Expected for: FP16 + batch=8 + Case C → recorded as `status: "oom"`, not a failure
- Unexpected OOM: check VRAM usage with `nvidia-smi` before starting

**Token counts outside tolerance:**
- Re-run `build_prompt_dataset.py --verify-only`
- Check context files have correct content and length
- See `docs/context_guide.md`

**Benchmark interrupted:**
- Results are saved incrementally per run — no data is lost
- Re-run the same command; it will start a new results directory
- Do not resume mid-run; start fresh for that quantization level

**Connection lost to RunPod:**
- Use `tmux` or `screen` to run commands in persistent sessions:
  ```bash
  tmux new -s benchmark
  # run your commands inside tmux
  # detach: Ctrl+B, then D
  # reattach later: tmux attach -t benchmark
  ```
