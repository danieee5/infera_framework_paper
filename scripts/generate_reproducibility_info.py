"""
generate_reproducibility_info.py
Captures a complete snapshot of the execution environment before benchmarking.
Saved to results/reproducibility.json

Run ONCE before starting any benchmark run:
  python scripts/generate_reproducibility_info.py

This file is what a reviewer checks first to validate reproducibility.
It records software versions, GPU identity, git commit, and the exact
vLLM server arguments used for each quantization scheme.
"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def cmd(args: list) -> str:
    """Run a shell command, return stdout. Returns 'unavailable' on any failure."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else "unavailable"
    except Exception:
        return "unavailable"


def pkg_version(name: str) -> str:
    """Return installed version of a Python package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        return "not_installed"


def check_clock_control() -> dict:
    """
    Attempt to lock GPU clocks. Documents the outcome regardless of result.

    METHODOLOGICAL NOTE:
    RunPod virtualized instances typically do not grant the root privileges
    required to lock GPU clocks via nvidia-smi. GPU frequency may therefore
    vary during execution due to DVFS (Dynamic Voltage and Frequency Scaling).
    This is a known source of run-to-run variance for consumer-grade GPU
    benchmarks (Bhatia et al., arXiv:2501.08219, 2025). Three repetitions per
    configuration are used to quantify and report this variance in the results.
    """
    pm     = cmd(["nvidia-smi", "-pm", "1"])
    clocks = cmd(["nvidia-smi", "--lock-gpu-clocks=1980,1980"])
    return {
        "persistence_mode_result": pm[:120],
        "clock_lock_result":       clocks[:120],
        "clocks_locked":           "Successfully" in clocks,
        "methodological_note": (
            "If clocks_locked=false, GPU DVFS is active during the benchmark. "
            "Run-to-run variance from DVFS is captured by 3 repetitions per config. "
            "Reference: Bhatia et al. 2025 (arXiv:2501.08219)."
        ),
    }


def get_vllm_server_process() -> str:
    """Return the command line of any running vLLM process, or 'not_running'."""
    result = cmd(["pgrep", "-a", "-f", "vllm"])
    return result if result and result != "unavailable" else "not_running"


def main(output: str = "results/reproducibility.json"):
    info = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "hostname":       platform.node(),
        "os":             platform.platform(),
        "python_version": sys.version.split()[0],
        "git_commit":     cmd(["git", "rev-parse", "--short", "HEAD"]),

        # ── GPU identity ──────────────────────────────────────────────────────
        "gpu": {
            "name":                   cmd(["nvidia-smi", "--query-gpu=name",
                                           "--format=csv,noheader"]),
            "vram_total":             cmd(["nvidia-smi", "--query-gpu=memory.total",
                                           "--format=csv,noheader"]),
            "driver_version":         cmd(["nvidia-smi", "--query-gpu=driver_version",
                                           "--format=csv,noheader"]),
            "cuda_version_from_driver": cmd(["nvidia-smi", "--query-gpu=cuda_version",
                                             "--format=csv,noheader"]),
            "gpu_uuid":               cmd(["nvidia-smi", "--query-gpu=uuid",
                                           "--format=csv,noheader"]),
        },

        # ── Python packages ───────────────────────────────────────────────────
        "software_versions": {
            "vllm":          pkg_version("vllm"),
            "torch":         pkg_version("torch"),
            "transformers":  pkg_version("transformers"),
            "bitsandbytes":  pkg_version("bitsandbytes"),
            "autoawq":       pkg_version("autoawq"),
            "pynvml":        pkg_version("pynvml"),
            "codecarbon":    pkg_version("codecarbon"),
        },

        # ── vLLM server at time of capture ────────────────────────────────────
        "vllm_server_process_at_capture": get_vllm_server_process(),

        # ── Exact vLLM launch arguments for all three quantization schemes ────
        # These are the arguments used to start the server for each benchmark run.
        # Saving them here ensures full command-line reproducibility.
        "vllm_server_args": {
            "fp16": (
                "python -m vllm.entrypoints.openai.api_server "
                "--model /workspace/models/llama3.1-8b-instruct "
                "--dtype float16 "
                "--max-model-len 8192 "
                "--port 8000 "
                "--disable-log-requests"
            ),
            "int8_w8a16": (
                "python -m vllm.entrypoints.openai.api_server "
                "--model /workspace/models/llama3.1-8b-instruct "
                "--quantization bitsandbytes "
                "--load-format bitsandbytes "
                "--dtype float16 "
                "--max-model-len 8192 "
                "--port 8000 "
                "--disable-log-requests"
            ),
            "int4_awq": (
                "python -m vllm.entrypoints.openai.api_server "
                "--model /workspace/models/llama3.1-8b-instruct-awq "
                "--quantization awq "
                "--dtype float16 "
                "--max-model-len 8192 "
                "--port 8000 "
                "--disable-log-requests"
            ),
        },

        # ── GPU clock control ─────────────────────────────────────────────────
        "gpu_clock_control": check_clock_control(),

        # ── Experiment configuration ──────────────────────────────────────────
        "experiment_config": {
            "model_id":        "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "hardware_target": "NVIDIA RTX 4090 24 GB (RunPod dedicated instance)",
            "design":          "3×3×3×3 fully factorial",
            "configurations":  81,
            "repetitions":     3,
            "total_runs":      243,
            "vi1_quantization":    ["fp16", "int8_w8a16", "int4_awq"],
            "vi2_batch_size":      [1, 4, 8],
            "vi3_output_length":   [64, 256, 512],
            "vi4_context_tokens":  ["~256", "~1024", "~4096"],
            "measurement_protocol": {
                "primary_method":      "NVML nvmlDeviceGetPowerUsage",
                "sampling_hz":         10,
                "sampling_interval_ms": 100,
                "buffer_pre_post_ms":  500,
                "buffer_justification": (
                    "Husom et al. 2026 (MELODI) Table 1: "
                    "500 ms achieves 100% capture completeness rate; "
                    "200 ms achieves 0% CCR (discarded); "
                    "400 ms achieves 48% CCR (insufficient)."
                ),
                "integration_method":  "trapezoidal",
                "vram_tracking":       "nvmlDeviceGetMemoryInfo at each sample",
                "co2_triangulation":   "CodeCarbon (secondary source, known 10-30% underestimate)",
                "warmup_requests_per_group": 5,
                "cooling_between_groups_s":  120,
                "randomization_seed":        42,
            },
        },
    }

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(info, f, indent=2)

    print(f"✓ reproducibility.json saved: {output}")
    print(f"  GPU:           {info['gpu']['name']}  |  {info['gpu']['vram_total']}")
    print(f"  vLLM:          {info['software_versions']['vllm']}")
    print(f"  PyTorch:       {info['software_versions']['torch']}")
    print(f"  Git commit:    {info['git_commit']}")
    print(f"  Clocks locked: {info['gpu_clock_control']['clocks_locked']}")
    print(f"  vLLM process:  {info['vllm_server_process_at_capture'][:80]}")


if __name__ == "__main__":
    main()
