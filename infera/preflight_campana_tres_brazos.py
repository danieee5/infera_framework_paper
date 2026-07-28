#!/usr/bin/env python3
"""Preflight sin inferencia para la campaña fija de tres brazos."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from infera_kb import build_fixed_context
from infera_session_runner import (
    TokenBudgetGuard,
    atomic_write_json,
    canonical_json,
    sha256_file,
    sha256_text,
)


ROOT = Path(__file__).resolve().parent
FINGERPRINT_KEYS = (
    "tokenizer_class",
    "vocab_size",
    "backend_sha256",
    "chat_template_sha256",
    "special_tokens_sha256",
)
EXPECTED_PACKAGE_VERSIONS = {
    "vllm": "0.5.3",
    "transformers": "4.43.3",
    "tokenizers": "0.19.1",
    "requests": "2.32.3",
    "nvidia-ml-py": "12.560.30",
}
MODEL_CORE_KEYS = (
    "model_type",
    "architectures",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
)
MODEL_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".json", ".model")
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida configuración/tokenizers sin emitir peticiones."
    )
    parser.add_argument("--fp16-tokenizer", required=True)
    parser.add_argument("--awq-tokenizer", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--kb-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=int, required=True)
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--expected-tasks", type=int, required=True)
    parser.add_argument("--baseline-seconds", type=float, required=True)
    parser.add_argument("--settle-seconds", type=float, required=True)
    parser.add_argument("--warmup-count", type=int, required=True)
    parser.add_argument("--cooldown-seconds", type=float, required=True)
    parser.add_argument("--request-timeout-seconds", type=float, required=True)
    parser.add_argument("--server-start-attempts", type=int, required=True)
    parser.add_argument("--required-vllm-version", default="0.5.3")
    return parser.parse_args()


def package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "vllm",
        "transformers",
        "tokenizers",
        "requests",
        "nvidia-ml-py",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(f"falta la dependencia {package}") from exc
    return versions


def git_metadata() -> dict[str, object]:
    repository = ROOT.parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"available": True, "commit": commit, "tracked_dirty": bool(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "tracked_dirty": None}


def comparable_fingerprint(guard: TokenBudgetGuard) -> dict[str, object]:
    return {key: guard.fingerprint[key] for key in FINGERPRINT_KEYS}


def local_model_inventory(reference: str) -> dict[str, object]:
    root = Path(reference)
    if not root.is_dir():
        raise RuntimeError(
            f"la campaña exige un directorio de modelo local: {reference}"
        )
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_FILE_SUFFIXES
    )
    weights = [
        path for path in paths if path.suffix.lower() in WEIGHT_FILE_SUFFIXES
    ]
    if not weights:
        raise RuntimeError(f"{reference}: no se encontraron archivos de pesos")
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return {
        "file_count": len(files),
        "weight_file_count": len(weights),
        "total_bytes": sum(item["bytes"] for item in files),
        "inventory_sha256": sha256_text(canonical_json(files)),
        "files": files,
    }


def model_fingerprint(reference: str) -> dict[str, object]:
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("falta transformers para validar modelos") from exc
    config = AutoConfig.from_pretrained(
        reference,
        local_files_only=True,
    ).to_dict()
    return {
        "reference": reference,
        "core_config": {
            key: config.get(key)
            for key in MODEL_CORE_KEYS
        },
        "quantization_config": config.get("quantization_config"),
        "inventory": local_model_inventory(reference),
    }


def main() -> int:
    args = parse_args()
    if sys.version_info[:2] != (3, 10):
        raise SystemExit(
            f"Python {sys.version_info.major}.{sys.version_info.minor}; "
            "se requiere 3.10"
        )
    if args.out.exists():
        raise SystemExit(f"no se sobrescribirá el preflight: {args.out}")
    if not args.tasks.is_file():
        raise SystemExit(f"no existe el escenario: {args.tasks}")
    if not args.kb_dir.is_dir():
        raise SystemExit(f"no existe la base: {args.kb_dir}")
    if args.threshold <= 0 or args.max_model_len <= 0:
        raise SystemExit("threshold y max-model-len deben ser positivos")
    if args.pairs <= 0 or args.expected_tasks <= 0:
        raise SystemExit("pairs y expected-tasks deben ser positivos")
    if args.baseline_seconds <= 0:
        raise SystemExit("baseline-seconds debe ser positivo")
    if args.settle_seconds < 0:
        raise SystemExit("settle-seconds no puede ser negativo")
    if args.warmup_count < 0 or args.cooldown_seconds < 0:
        raise SystemExit("warmup-count y cooldown-seconds no pueden ser negativos")
    if args.request_timeout_seconds <= 0 or args.server_start_attempts <= 0:
        raise SystemExit("timeouts/intentos deben ser positivos")

    document = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != args.expected_tasks:
        raise SystemExit(
            f"el escenario debe contener exactamente {args.expected_tasks} tareas"
        )
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise SystemExit(f"tarea {index}: se esperaba un objeto")
        for field in ("id", "type", "prompt"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise SystemExit(
                    f"tarea {index}: {field} debe ser texto no vacío"
                )
    identifiers = [task["id"] for task in tasks]
    if len(set(identifiers)) != len(identifiers):
        raise SystemExit("hay ids de tarea duplicados")
    decoding = document.get("decoding", {})
    max_tokens = int(decoding.get("max_tokens", 0))
    temperature = float(decoding.get("temperature", 0.0))
    seed = int(decoding.get("seed", 42))
    if max_tokens <= 0:
        raise SystemExit("decoding.max_tokens debe ser positivo")
    if args.threshold + max_tokens > args.max_model_len:
        raise SystemExit(
            f"{args.threshold} + {max_tokens} > {args.max_model_len}"
        )

    versions = package_versions()
    required_versions = {
        **EXPECTED_PACKAGE_VERSIONS,
        "vllm": args.required_vllm_version,
    }
    mismatches = {
        package: {"actual": versions[package], "required": required}
        for package, required in required_versions.items()
        if versions[package] != required
    }
    if mismatches:
        raise SystemExit(
            "versiones del stack no congeladas: "
            + json.dumps(mismatches, ensure_ascii=False)
        )
    if versions["torch"].split("+", maxsplit=1)[0] != "2.3.1":
        raise SystemExit(
            f"torch={versions['torch']}; se requiere la serie exacta 2.3.1"
        )
    import torch

    if torch.version.cuda != "12.1":
        raise SystemExit(
            f"torch CUDA={torch.version.cuda}; se requiere CUDA 12.1"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SystemExit("PyTorch debe ver exactamente una GPU CUDA")
    cuda_properties = torch.cuda.get_device_properties(0)

    kb = build_fixed_context(str(args.kb_dir))
    fp16 = TokenBudgetGuard.from_reference(args.fp16_tokenizer)
    awq = TokenBudgetGuard.from_reference(args.awq_tokenizer)
    if comparable_fingerprint(fp16) != comparable_fingerprint(awq):
        raise SystemExit(
            "los tokenizers FP16 y AWQ no son idénticos en backend/plantilla/"
            "tokens especiales"
        )
    models = {
        "FP16": model_fingerprint(args.fp16_tokenizer),
        "AWQ": model_fingerprint(args.awq_tokenizer),
    }
    if models["FP16"]["core_config"] != models["AWQ"]["core_config"]:
        raise SystemExit(
            "FP16 y AWQ no declaran la misma arquitectura/configuración base"
        )
    fp16_quant = models["FP16"]["quantization_config"]
    if isinstance(fp16_quant, dict) and fp16_quant.get("quant_method"):
        raise SystemExit("el directorio FP16 declara una cuantización")
    awq_quant = models["AWQ"]["quantization_config"]
    if (
        not isinstance(awq_quant, dict)
        or str(awq_quant.get("quant_method", "")).lower() != "awq"
    ):
        raise SystemExit("el directorio AWQ no declara quant_method=awq")

    first_messages = [
        {"role": "system", "content": kb},
        {"role": "user", "content": tasks[0]["prompt"]},
    ]
    first_counts = {
        "FP16": fp16.check(
            first_messages,
            max_tokens,
            args.max_model_len,
        ),
        "AWQ": awq.check(
            first_messages,
            max_tokens,
            args.max_model_len,
        ),
    }
    if len(set(first_counts.values())) != 1:
        raise SystemExit("los tokenizers no coinciden en la primera petición")

    code_paths = (
        ROOT / "infera_session_runner.py",
        ROOT / "gpu_power_monitor.py",
        ROOT / "infera_compaction.py",
        ROOT / "infera_quality.py",
        ROOT / "analiza_tres_brazos.py",
        ROOT / "preflight_campana_tres_brazos.py",
        ROOT / "escribe_manifiesto_campana.py",
        ROOT / "run_campana_tres_brazos.sh",
        ROOT / "reanaliza_campana_tres_brazos.sh",
    )
    report = {
        "schema_version": 1,
        "status": "ready_without_gpu_inference",
        "tasks": str(args.tasks.resolve()),
        "tasks_sha256": sha256_file(args.tasks),
        "task_count": len(tasks),
        "task_ids_sha256": sha256_text(
            json.dumps(identifiers, ensure_ascii=False, separators=(",", ":"))
        ),
        "kb_dir": str(args.kb_dir.resolve()),
        "kb_sha256": sha256_text(kb),
        "threshold_tokens": args.threshold,
        "pairs_kept": args.pairs,
        "max_model_len": args.max_model_len,
        "requested_max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
        "baseline_seconds": args.baseline_seconds,
        "post_warmup_settle_seconds": args.settle_seconds,
        "warmup_count": args.warmup_count,
        "cooldown_seconds": args.cooldown_seconds,
        "request_timeout_seconds": args.request_timeout_seconds,
        "server_start_attempts": args.server_start_attempts,
        "first_request_prompt_tokens": first_counts,
        "tokenizers": {
            "FP16": fp16.fingerprint,
            "AWQ": awq.fingerprint,
        },
        "models": models,
        "package_versions": versions,
        "torch_cuda_version": torch.version.cuda,
        "runtime_environment": {
            "python_version": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git": git_metadata(),
            "cuda_device": {
                "name": cuda_properties.name,
                "total_memory_bytes": int(cuda_properties.total_memory),
                "compute_capability": [
                    int(cuda_properties.major),
                    int(cuda_properties.minor),
                ],
                "multiprocessor_count": int(
                    cuda_properties.multi_processor_count
                ),
            },
        },
        "code_sha256": {
            path.name: sha256_file(path)
            for path in code_paths
        },
        "runtime_guards": {
            "pre_request_token_budget": True,
            "usage_must_equal_preflight": True,
            "all_29_tasks_required": True,
            "managed_arm_must_intervene": True,
            "post_intervention_must_fall_below_threshold": True,
            "nvml_required": True,
            "raw_nvml_trace_required": True,
            "thermal_clock_utilization_trace_required": True,
            "continuous_compute_process_check": True,
        },
        "prefix_caching_enabled": False,
        "uncertainty": (
            "Los prompts posteriores dependen de outputs aún no generados. "
            "No se prueba su ocupación ex ante; el runner vuelve a contar y "
            "aborta antes de cada petición."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
