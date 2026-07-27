#!/usr/bin/env python3
"""Valida tareas, base de conocimiento y presupuesto antes de usar GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STAGE_ROOT))

from infera_kb import build_fixed_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida una configuración INFERA sin ejecutar inferencia."
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=STAGE_ROOT / "config/session_tasks.example.json",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=STAGE_ROOT / "kb",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--threshold", type=int, default=4500)
    parser.add_argument(
        "--tokenizer",
        help=(
            "ruta local o identificador de Hugging Face. Si se omite, "
            "solo se validan estructura y márgenes declarados."
        ),
    )
    return parser.parse_args()


def load_tasks(path: Path) -> tuple[dict[str, Any], list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return {}, [f"No existe el archivo de tareas: {path}"], warnings
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {}, [f"No se pudo leer JSON de tareas: {exc}"], warnings

    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("tasks debe ser una lista no vacía")
        return document, errors, warnings

    identifiers: list[str] = []
    for index, task in enumerate(tasks, start=1):
        if not isinstance(task, dict):
            errors.append(f"tarea {index}: debe ser un objeto")
            continue
        task_id = str(task.get("id", "")).strip()
        prompt = str(task.get("prompt", "")).strip()
        if not task_id:
            errors.append(f"tarea {index}: falta id")
        if not prompt:
            errors.append(f"tarea {task_id or index}: falta prompt")
        if not isinstance(task.get("verify", {}), dict):
            errors.append(f"tarea {task_id or index}: verify debe ser un objeto")
        identifiers.append(task_id)

    duplicates = sorted(
        identifier
        for identifier in set(identifiers)
        if identifier and identifiers.count(identifier) > 1
    )
    if duplicates:
        errors.append(f"identificadores duplicados: {', '.join(duplicates)}")

    known = set(identifiers)
    for task in tasks:
        if not isinstance(task, dict):
            continue
        for dependency in task.get("depends_on", []):
            if dependency not in known:
                errors.append(
                    f"{task.get('id', '?')}: dependencia inexistente {dependency}"
                )

    decoding = document.get("decoding", {})
    if float(decoding.get("temperature", 0.0)) != 0.0:
        warnings.append(
            "temperature no es 0; las réplicas pueden variar aunque la semilla sea fija"
        )
    if int(decoding.get("max_tokens", 0)) <= 0:
        errors.append("decoding.max_tokens debe ser positivo")
    return document, errors, warnings


def token_counts(
    tokenizer_ref: str,
    kb_text: str,
    tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "transformers no está instalado; usa requirements-gpu.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        local_files_only=Path(tokenizer_ref).exists(),
    )

    def count_text(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    first_messages = [
        {"role": "system", "content": kb_text},
        {"role": "user", "content": tasks[0]["prompt"]},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        first_prompt_tokens = len(
            tokenizer.apply_chat_template(
                first_messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    else:
        first_prompt_tokens = count_text(kb_text) + count_text(tasks[0]["prompt"])

    prompt_lengths = [count_text(str(task["prompt"])) for task in tasks]
    return {
        "tokenizer": tokenizer_ref,
        "fixed_context_tokens": count_text(kb_text),
        "first_request_prompt_tokens": first_prompt_tokens,
        "task_prompt_tokens_min": min(prompt_lengths),
        "task_prompt_tokens_max": max(prompt_lengths),
        "task_prompt_tokens_total": sum(prompt_lengths),
    }


def main() -> int:
    args = parse_args()
    document, errors, warnings = load_tasks(args.tasks)
    decoding = document.get("decoding", {}) if document else {}
    max_output = int(decoding.get("max_tokens", 0) or 0)

    if args.max_model_len <= 0:
        errors.append("max-model-len debe ser positivo")
    if args.threshold <= 0:
        errors.append("threshold debe ser positivo")
    if args.threshold + max_output > args.max_model_len:
        errors.append(
            "threshold + decoding.max_tokens supera max-model-len; "
            "no queda presupuesto para la salida"
        )

    required_kb = (
        args.kb_dir / "vigia_kb.md",
        args.kb_dir / "permisos_medicos.csv",
        args.kb_dir / "inventario_uniformes.csv",
    )
    missing_kb = [str(path) for path in required_kb if not path.is_file()]
    if missing_kb:
        errors.append(
            "faltan archivos esperados de la base: " + ", ".join(missing_kb)
        )

    counts: dict[str, Any] | None = None
    if not missing_kb and document and args.tokenizer:
        try:
            counts = token_counts(
                args.tokenizer,
                build_fixed_context(str(args.kb_dir)),
                document["tasks"],
            )
            if counts["first_request_prompt_tokens"] + max_output > args.max_model_len:
                errors.append(
                    "la primera petición y su salida reservada ya superan "
                    "max-model-len"
                )
        except Exception as exc:
            errors.append(f"no se pudo calcular tokens: {exc}")
    elif not args.tokenizer:
        warnings.append(
            "no se calcularon tokens reales; vuelve a ejecutar con --tokenizer"
        )

    report = {
        "ok": not errors,
        "tasks_file": str(args.tasks),
        "kb_dir": str(args.kb_dir),
        "task_count": len(document.get("tasks", [])) if document else 0,
        "max_model_len": args.max_model_len,
        "max_output_tokens": max_output,
        "compaction_threshold": args.threshold,
        "token_counts": counts,
        "errors": errors,
        "warnings": warnings,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
