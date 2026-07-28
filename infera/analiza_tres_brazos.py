#!/usr/bin/env python3
"""Valida y describe la campaña fija completo/resumen/descarte."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics as st
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

from escribe_manifiesto_campana import flattened_schedule
from infera_kb import build_fixed_context
from infera_quality import score_task


BRAZOS = ("completo", "resumen", "descarte")
QUANTS = ("AWQ", "FP16")
REPS = (1, 2, 3)
EVENT_TYPES = ("COMPACTION", "DESCARTE")
EXPECTED_THRESHOLD = 4500
ROOT = Path(__file__).resolve().parent
FINGERPRINT_KEYS = (
    "tokenizer_class",
    "vocab_size",
    "backend_sha256",
    "chat_template_sha256",
    "special_tokens_sha256",
)
NOMBRE = {
    "completo": "historial completo (calibración)",
    "resumen": "resumen del historial",
    "descarte": "recencia ciega por antigüedad",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tareas(filas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [fila for fila in filas if fila["task_type"] not in EVENT_TYPES]


def eventos(filas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [fila for fila in filas if fila["task_type"] in EVENT_TYPES]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw:
        raise ValueError(f"{path.name}: archivo vacío")
    if not raw.endswith(b"\n"):
        raise ValueError(f"{path.name}: falta salto final; posible truncamiento")
    rows = []
    for line_number, line in enumerate(
        raw.decode("utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path.name}:{line_number}: línea vacía")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path.name}:{line_number}: JSON truncado/inválido: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number}: se esperaba objeto")
        rows.append(row)
    return rows


def close(left: float, right: float, tolerance: float = 0.02) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def fingerprint_subset(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("fingerprint de tokenizer ausente")
    missing = [key for key in FINGERPRINT_KEYS if key not in value]
    if missing:
        raise ValueError(f"fingerprint de tokenizer incompleto: {missing}")
    return {key: value[key] for key in FINGERPRINT_KEYS}


def load_campaign_preflight(
    crudos: Path,
    expected_tasks: int,
    max_model_len: int,
    pairs_kept: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    campaign_path = crudos.parent / "manifiesto_campana.json"
    if not campaign_path.is_file():
        raise ValueError("falta manifiesto_campana.json")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("status") not in ("running", "complete", "failed"):
        raise ValueError(
            "solo se analizan campañas iniciadas (running/complete/failed)"
        )
    if campaign.get("schedule") != flattened_schedule():
        raise ValueError("el schedule no coincide con el protocolo congelado")

    preflight_value = campaign.get("preflight")
    if not isinstance(preflight_value, str) or not preflight_value:
        raise ValueError("el manifiesto no identifica el preflight")
    preflight_path = Path(preflight_value)
    if not preflight_path.is_file():
        raise ValueError(f"no existe el preflight declarado: {preflight_path}")
    if campaign.get("preflight_sha256") != sha256_file(preflight_path):
        raise ValueError("el preflight cambió después de iniciar la campaña")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "ready_without_gpu_inference":
        raise ValueError("el preflight declarado no está listo")

    expected_configuration = {
        "task_count": expected_tasks,
        "threshold_tokens": EXPECTED_THRESHOLD,
        "pairs_kept": pairs_kept,
        "max_model_len": max_model_len,
        "baseline_seconds": 30.0,
        "post_warmup_settle_seconds": 30.0,
        "warmup_count": 5,
        "cooldown_seconds": 120.0,
        "request_timeout_seconds": 600.0,
        "server_start_attempts": 2,
    }
    for key, expected in expected_configuration.items():
        if preflight.get(key) != expected:
            raise ValueError(
                f"preflight {key}={preflight.get(key)!r}; se esperaba {expected!r}"
            )
    if preflight.get("prefix_caching_enabled") is not False:
        raise ValueError("el estado de caché de prefijos no es el predeclarado")
    runtime_guards = preflight.get("runtime_guards")
    required_guards = (
        "raw_nvml_trace_required",
        "thermal_clock_utilization_trace_required",
        "continuous_compute_process_check",
    )
    if not isinstance(runtime_guards, dict) or any(
        runtime_guards.get(key) is not True for key in required_guards
    ):
        raise ValueError("el preflight no exige las trazas de diagnóstico")

    tasks_value = preflight.get("tasks")
    if not isinstance(tasks_value, str) or not tasks_value:
        raise ValueError("el preflight no identifica el escenario")
    tasks_path = Path(tasks_value)
    if (
        not tasks_path.is_file()
        or sha256_file(tasks_path) != preflight.get("tasks_sha256")
    ):
        raise ValueError("el escenario falta o cambió después del preflight")
    tasks_document = json.loads(tasks_path.read_text(encoding="utf-8"))
    scenario_tasks = tasks_document.get("tasks")
    if not isinstance(scenario_tasks, list) or len(scenario_tasks) != expected_tasks:
        raise ValueError("el escenario validado ya no contiene 29 tareas")
    identifiers = [task.get("id") for task in scenario_tasks]
    if preflight.get("task_ids_sha256") != sha256_text(
        json.dumps(identifiers, ensure_ascii=False, separators=(",", ":"))
    ):
        raise ValueError("la secuencia de IDs no coincide con el preflight")
    decoding = tasks_document.get("decoding")
    if not isinstance(decoding, dict):
        raise ValueError("el escenario no declara decoding")
    if decoding.get("max_tokens") != preflight.get("requested_max_tokens"):
        raise ValueError("max_tokens del escenario no coincide con el preflight")
    if decoding.get("temperature") != preflight.get("temperature"):
        raise ValueError("temperature del escenario no coincide con el preflight")
    if decoding.get("seed") != preflight.get("seed"):
        raise ValueError("seed del escenario no coincide con el preflight")

    kb_value = preflight.get("kb_dir")
    if not isinstance(kb_value, str) or not kb_value:
        raise ValueError("el preflight no identifica la base de conocimiento")
    kb_dir = Path(kb_value)
    if not kb_dir.is_dir():
        raise ValueError("la base de conocimiento declarada ya no existe")
    if sha256_text(build_fixed_context(str(kb_dir))) != preflight.get("kb_sha256"):
        raise ValueError("la base de conocimiento cambió después del preflight")

    code_hashes = preflight.get("code_sha256")
    if not isinstance(code_hashes, dict):
        raise ValueError("el preflight no congeló hashes del código")
    required_code = {
        "infera_session_runner.py",
        "gpu_power_monitor.py",
        "analiza_tres_brazos.py",
        "preflight_campana_tres_brazos.py",
        "escribe_manifiesto_campana.py",
        "run_campana_tres_brazos.sh",
    }
    if not required_code.issubset(code_hashes):
        raise ValueError("el preflight no congeló todo el código crítico")
    # Los hashes originales quedan autenticados por preflight_sha256 y los
    # manifiestos de sesión. No se exige que el checkout actual sea idéntico:
    # eso impediría recuperar el análisis después de corregir un bug del
    # analizador. El manifiesto de análisis registra cualquier deriva.

    tokenizers = preflight.get("tokenizers")
    if not isinstance(tokenizers, dict):
        raise ValueError("el preflight no identifica ambos tokenizers")
    fingerprints = [
        fingerprint_subset(tokenizers.get(quant))
        for quant in QUANTS
    ]
    if fingerprints[0] != fingerprints[1]:
        raise ValueError("los tokenizers del preflight no son comparables")
    models = preflight.get("models")
    if not isinstance(models, dict):
        raise ValueError("el preflight no congeló ambos modelos")
    for quant in QUANTS:
        model = models.get(quant)
        if not isinstance(model, dict):
            raise ValueError(f"el preflight no congeló el modelo {quant}")
        reference = model.get("reference")
        inventory = model.get("inventory")
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"el modelo {quant} no tiene referencia")
        if (
            not isinstance(inventory, dict)
            or not inventory.get("inventory_sha256")
            or not inventory.get("files")
        ):
            raise ValueError(f"el modelo {quant} no tiene inventario autenticado")
    if models["FP16"].get("core_config") != models["AWQ"].get("core_config"):
        raise ValueError("los modelos no declaran la misma configuración base")
    awq_quant = models["AWQ"].get("quantization_config")
    if (
        not isinstance(awq_quant, dict)
        or str(awq_quant.get("quant_method", "")).lower() != "awq"
    ):
        raise ValueError("el modelo AWQ no declara quant_method=awq")
    return campaign, preflight, scenario_tasks


def validate_discard_selection(
    path: Path,
    rows: list[dict[str, Any]],
    pairs_kept: int,
) -> None:
    history: list[dict[str, str]] = []
    active_ids: list[str] = []
    for row in rows:
        if row["task_type"] not in EVENT_TYPES:
            history.extend((
                {"role": "user", "content": row["prompt_text"]},
                {"role": "assistant", "content": row["response_text"]},
            ))
            active_ids.append(row["task_id"])
            continue
        if row["task_type"] != "DESCARTE":
            raise ValueError(f"{path.name}: descarte contiene COMPACTION")
        if len(history) % 2:
            raise ValueError(f"{path.name}: historial impar antes del descarte")
        before_hash = sha256_text(canonical_json(history))
        if row.get("historial_antes_sha256") != before_hash:
            raise ValueError(f"{path.name}: hash previo al descarte no coincide")
        discarded = max(0, len(active_ids) - pairs_kept)
        expected_ids = active_ids[-pairs_kept:]
        retained_history = history[discarded * 2:]
        if row.get("pares_descartados") != discarded or discarded <= 0:
            raise ValueError(f"{path.name}: conteo de pares descartados inválido")
        if row.get("pares_conservados") != len(expected_ids):
            raise ValueError(f"{path.name}: no conservó exactamente K pares")
        if row.get("tareas_conservadas_ids") != expected_ids:
            raise ValueError(
                f"{path.name}: no conservó los pares completos más recientes"
            )
        if row.get("historial_conservado_sha256") != sha256_text(
            canonical_json(retained_history)
        ):
            raise ValueError(f"{path.name}: hash retenido no coincide")
        history = retained_history
        active_ids = expected_ids


TRACE_REQUIRED_FIELDS = (
    "timestamp_monotonic_s",
    "timestamp_unix_s",
    "power_w",
    "vram_used_mb",
    "vram_free_mb",
    "vram_total_mb",
    "temperature_c",
    "graphics_clock_mhz",
    "sm_clock_mhz",
    "memory_clock_mhz",
    "gpu_utilization_pct",
    "memory_utilization_pct",
    "throttle_reasons_mask",
    "throttle_reasons_active",
    "performance_state",
    "process_query_available",
    "compute_pids",
    "compute_process_groups",
    "foreign_compute_pids",
)


def validate_trace_samples(
    trace: Any,
    expected_count: int,
    context: str,
    expected_process_group: int | None = None,
) -> dict[str, float]:
    """Valida evidencia NVML y devuelve agregados recalculados."""
    if not isinstance(trace, list):
        raise ValueError(f"{context}: nvml_trace no es una lista")
    if len(trace) != expected_count or expected_count < 2:
        raise ValueError(
            f"{context}: traza tiene {len(trace)} muestras; "
            f"se declararon {expected_count}"
        )
    for index, sample in enumerate(trace):
        if not isinstance(sample, dict):
            raise ValueError(f"{context}: muestra {index} no es objeto")
        missing = [key for key in TRACE_REQUIRED_FIELDS if key not in sample]
        if missing:
            raise ValueError(
                f"{context}: muestra {index} sin campos {missing}"
            )
        numeric_positive = (
            "power_w",
            "vram_total_mb",
            "temperature_c",
            "graphics_clock_mhz",
            "sm_clock_mhz",
            "memory_clock_mhz",
        )
        if any(
            sample[key] is not None and float(sample[key]) <= 0
            for key in numeric_positive
        ):
            raise ValueError(
                f"{context}: muestra {index} tiene telemetría no positiva"
            )
        for utilization_key in (
            "gpu_utilization_pct",
            "memory_utilization_pct",
        ):
            utilization = sample[utilization_key]
            if (
                utilization is not None
                and not 0 <= float(utilization) <= 100
            ):
                raise ValueError(
                    f"{context}: {utilization_key} fuera de [0,100]"
                )
        if sample["process_query_available"] is True:
            if not isinstance(sample["compute_pids"], list):
                raise ValueError(f"{context}: compute_pids no es lista")
            if not isinstance(sample["compute_process_groups"], dict):
                raise ValueError(
                    f"{context}: compute_process_groups no es objeto"
                )
            if sorted(
                int(pid) for pid in sample["compute_pids"]
            ) != sorted(
                int(pid) for pid in sample["compute_process_groups"]
            ):
                raise ValueError(
                    f"{context}: PIDs y grupos de proceso no corresponden"
                )
            recomputed_foreign = (
                sorted(
                    int(pid)
                    for pid, pgid in sample[
                        "compute_process_groups"
                    ].items()
                    if int(pgid) != expected_process_group
                )
                if expected_process_group is not None
                else []
            )
            if sorted(sample["foreign_compute_pids"]) != recomputed_foreign:
                raise ValueError(
                    f"{context}: clasificación de procesos ajenos inconsistente"
                )
        elif (
            sample["compute_pids"] is not None
            or sample["compute_process_groups"] is not None
        ):
            raise ValueError(
                f"{context}: consulta fallida contiene procesos simulados"
            )
        if sample["foreign_compute_pids"]:
            raise ValueError(
                f"{context}: procesos GPU ajenos "
                f"{sample['foreign_compute_pids']}"
            )
        if not isinstance(sample["throttle_reasons_active"], list):
            raise ValueError(
                f"{context}: throttle_reasons_active no es lista"
            )

    coverage_fields = (
        "temperature_c",
        "graphics_clock_mhz",
        "sm_clock_mhz",
        "memory_clock_mhz",
        "gpu_utilization_pct",
        "memory_utilization_pct",
        "throttle_reasons_mask",
        "performance_state",
    )
    coverage = {
        key: sum(sample[key] is not None for sample in trace) / len(trace)
        for key in coverage_fields
    }
    coverage["compute_process_query"] = sum(
        sample["process_query_available"] is True for sample in trace
    ) / len(trace)
    insufficient = {
        key: fraction
        for key, fraction in coverage.items()
        if fraction < 0.9
    }
    if insufficient:
        raise ValueError(
            f"{context}: cobertura de telemetría menor a 90%: {insufficient}"
        )

    monotonic = [float(sample["timestamp_monotonic_s"]) for sample in trace]
    unix = [float(sample["timestamp_unix_s"]) for sample in trace]
    if any(
        monotonic[index] <= monotonic[index - 1]
        for index in range(1, len(monotonic))
    ):
        raise ValueError(f"{context}: timestamps monotónicos no crecen")
    if any(
        unix[index] < unix[index - 1]
        for index in range(1, len(unix))
    ):
        raise ValueError(f"{context}: timestamps Unix retroceden")

    powers = [float(sample["power_w"]) for sample in trace]
    energy = sum(
        (powers[index] + powers[index - 1])
        / 2.0
        * (monotonic[index] - monotonic[index - 1])
        for index in range(1, len(trace))
    )
    temperatures = [
        float(sample["temperature_c"])
        for sample in trace
        if sample["temperature_c"] is not None
    ]
    return {
        "energy_j": energy,
        "duration_s": monotonic[-1] - monotonic[0],
        "avg_power_w": st.mean(powers),
        "peak_power_w": max(powers),
        "vram_peak_mb": max(
            float(sample["vram_used_mb"]) for sample in trace
        ),
        "temperature_start_c": temperatures[0],
        "temperature_end_c": temperatures[-1],
        "temperature_max_c": max(temperatures),
    }


def validate_measured_trace(
    row: dict[str, Any],
    baseline_power_w: float,
    context: str,
    expected_process_group: int,
) -> None:
    stats = validate_trace_samples(
        row.get("nvml_trace"),
        int(row.get("nvml_samples", 0)),
        context,
        expected_process_group,
    )
    comparisons = (
        ("energy_j", stats["energy_j"], 0.02),
        ("duration_s", stats["duration_s"], 0.002),
        ("avg_power_w", stats["avg_power_w"], 0.02),
        ("peak_power_w", stats["peak_power_w"], 0.02),
        ("vram_peak_mb", stats["vram_peak_mb"], 0.11),
    )
    for field, recalculated, tolerance in comparisons:
        if not close(row[field], recalculated, tolerance):
            raise ValueError(
                f"{context}: {field} no se reproduce desde la traza "
                f"({row[field]} != {recalculated})"
            )
    expected_samples = max(2, int(stats["duration_s"] / 0.1) + 1)
    sampling_fraction = int(row["nvml_samples"]) / expected_samples
    if sampling_fraction < 0.9 or not close(
        row["nvml_sampling_fraction"],
        sampling_fraction,
        0.00011,
    ):
        raise ValueError(f"{context}: cobertura temporal NVML inconsistente")
    buffer_coverage = stats["duration_s"] - float(row["request_wall_s"])
    if buffer_coverage < 0.75 or not close(
        row["nvml_buffer_coverage_s"],
        buffer_coverage,
        0.002,
    ):
        raise ValueError(f"{context}: buffers NVML no cubiertos")
    trace = row["nvml_trace"]
    dynamic = sum(
        max(
            0.0,
            (
                float(trace[index]["power_w"])
                + float(trace[index - 1]["power_w"])
            )
            / 2.0
            - baseline_power_w,
        )
        * (
            float(trace[index]["timestamp_monotonic_s"])
            - float(trace[index - 1]["timestamp_monotonic_s"])
        )
        for index in range(1, len(trace))
    )
    if not close(row["energy_above_baseline_j"], dynamic, 0.02):
        raise ValueError(
            f"{context}: energía sobre baseline no se reproduce desde traza"
        )


def validate_baseline_trace(
    manifest: dict[str, Any],
    context: str,
) -> None:
    reposo = manifest.get("reposo")
    if not isinstance(reposo, dict) or not reposo.get("disponible"):
        raise ValueError(f"{context}: baseline ausente")
    count = int(reposo.get("muestras", 0))
    expected_process_group = manifest.get("expected_server_pgid")
    if not isinstance(expected_process_group, int) or expected_process_group <= 0:
        raise ValueError(f"{context}: PGID de vLLM ausente")
    stats = validate_trace_samples(
        reposo.get("trace"),
        count,
        f"{context}/baseline",
        expected_process_group,
    )
    seconds = float(reposo.get("segundos", 0.0))
    expected = int(reposo.get("muestras_esperadas", 0))
    sample_fraction = count / expected if expected > 0 else 0.0
    if (
        seconds != 30.0
        or sample_fraction < 0.9
        or not close(reposo.get("fraccion_muestras", 0.0), sample_fraction, 0.00011)
        or stats["duration_s"] < seconds - 0.3
    ):
        raise ValueError(f"{context}: cobertura temporal de baseline inválida")
    powers = [float(sample["power_w"]) for sample in reposo["trace"]]
    if not close(reposo["potencia_reposo_media_w"], st.mean(powers), 0.002):
        raise ValueError(f"{context}: media de baseline no reproducible")
    if not close(
        reposo["potencia_reposo_mediana_w"],
        st.median(powers),
        0.002,
    ):
        raise ValueError(f"{context}: mediana de baseline no reproducible")
    process_fraction = sum(
        sample["process_query_available"] is True
        for sample in reposo["trace"]
    ) / count
    if (
        process_fraction < 0.9
        or not close(
            reposo.get("process_query_fraction", 0.0),
            process_fraction,
            0.00011,
        )
    ):
        raise ValueError(f"{context}: baseline sin suficiente control de procesos")
    if reposo.get("foreign_compute_pids"):
        raise ValueError(f"{context}: baseline contaminado")


def validate_session(
    path: Path,
    rows: list[dict[str, Any]],
    expected: tuple[str, str, int],
    expected_tasks: int,
    max_model_len: int,
    pairs_kept: int,
) -> dict[str, Any]:
    quant, arm, rep = expected
    manifest_path = Path(str(path) + ".manifiesto.json")
    if not manifest_path.is_file():
        raise ValueError(f"{path.name}: falta manifiesto de sesión")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"{path.name}: manifiesto no completo")
    if manifest.get("raw_sha256") != sha256_file(path):
        raise ValueError(f"{path.name}: SHA-256 no coincide con manifiesto")
    if (
        manifest.get("quant"),
        manifest.get("brazo"),
        manifest.get("rep"),
    ) != expected:
        raise ValueError(f"{path.name}: identidad de manifiesto incorrecta")
    if manifest.get("schema_version") != 3:
        raise ValueError(f"{path.name}: schema de manifiesto no es 3")
    probe = manifest.get("telemetry_probe")
    if not isinstance(probe, dict) or probe.get("complete") is not True:
        raise ValueError(f"{path.name}: preflight de telemetría incompleto")
    validate_baseline_trace(manifest, path.name)
    warmup_records = manifest.get("warmup_records")
    warmup_requested = int(manifest.get("calentamiento_solicitado", -1))
    if (
        not isinstance(warmup_records, list)
        or len(warmup_records) != warmup_requested
        or manifest.get("calentamiento_completado") != warmup_requested
    ):
        raise ValueError(f"{path.name}: warmup incompleto/no auditable")
    if any(
        record.get("finish_reason") == "length"
        or int(record.get("prompt_tokens", 0)) <= 0
        for record in warmup_records
    ):
        raise ValueError(f"{path.name}: warmup inválido")

    for line_number, row in enumerate(rows, start=1):
        if (row.get("quant"), row.get("arm"), row.get("rep")) != expected:
            raise ValueError(
                f"{path.name}:{line_number}: identidad de fila incorrecta"
            )
        if row.get("status") != "ok":
            raise ValueError(f"{path.name}:{line_number}: status no es ok")
        if row.get("schema_version") != 3:
            raise ValueError(
                f"{path.name}:{line_number}: schema_version no es 3"
            )
        if row.get("run_id") != f"{quant}_{arm}_rep{rep}":
            raise ValueError(f"{path.name}:{line_number}: run_id inconsistente")
        if row.get("threshold_tokens") != EXPECTED_THRESHOLD:
            raise ValueError(f"{path.name}:{line_number}: umbral inconsistente")
        if row.get("mechanism_energy_j", 0.0) and arm != "resumen":
            raise ValueError(
                f"{path.name}:{line_number}: costo de mecanismo fuera de resumen"
            )
        is_event = row.get("task_type") in EVENT_TYPES
        if bool(row.get("is_mechanism_event")) != is_event:
            raise ValueError(
                f"{path.name}:{line_number}: etiqueta de evento inconsistente"
            )

    task_rows = tareas(rows)
    event_rows = eventos(rows)
    if len(task_rows) != expected_tasks:
        raise ValueError(
            f"{path.name}: {len(task_rows)} tareas; se esperaban {expected_tasks}"
        )
    if [row["task_index"] for row in task_rows] != list(range(expected_tasks)):
        raise ValueError(f"{path.name}: índices de tarea incompletos/desordenados")

    cumulative = 0.0
    cumulative_mechanism = 0.0
    session_baseline = float(
        manifest["reposo"]["potencia_reposo_mediana_w"]
    )
    for line_number, row in enumerate(rows, start=1):
        energy = float(row["energy_j"])
        mechanism_energy = float(row.get("mechanism_energy_j", 0.0))
        if energy < 0 or mechanism_energy < 0:
            raise ValueError(f"{path.name}:{line_number}: energía negativa")
        cumulative += energy
        cumulative_mechanism += mechanism_energy
        if not close(row["cumulative_energy_j"], cumulative):
            raise ValueError(
                f"{path.name}:{line_number}: acumulado de energía inconsistente"
            )
        if not close(
            row.get("cumulative_mechanism_energy_j", 0.0),
            cumulative_mechanism,
        ):
            raise ValueError(
                f"{path.name}:{line_number}: acumulado de mecanismo inconsistente"
            )
        if not close(row["baseline_power_w"], session_baseline, 0.001):
            raise ValueError(
                f"{path.name}:{line_number}: baseline no es el de la sesión"
            )

        if row["task_type"] == "DESCARTE":
            if row.get("nvml_trace") != []:
                raise ValueError(
                    f"{path.name}:{line_number}: descarte simula traza NVML"
                )
            continue
        validate_measured_trace(
            row,
            session_baseline,
            f"{path.name}:{line_number}",
            manifest["expected_server_pgid"],
        )
        if not isinstance(row.get("finish_reason"), str):
            raise ValueError(
                f"{path.name}:{line_number}: finish_reason ausente"
            )
        if row.get("vllm_response_model") != f"infera-{quant.lower()}":
            raise ValueError(
                f"{path.name}:{line_number}: modelo de respuesta inconsistente"
            )
        if row["task_type"] == "COMPACTION":
            continue
        requested = int(row["requested_max_tokens"])
        prompt = int(row["accumulated_prompt_tokens"])
        preflight = int(row["preflight_prompt_tokens"])
        if prompt != preflight:
            raise ValueError(
                f"{path.name}:{line_number}: preflight != usage"
            )
        if prompt + requested > max_model_len:
            raise ValueError(
                f"{path.name}:{line_number}: {prompt}+{requested} "
                f"> {max_model_len}"
            )
        if row.get("max_model_len") != max_model_len:
            raise ValueError(
                f"{path.name}:{line_number}: max_model_len inconsistente"
            )
        if not row.get("token_budget_ok"):
            raise ValueError(f"{path.name}:{line_number}: budget no aprobado")
        if row.get("quality_is_programmatic") is not True:
            raise ValueError(
                f"{path.name}:{line_number}: calidad no es programática"
            )
        quality = float(row["quality"])
        if not 0.0 <= quality <= 1.0:
            raise ValueError(
                f"{path.name}:{line_number}: puntaje fuera de [0,1]"
            )
        if not row.get("nvml_available") or int(row["nvml_samples"]) < 2:
            raise ValueError(f"{path.name}:{line_number}: NVML inválido")
        if energy <= 0:
            raise ValueError(f"{path.name}:{line_number}: tarea con 0 J")
        if int(row["completion_tokens"]) > requested:
            raise ValueError(
                f"{path.name}:{line_number}: salida excede max_tokens"
            )

    if arm == "completo":
        if event_rows:
            raise ValueError(f"{path.name}: completo no debe intervenir")
    elif arm == "resumen":
        if not event_rows or any(
            row["task_type"] != "COMPACTION" for row in event_rows
        ):
            raise ValueError(f"{path.name}: eventos de resumen inválidos")
        for row in event_rows:
            prompt = int(row["accumulated_prompt_tokens"])
            requested = int(row["requested_max_tokens"])
            if prompt != int(row["preflight_prompt_tokens"]):
                raise ValueError(f"{path.name}: preflight de resumen no coincide")
            if prompt + requested > max_model_len:
                raise ValueError(f"{path.name}: resumen excedió presupuesto")
            if row.get("max_model_len") != max_model_len:
                raise ValueError(
                    f"{path.name}: max_model_len de resumen inconsistente"
                )
            if not row.get("token_budget_ok"):
                raise ValueError(f"{path.name}: resumen sin gate de presupuesto")
            if int(row["completion_tokens"]) > requested:
                raise ValueError(f"{path.name}: resumen excedió max_tokens")
            if float(row["energy_j"]) <= 0:
                raise ValueError(f"{path.name}: resumen sin costo energético")
            if row.get("is_compaction") is not True:
                raise ValueError(f"{path.name}: resumen mal etiquetado")
            if not close(row["mechanism_energy_j"], row["energy_j"], 0.001):
                raise ValueError(f"{path.name}: costo de resumen mal etiquetado")
            if not row.get("nvml_available") or int(row["nvml_samples"]) < 2:
                raise ValueError(f"{path.name}: NVML inválido en resumen")
    else:
        if not event_rows or any(
            row["task_type"] != "DESCARTE" for row in event_rows
        ):
            raise ValueError(f"{path.name}: eventos de descarte inválidos")
        for row in event_rows:
            zero_fields = (
                "energy_j",
                "mechanism_energy_j",
                "energy_above_baseline_j",
                "accumulated_prompt_tokens",
                "completion_tokens",
                "requested_max_tokens",
            )
            if any(float(row[field]) != 0.0 for field in zero_fields):
                raise ValueError(
                    f"{path.name}: descarte acumuló energía/tokens inexistentes"
                )
            if row.get("nvml_samples") is not None:
                raise ValueError(f"{path.name}: descarte simula muestras NVML")
            if row.get("es_descarte") is not True:
                raise ValueError(f"{path.name}: descarte mal etiquetado")
        validate_discard_selection(path, rows, pairs_kept)

    if not close(manifest["energia_total_j"], cumulative):
        raise ValueError(f"{path.name}: energía total no coincide con manifiesto")
    if not close(
        manifest["energia_mecanismo_j"],
        cumulative_mechanism,
    ):
        raise ValueError(
            f"{path.name}: energía de mecanismo no coincide con manifiesto"
        )
    if manifest.get("tareas_medidas") != expected_tasks:
        raise ValueError(f"{path.name}: manifiesto no declara 29 tareas")
    if manifest.get("intervenciones") != len(event_rows):
        raise ValueError(f"{path.name}: intervenciones no coinciden")
    if manifest.get("umbral_tokens") != EXPECTED_THRESHOLD:
        raise ValueError(f"{path.name}: umbral de manifiesto inconsistente")
    if arm in ("resumen", "descarte"):
        for index, row in enumerate(rows):
            if row["task_type"] not in EVENT_TYPES:
                continue
            trigger = row.get(
                "trigger_prompt_tokens",
                row.get("context_tokens_before_intervention"),
            )
            if trigger is None or int(trigger) < EXPECTED_THRESHOLD:
                raise ValueError(f"{path.name}: intervención sin trigger válido")
            next_task = next(
                (
                    candidate
                    for candidate in rows[index + 1:]
                    if candidate["task_type"] not in EVENT_TYPES
                ),
                None,
            )
            if (
                next_task is not None
                and int(next_task["accumulated_prompt_tokens"])
                >= EXPECTED_THRESHOLD
            ):
                raise ValueError(
                    f"{path.name}: la intervención no bajó del umbral"
                )
    return manifest


def cargar(
    crudos: Path,
    expected_tasks: int,
    max_model_len: int,
    pairs_kept: int,
) -> tuple[
    dict[tuple[str, str, int], list[dict[str, Any]]],
    dict[tuple[str, str, int], dict[str, Any]],
]:
    campaign, preflight, scenario_tasks = load_campaign_preflight(
        crudos,
        expected_tasks,
        max_model_len,
        pairs_kept,
    )
    expected_paths = {
        f"run_{quant}_{arm}_rep{rep}.jsonl": (quant, arm, rep)
        for quant in QUANTS
        for arm in BRAZOS
        for rep in REPS
    }
    found = {path.name: path for path in crudos.glob("run_*.jsonl")}
    missing = sorted(set(expected_paths) - set(found))
    extras = sorted(set(found) - set(expected_paths))
    if missing or extras:
        raise ValueError(
            f"campaña incompleta/contaminada; faltan={missing}, extras={extras}"
        )
    partials = sorted(path.name for path in crudos.glob("*.partial"))
    if partials:
        raise ValueError(f"quedaron raws parciales: {partials}")

    sessions = {}
    manifests = {}
    canonical_task_ids = None
    for name, expected in expected_paths.items():
        rows = read_jsonl(found[name])
        manifest = validate_session(
            found[name],
            rows,
            expected,
            expected_tasks,
            max_model_len,
            pairs_kept,
        )
        ids = [row["task_id"] for row in tareas(rows)]
        task_rows = tareas(rows)
        for index, (row, task) in enumerate(zip(task_rows, scenario_tasks)):
            expected_task = (
                task.get("id"),
                task.get("type"),
                task.get("prompt"),
            )
            observed_task = (
                row.get("task_id"),
                row.get("task_type"),
                row.get("prompt_text"),
            )
            if observed_task != expected_task:
                raise ValueError(
                    f"{name}: tarea {index} no coincide con el escenario"
                )
            recalculated_quality = score_task(
                row.get("response_text", ""),
                task.get("verify", {}),
                judge_fn=None,
            )
            if not close(
                row.get("quality"),
                recalculated_quality["quality"],
                0.00001,
            ) or row.get("quality_subscores") != recalculated_quality[
                "subscores"
            ]:
                raise ValueError(
                    f"{name}: calidad de tarea {index} no se reproduce "
                    "desde respuesta y rúbrica"
                )
        if canonical_task_ids is None:
            canonical_task_ids = ids
        elif ids != canonical_task_ids:
            raise ValueError(f"{name}: secuencia de tareas divergente")
        expected_manifest = {
            "escenario_sha256": preflight.get("tasks_sha256"),
            "kb_sha256": preflight.get("kb_sha256"),
            "runner_sha256": preflight["code_sha256"].get(
                "infera_session_runner.py"
            ),
            "expected_tasks": expected_tasks,
            "max_model_len": max_model_len,
            "max_tokens": preflight.get("requested_max_tokens"),
            "temperatura": preflight.get("temperature"),
            "semilla": preflight.get("seed"),
            "post_warmup_settle_s": preflight.get(
                "post_warmup_settle_seconds"
            ),
            "calentamiento_solicitado": preflight.get("warmup_count"),
            "request_timeout_s": preflight.get("request_timeout_seconds"),
        }
        for field, expected_value in expected_manifest.items():
            if manifest.get(field) != expected_value:
                raise ValueError(
                    f"{name}: {field} no coincide con el preflight"
                )
        if fingerprint_subset(manifest.get("tokenizer")) != fingerprint_subset(
            preflight["tokenizers"].get(expected[0])
        ):
            raise ValueError(f"{name}: tokenizer no coincide con el preflight")
        if manifest.get("modelo_fuente") != preflight["models"][
            expected[0]
        ].get("reference"):
            raise ValueError(f"{name}: modelo fuente no coincide con preflight")
        if manifest.get("modelo_servido") != f"infera-{expected[0].lower()}":
            raise ValueError(f"{name}: nombre de modelo servido inconsistente")
        campaign_gpu = campaign.get("gpu")
        session_gpu = manifest.get("gpu")
        if not isinstance(campaign_gpu, dict) or not campaign_gpu.get("uuid"):
            raise ValueError("el manifiesto de campaña no identifica la GPU")
        campaign_probe = campaign_gpu.get("telemetry_probe")
        if (
            not isinstance(campaign_probe, dict)
            or campaign_probe.get("complete") is not True
        ):
            raise ValueError("la campaña inició sin telemetría NVML completa")
        stable_gpu_fields = (
            "name",
            "uuid",
            "driver_version",
            "nvml_version",
            "memory_total_bytes",
            "power_limit_w",
        )
        if not isinstance(session_gpu, dict) or any(
            session_gpu.get(field) != campaign_gpu.get(field)
            for field in stable_gpu_fields
        ):
            raise ValueError(f"{name}: la GPU cambió durante la campaña")
        sessions[expected] = rows
        manifests[expected] = manifest

    schedule = [
        (item["quant"], item["arm"], item["rep"])
        for item in campaign.get("schedule", [])
    ]
    actual = [
        key
        for key, _ in sorted(
            manifests.items(),
            key=lambda item: item[1]["iniciado_utc"],
        )
    ]
    if schedule != actual:
        raise ValueError(
            f"orden real no coincide con schedule; esperado={schedule}, real={actual}"
        )
    return sessions, manifests


def por_sesion(
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
    manifests: (
        dict[tuple[str, str, int], dict[str, Any]] | None
    ) = None,
) -> list[dict[str, Any]]:
    output = []
    for (quant, arm, rep), rows in sorted(sessions.items()):
        task_rows, event_rows = tareas(rows), eventos(rows)
        task_energy = sum(float(row["energy_j"]) for row in task_rows)
        mechanism_energy = sum(
            float(row["mechanism_energy_j"]) for row in event_rows
        )
        dynamic = sum(
            float(row.get("energy_above_baseline_j") or 0.0)
            for row in rows
        )
        quality = [float(row["quality"]) for row in task_rows]
        measured_rows = [
            row for row in rows if row["task_type"] != "DESCARTE"
        ]
        trace = [
            sample
            for row in measured_rows
            for sample in row["nvml_trace"]
        ]
        baseline = (
            manifests[(quant, arm, rep)]["reposo"]
            if manifests is not None
            else None
        )
        baseline_trace = baseline["trace"] if baseline is not None else []
        baseline_temperatures = [
            float(sample["temperature_c"])
            for sample in baseline_trace
            if sample["temperature_c"] is not None
        ]
        request_temperatures = [
            float(sample["temperature_c"])
            for sample in trace
            if sample["temperature_c"] is not None
        ]
        gpu_utilizations = [
            float(sample["gpu_utilization_pct"])
            for sample in trace
            if sample["gpu_utilization_pct"] is not None
        ]
        graphics_clocks = [
            int(sample["graphics_clock_mhz"])
            for sample in trace
            if sample["graphics_clock_mhz"] is not None
        ]
        sm_clocks = [
            int(sample["sm_clock_mhz"])
            for sample in trace
            if sample["sm_clock_mhz"] is not None
        ]
        memory_clocks = [
            int(sample["memory_clock_mhz"])
            for sample in trace
            if sample["memory_clock_mhz"] is not None
        ]
        output.append({
            "quant": quant,
            "brazo": arm,
            "rep": rep,
            "tareas": len(task_rows),
            "eventos": len(event_rows),
            "energia_tareas_j": round(task_energy, 4),
            "energia_mecanismo_j": round(mechanism_energy, 4),
            "energia_total_j": round(task_energy + mechanism_energy, 4),
            "energia_sobre_reposo_j": round(dynamic, 4),
            "tareas_score_programatico_1": sum(
                1 for score in quality if score >= 1.0
            ),
            "puntaje_programatico_medio": round(st.mean(quality), 4),
            "tokens_entrada_de_peticiones": sum(
                int(row["accumulated_prompt_tokens"]) for row in rows
            ),
            "tokens_salida_de_peticiones": sum(
                int(row["completion_tokens"]) for row in rows
            ),
            "prompt_min_tareas": min(
                int(row["accumulated_prompt_tokens"]) for row in task_rows
            ),
            "prompt_max_tareas": max(
                int(row["accumulated_prompt_tokens"]) for row in task_rows
            ),
            "llamadas_terminadas_por_max_tokens": sum(
                row.get("finish_reason") == "length"
                for row in measured_rows
            ),
            "muestras_nvml_llamadas": len(trace),
            "cobertura_temperatura_pct": round(
                100 * len(request_temperatures) / len(trace), 3
            ),
            "cobertura_procesos_pct": round(
                100
                * sum(
                    sample["process_query_available"] is True
                    for sample in trace
                )
                / len(trace),
                3,
            ),
            "temperatura_llamadas_inicio_c": request_temperatures[0],
            "temperatura_llamadas_fin_c": request_temperatures[-1],
            "temperatura_llamadas_max_c": max(request_temperatures),
            "gpu_utilizacion_media_pct": round(
                st.mean(gpu_utilizations),
                3,
            ),
            "graphics_clock_min_mhz": min(graphics_clocks),
            "graphics_clock_max_mhz": max(graphics_clocks),
            "sm_clock_min_mhz": min(sm_clocks),
            "sm_clock_max_mhz": max(sm_clocks),
            "memory_clock_min_mhz": min(memory_clocks),
            "memory_clock_max_mhz": max(memory_clocks),
            "muestras_clock_event_mask_no_cero": sum(
                sample["throttle_reasons_mask"] is not None
                and int(sample["throttle_reasons_mask"]) != 0
                for sample in trace
            ),
            "baseline_muestras_nvml": len(baseline_trace) or None,
            "baseline_temperatura_inicio_c": (
                baseline_temperatures[0] if baseline_temperatures else None
            ),
            "baseline_temperatura_fin_c": (
                baseline_temperatures[-1] if baseline_temperatures else None
            ),
            "baseline_temperatura_media_c": (
                round(st.mean(baseline_temperatures), 3)
                if baseline_temperatures else None
            ),
            "baseline_temperatura_rango_c": (
                round(
                    max(baseline_temperatures)
                    - min(baseline_temperatures),
                    3,
                )
                if baseline_temperatures else None
            ),
            "baseline_potencia_cv_pct": (
                baseline.get("potencia_reposo_cv_pct")
                if baseline is not None else None
            ),
        })
    return output


def agregado(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["quant"], row["brazo"])].append(row)
    output = []
    for (quant, arm), group in sorted(groups.items()):
        totals = [row["energia_total_j"] for row in group]
        successes = [
            row["tareas_score_programatico_1"] for row in group
        ]
        output.append({
            "quant": quant,
            "brazo": arm,
            "nombre": NOMBRE[arm],
            "repeticiones_instrumentales": len(group),
            "energia_tareas_media_j": round(
                st.mean(row["energia_tareas_j"] for row in group), 2
            ),
            "energia_mecanismo_media_j": round(
                st.mean(row["energia_mecanismo_j"] for row in group), 2
            ),
            "energia_total_media_j": round(st.mean(totals), 2),
            "energia_total_min_j": round(min(totals), 2),
            "energia_total_max_j": round(max(totals), 2),
            "cv_instrumental_pct": round(
                100 * st.stdev(totals) / st.mean(totals), 3
            ),
            "energia_sobre_reposo_media_j": round(
                st.mean(
                    row["energia_sobre_reposo_j"] for row in group
                ),
                2,
            ),
            "score_1_media_tareas": round(st.mean(successes), 2),
            "score_1_min_tareas": min(successes),
            "score_1_max_tareas": max(successes),
            "puntaje_programatico_medio": round(
                st.mean(
                    row["puntaje_programatico_medio"] for row in group
                ),
                4,
            ),
        })
    return output


def efecto_por_tarea(
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Diagnóstico descriptivo por repetición; no crea n independientes."""
    output = []
    for quant in QUANTS:
        for rep in REPS:
            maps = {
                arm: {
                    row["task_id"]: row
                    for row in tareas(sessions[(quant, arm, rep)])
                }
                for arm in BRAZOS
            }
            for task_id, base in maps["completo"].items():
                row = {
                    "quant": quant,
                    "rep_instrumental": rep,
                    "task_id": task_id,
                    "task_type": base["task_type"],
                    "q_completo": base["quality"],
                }
                for arm in ("resumen", "descarte"):
                    score = maps[arm][task_id]["quality"]
                    base_score = base["quality"]
                    row[f"q_{arm}"] = score
                    if base_score >= 1.0 and score >= 1.0:
                        effect = "preserva"
                    elif base_score >= 1.0 and score < 1.0:
                        effect = "perjudica"
                    elif base_score < 1.0 and score >= 1.0:
                        effect = "repara"
                    else:
                        effect = "ambos fallan"
                    row[f"efecto_{arm}"] = effect
                output.append(row)
    return output


def ocupacion_post_intervencion(
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    output = []
    for (quant, arm, rep), rows in sorted(sessions.items()):
        for index, row in enumerate(rows):
            if row["task_type"] not in EVENT_TYPES:
                continue
            next_task = next(
                (
                    candidate
                    for candidate in rows[index + 1:]
                    if candidate["task_type"] not in EVENT_TYPES
                ),
                None,
            )
            output.append({
                "quant": quant,
                "brazo": arm,
                "rep": rep,
                "intervencion": row["intervention_index"],
                "tipo": row["task_type"],
                "trigger_prompt_tokens": row.get(
                    "trigger_prompt_tokens",
                    row.get("context_tokens_before_intervention"),
                ),
                "siguiente_tarea": (
                    next_task["task_id"] if next_task else None
                ),
                "siguiente_prompt_tokens": (
                    next_task["accumulated_prompt_tokens"]
                    if next_task
                    else None
                ),
            })
    return output


def escribir_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"no hay filas para {path.name}")
    with path.open("x", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        fh.flush()
        os.fsync(fh.fileno())


def escribir_texto(path: Path, value: str) -> None:
    with path.open("x", encoding="utf-8") as fh:
        fh.write(value)
        fh.flush()
        os.fsync(fh.fileno())


def analysis_code_provenance(crudos: Path) -> dict[str, Any]:
    campaign_path = crudos.parent / "manifiesto_campana.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    preflight_path = Path(campaign["preflight"])
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    frozen = preflight["code_sha256"]
    current = {}
    drift = []
    for name, frozen_hash in sorted(frozen.items()):
        path = ROOT / name
        current_hash = sha256_file(path) if path.is_file() else None
        current[name] = current_hash
        if current_hash != frozen_hash:
            drift.append({
                "name": name,
                "preflight_sha256": frozen_hash,
                "analysis_sha256": current_hash,
            })
    return {
        "campaign_status_at_analysis": campaign.get("status"),
        "analysis_entrypoint_sha256": sha256_file(Path(__file__)),
        "preflight_code_sha256": frozen,
        "current_code_sha256": current,
        "code_drift": drift,
    }


def informe(
    aggregate: list[dict[str, Any]],
    effects: list[dict[str, Any]],
) -> str:
    lines = [
        "# Campaña de tres políticas de gestión del historial",
        "",
        "Energía y puntaje programático se informan por separado. Las tres",
        "pasadas son repeticiones instrumentales de una trayectoria fija;",
        "no son réplicas independientes de tareas o calidad.",
        "",
        "## Energía media por sesión",
        "",
        "| Precisión | Política | Tareas, J | Mecanismo, J | Total, J | Sobre reposo, J | CV instrumental % |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            f"| {row['quant']} | {row['nombre']} | "
            f"{row['energia_tareas_media_j']:.2f} | "
            f"{row['energia_mecanismo_media_j']:.2f} | "
            f"{row['energia_total_media_j']:.2f} | "
            f"{row['energia_sobre_reposo_media_j']:.2f} | "
            f"{row['cv_instrumental_pct']:.3f} |"
        )
    lines.extend(("", "## Contraste primario resumen − descarte", ""))
    for quant in QUANTS:
        summary = next(
            row for row in aggregate
            if row["quant"] == quant and row["brazo"] == "resumen"
        )
        discard = next(
            row for row in aggregate
            if row["quant"] == quant and row["brazo"] == "descarte"
        )
        delta = (
            summary["energia_total_media_j"]
            - discard["energia_total_media_j"]
        )
        lines.extend((
            f"- **{quant}: diferencia end-to-end {delta:+.2f} J.**",
            f"  Costo directo medio de llamadas de resumen: "
            f"{summary['energia_mecanismo_media_j']:.2f} J; descarte local: 0 J.",
            f"  Tareas con puntaje programático 1: resumen "
            f"{summary['score_1_media_tareas']:.2f} "
            f"(rango {summary['score_1_min_tareas']}–"
            f"{summary['score_1_max_tareas']}), descarte "
            f"{discard['score_1_media_tareas']:.2f} "
            f"(rango {discard['score_1_min_tareas']}–"
            f"{discard['score_1_max_tareas']}).",
        ))
    lines.extend((
        "",
        "La diferencia end-to-end incluye la llamada de resumen, la ocupación",
        "realizada, el contenido retenido, las respuestas y su propagación.",
        "No es solo el precio del mecanismo.",
        "",
        "## Diagnóstico por tarea frente a completo",
        "",
    ))
    counts = defaultdict(int)
    for row in effects:
        for arm in ("resumen", "descarte"):
            counts[
                (row["quant"], row["rep_instrumental"], arm, row[f"efecto_{arm}"])
            ] += 1
    for quant in QUANTS:
        for rep in REPS:
            for arm in ("resumen", "descarte"):
                values = [
                    f"{effect}: {counts[(quant, rep, arm, effect)]}"
                    for effect in (
                        "preserva",
                        "perjudica",
                        "repara",
                        "ambos fallan",
                    )
                ]
                lines.append(
                    f"- {quant}, repetición {rep}, {NOMBRE[arm]} — "
                    + ", ".join(values)
                )
    lines.extend((
        "",
        "## Diagnóstico de medición",
        "",
        "La traza NVML cruda de cada llamada y baseline fue validada y permite",
        "recalcular energía, duración, potencia y memoria. `por_sesion.csv`",
        "incluye temperatura, clocks, utilización, eventos de clock y salidas",
        "terminadas por `max_tokens`. Cualquier proceso GPU fuera del grupo de",
        "vLLM invalida la sesión.",
        "",
        "## Alcance y amenazas",
        "",
        "Estas cifras describen una trayectoria sintética, un orden, K=4,",
        "un umbral, dos checkpoints, una GPU y vLLM sin caché de prefijos.",
        "K=4 es un parámetro libre sin análisis de sensibilidad. Un umbral",
        "común no iguala ocupación, y los prompts divergen al acumular outputs.",
        "El descarte por antigüedad es una línea base débil, no el estado del arte.",
        "El puntaje es programático; tareas con reglas semánticas abiertas no",
        "equivalen a una evaluación humana de éxito.",
        "No se estiman valores p, población, umbral óptimo, política universal,",
        "energía por fase ni generalización a otros sistemas.",
    ))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate y análisis de la campaña de tres brazos"
    )
    parser.add_argument("--crudos", type=Path, required=True)
    parser.add_argument("--salida", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=29)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--pairs-kept", type=int, default=4)
    args = parser.parse_args()
    if not args.crudos.is_dir():
        raise SystemExit(f"no existe la carpeta {args.crudos}")
    if args.salida.exists():
        raise SystemExit(f"no se sobrescribirá el análisis: {args.salida}")

    try:
        sessions, manifests = cargar(
            args.crudos,
            args.expected_tasks,
            args.max_model_len,
            args.pairs_kept,
        )
    except ValueError as exc:
        raise SystemExit(f"campaña inválida: {exc}") from exc

    session_rows = por_sesion(sessions, manifests)
    aggregate = agregado(session_rows)
    effects = efecto_por_tarea(sessions)
    occupancy = ocupacion_post_intervencion(sessions)
    text = informe(aggregate, effects)
    provenance = analysis_code_provenance(args.crudos)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.salida.parent / (
        f".{args.salida.name}.{uuid.uuid4().hex}.partial"
    )
    temporary.mkdir()
    try:
        escribir_csv(temporary / "por_sesion.csv", session_rows)
        escribir_csv(temporary / "agregado.csv", aggregate)
        escribir_csv(temporary / "efecto_por_tarea.csv", effects)
        escribir_csv(
            temporary / "ocupacion_post_intervencion.csv",
            occupancy,
        )
        escribir_texto(temporary / "informe.md", text)
        analysis_manifest = {
            "schema_version": 2,
            "sessions": len(sessions),
            "rows": sum(len(rows) for rows in sessions.values()),
            "expected_tasks_per_session": args.expected_tasks,
            "population_inference": False,
            "p_values": False,
            "raw_nvml_traces_validated": True,
            "energy_recalculated_from_trace": True,
            "continuous_process_exclusivity_validated": True,
            **provenance,
            "raw_sha256": {
                path.name: sha256_file(path)
                for path in sorted(args.crudos.glob("run_*.jsonl"))
            },
            "session_manifest_sha256": {
                path.name: sha256_file(path)
                for path in sorted(
                    args.crudos.glob("run_*.jsonl.manifiesto.json")
                )
            },
        }
        escribir_texto(
            temporary / "manifiesto_analisis.json",
            json.dumps(
                analysis_manifest,
                ensure_ascii=False,
                indent=2,
            ) + "\n",
        )
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, args.salida)
        parent_fd = os.open(args.salida.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except BaseException:
        raise

    print(f"Campaña válida: {len(sessions)} sesiones")
    print(f"Productos completos: {args.salida}")
    print()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
