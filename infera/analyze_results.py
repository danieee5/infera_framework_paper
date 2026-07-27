#!/usr/bin/env python3
"""Analiza una comparación INFERA desde los JSONL declarados.

No modifica datos crudos. Produce un manifiesto con hashes, validaciones y
tablas derivadas deterministas. Los valores predeterminados procesan el
conjunto de referencia publicado; los parámetros permiten analizar otra
ejecución del mismo diseño de dos estrategias.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - se valida explícitamente en main
    Image = ImageDraw = ImageFont = None


QUANTS = ("AWQ", "FP16")
ARMS = ("naive", "compaction")
REPS = (1, 2, 3)
SESSION_TAG = "v3"
EXPECTED_TASKS = 29
EXPECTED_COMPACTIONS = 3
REQUIRED_FIELDS = (
    "run_id",
    "quant",
    "arm",
    "rep",
    "task_index",
    "task_id",
    "task_type",
    "accumulated_prompt_tokens",
    "completion_tokens",
    "energy_j",
    "quality",
    "cumulative_energy_j",
    "is_compaction",
    "status",
    "nvml_available",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(source: Path) -> list[tuple[str, str, int, Path]]:
    return [
        (
            quant,
            arm,
            rep,
            source / f"run_{SESSION_TAG}_{quant}_{arm}_rep{rep}.jsonl",
        )
        for quant in QUANTS
        for arm in ARMS
        for rep in REPS
    ]


def canonical_index(value: Any) -> str:
    return f"{float(value):g}"


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized) if materialized else math.nan


def sample_sd(values: Iterable[float]) -> float:
    materialized = list(values)
    return statistics.stdev(materialized) if len(materialized) >= 2 else 0.0


def rounded(value: float | int | None, digits: int = 4) -> float | int | None:
    if value is None:
        return None
    return round(float(value), digits)


def read_runs(source: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    errors: list[str] = []

    for quant, arm, rep, path in expected_files(source):
        if not path.exists():
            errors.append(f"Falta archivo esperado: {path}")
            continue

        file_hash = sha256(path)
        file_rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"{path.name}:{line_number}: JSON inválido: {exc}")
                continue
            missing = [field for field in REQUIRED_FIELDS if field not in row]
            if missing:
                errors.append(
                    f"{path.name}:{line_number}: campos faltantes {', '.join(missing)}"
                )
            row["_source_file"] = path.name
            row["_source_line"] = line_number
            row["_source_sha256"] = file_hash
            file_rows.append(row)

            if row.get("quant") != quant:
                errors.append(
                    f"{path.name}:{line_number}: quant={row.get('quant')} != {quant}"
                )
            if row.get("arm") != arm:
                errors.append(
                    f"{path.name}:{line_number}: arm={row.get('arm')} != {arm}"
                )
            if int(row.get("rep", -1)) != rep:
                errors.append(
                    f"{path.name}:{line_number}: rep={row.get('rep')} != {rep}"
                )

        files.append(
            {
                "path": str(path),
                "name": path.name,
                "sha256": file_hash,
                "bytes": path.stat().st_size,
                "rows": len(file_rows),
                "contains_prompt_text": any("prompt_text" in row for row in file_rows),
                "contains_response_text": any(
                    "response_text" in row for row in file_rows
                ),
            }
        )
        rows.extend(file_rows)

    return rows, files, errors


def group_runs(rows: list[dict[str, Any]]) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["quant"]), str(row["arm"]), int(row["rep"]))
        grouped[key].append(row)
    for run_rows in grouped.values():
        run_rows.sort(key=lambda row: float(row["task_index"]))
    return dict(grouped)


def validate_rows(
    rows: list[dict[str, Any]],
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    expected_keys = {(quant, arm, rep) for quant in QUANTS for arm in ARMS for rep in REPS}
    present_keys = set(grouped)

    for key in sorted(expected_keys - present_keys):
        errors.append(f"Falta corrida: {key}")
    for key in sorted(present_keys - expected_keys):
        warnings.append(f"Corrida fuera del conjunto declarado: {key}")

    seen_row_keys: set[tuple[str, str, int, str, str]] = set()
    reference_sequence: list[str] | None = None

    for key in sorted(expected_keys & present_keys):
        quant, arm, rep = key
        run_rows = grouped[key]
        ordinary = [row for row in run_rows if not bool(row["is_compaction"])]
        compactions = [row for row in run_rows if bool(row["is_compaction"])]

        if len(ordinary) != EXPECTED_TASKS:
            errors.append(
                f"{key}: {len(ordinary)} tareas ordinarias; "
                f"se esperaban {EXPECTED_TASKS}"
            )
        expected_compactions = EXPECTED_COMPACTIONS if arm == "compaction" else 0
        if expected_compactions >= 0 and len(compactions) != expected_compactions:
            errors.append(
                f"{key}: {len(compactions)} compactaciones; "
                f"se esperaban {expected_compactions}"
            )

        sequence = [str(row["task_id"]) for row in ordinary]
        if reference_sequence is None:
            reference_sequence = sequence
        elif sequence != reference_sequence:
            errors.append(f"{key}: secuencia de tareas distinta al referente")

        running_energy = 0.0
        for row in run_rows:
            row_key = (
                quant,
                arm,
                rep,
                canonical_index(row["task_index"]),
                str(row["task_id"]),
            )
            if row_key in seen_row_keys:
                errors.append(f"Fila duplicada: {row_key}")
            seen_row_keys.add(row_key)

            if row.get("status") != "ok":
                errors.append(f"{row_key}: status={row.get('status')}")
            if not bool(row.get("nvml_available")):
                errors.append(f"{row_key}: NVML no disponible")
            if float(row.get("energy_j", 0.0)) <= 0:
                errors.append(f"{row_key}: energía no positiva")

            running_energy += float(row["energy_j"])
            recorded = float(row["cumulative_energy_j"])
            if not math.isclose(running_energy, recorded, abs_tol=0.15):
                errors.append(
                    f"{row_key}: acumulado {recorded:.4f} != suma {running_energy:.4f}"
                )

    if EXPECTED_COMPACTIONS >= 0:
        expected_rows = len(QUANTS) * len(REPS) * (
            len(ARMS) * EXPECTED_TASKS + EXPECTED_COMPACTIONS
        )
        if len(rows) != expected_rows:
            errors.append(
                f"Total de filas {len(rows)}; se esperaban {expected_rows}"
            )

    has_response_text = bool(
        rows and all("response_text" in row for row in rows)
    )
    has_prompt_text = bool(rows and all("prompt_text" in row for row in rows))
    if rows and not has_response_text:
        warnings.append(
            "Los raws declarados no contienen response_text; la calidad almacenada "
            "no puede auditarse manualmente desde este conjunto."
        )
    if rows and not has_prompt_text:
        warnings.append(
            "Los raws declarados no contienen prompt_text; los prompts deben "
            "reconstruirse desde la configuración de tareas."
        )

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "rows": len(rows),
        "runs": len(grouped),
        "contains_prompt_text": has_prompt_text,
        "contains_response_text": has_response_text,
        "ordinary_task_sequence": reference_sequence or [],
    }


def per_run_summary(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (quant, arm, rep), run_rows in sorted(grouped.items()):
        ordinary = [row for row in run_rows if not bool(row["is_compaction"])]
        compactions = [row for row in run_rows if bool(row["is_compaction"])]
        quality_values = [
            float(row["quality"]) for row in ordinary if row.get("quality") is not None
        ]
        output.append(
            {
                "quant": quant,
                "arm": arm,
                "rep": rep,
                "ordinary_tasks": len(ordinary),
                "compactions": len(compactions),
                "ordinary_energy_j": rounded(
                    sum(float(row["energy_j"]) for row in ordinary)
                ),
                "compaction_tax_j": rounded(
                    sum(float(row["energy_j"]) for row in compactions)
                ),
                "total_energy_j": rounded(
                    sum(float(row["energy_j"]) for row in run_rows)
                ),
                "mean_stored_quality": rounded(mean(quality_values)),
                "final_prompt_tokens": int(ordinary[-1]["accumulated_prompt_tokens"]),
            }
        )
    return output


def aggregate_summary(run_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in run_summary:
        groups[(str(row["quant"]), str(row["arm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (quant, arm), group in sorted(groups.items()):
        totals = [float(row["total_energy_j"]) for row in group]
        ordinary = [float(row["ordinary_energy_j"]) for row in group]
        taxes = [float(row["compaction_tax_j"]) for row in group]
        qualities = [float(row["mean_stored_quality"]) for row in group]
        output.append(
            {
                "quant": quant,
                "arm": arm,
                "n_reps": len(group),
                "total_energy_mean_j": rounded(mean(totals)),
                "total_energy_sd_j": rounded(sample_sd(totals)),
                "ordinary_energy_mean_j": rounded(mean(ordinary)),
                "compaction_tax_mean_j": rounded(mean(taxes)),
                "stored_quality_mean": rounded(mean(qualities)),
            }
        )
    return output


def compaction_accounting(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for quant in QUANTS:
        for rep in REPS:
            naive = grouped[(quant, "naive", rep)]
            compact = grouped[(quant, "compaction", rep)]
            naive_tasks = sum(
                float(row["energy_j"]) for row in naive if not bool(row["is_compaction"])
            )
            compact_tasks = sum(
                float(row["energy_j"])
                for row in compact
                if not bool(row["is_compaction"])
            )
            tax = sum(
                float(row["energy_j"]) for row in compact if bool(row["is_compaction"])
            )
            ordinary_savings = naive_tasks - compact_tasks
            output.append(
                {
                    "quant": quant,
                    "rep": rep,
                    "naive_ordinary_energy_j": rounded(naive_tasks),
                    "compact_ordinary_energy_j": rounded(compact_tasks),
                    "ordinary_call_savings_j": rounded(ordinary_savings),
                    "compaction_tax_j": rounded(tax),
                    "net_extra_compaction_j": rounded(tax - ordinary_savings),
                    "break_even_observed": (compact_tasks + tax) <= naive_tasks,
                }
            )
    return output


def cumulative_by_task(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for quant in QUANTS:
        for rep in REPS:
            naive = grouped[(quant, "naive", rep)]
            compact = grouped[(quant, "compaction", rep)]
            for task in (row for row in naive if not bool(row["is_compaction"])):
                index = float(task["task_index"])
                naive_cumulative = sum(
                    float(row["energy_j"])
                    for row in naive
                    if float(row["task_index"]) <= index
                )
                compact_cumulative = sum(
                    float(row["energy_j"])
                    for row in compact
                    if float(row["task_index"]) <= index
                )
                output.append(
                    {
                        "quant": quant,
                        "rep": rep,
                        "task_index": canonical_index(index),
                        "task_id": task["task_id"],
                        "naive_cumulative_j": rounded(naive_cumulative),
                        "compact_cumulative_j": rounded(compact_cumulative),
                        "compact_minus_naive_j": rounded(
                            compact_cumulative - naive_cumulative
                        ),
                    }
                )
    return output


def paired_task_effects(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    task_rows: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (quant, arm, _rep), run_rows in grouped.items():
        for row in run_rows:
            if not bool(row["is_compaction"]):
                task_rows[(quant, arm, str(row["task_id"]))].append(row)

    output: list[dict[str, Any]] = []
    reference = [
        row["task_id"]
        for row in grouped[(QUANTS[0], "naive", REPS[0])]
        if not bool(row["is_compaction"])
    ]
    order = {task_id: index for index, task_id in enumerate(reference)}

    for quant in QUANTS:
        for task_id in reference:
            naive = task_rows[(quant, "naive", task_id)]
            compact = task_rows[(quant, "compaction", task_id)]
            naive_energy = mean(float(row["energy_j"]) for row in naive)
            compact_energy = mean(float(row["energy_j"]) for row in compact)
            naive_quality = mean(float(row["quality"]) for row in naive)
            compact_quality = mean(float(row["quality"]) for row in compact)
            output.append(
                {
                    "quant": quant,
                    "task_index": order[task_id],
                    "task_id": task_id,
                    "naive_context_mean": rounded(
                        mean(float(row["accumulated_prompt_tokens"]) for row in naive)
                    ),
                    "compact_context_mean": rounded(
                        mean(float(row["accumulated_prompt_tokens"]) for row in compact)
                    ),
                    "naive_completion_mean": rounded(
                        mean(float(row["completion_tokens"]) for row in naive)
                    ),
                    "compact_completion_mean": rounded(
                        mean(float(row["completion_tokens"]) for row in compact)
                    ),
                    "energy_saved_mean_j": rounded(naive_energy - compact_energy),
                    "stored_quality_change": rounded(
                        compact_quality - naive_quality
                    ),
                }
            )
    return output


def trigger_events(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for (quant, arm, rep), run_rows in sorted(grouped.items()):
        if arm != "compaction":
            continue
        previous_task: dict[str, Any] | None = None
        for row in run_rows:
            if bool(row["is_compaction"]):
                output.append(
                    {
                        "quant": quant,
                        "rep": rep,
                        "event": row["task_id"],
                        "previous_task_id": (
                            previous_task["task_id"] if previous_task else None
                        ),
                        "previous_task_prompt_tokens": (
                            previous_task["accumulated_prompt_tokens"]
                            if previous_task
                            else None
                        ),
                        "compaction_prompt_tokens": row[
                            "accumulated_prompt_tokens"
                        ],
                        "handoff_completion_tokens": row["completion_tokens"],
                        "compaction_energy_j": row["energy_j"],
                    }
                )
            else:
                previous_task = row
    return output


def integrity_summary(files: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": row["name"],
            "sha256": row["sha256"],
            "bytes": row["bytes"],
            "rows": row["rows"],
            "contains_prompt_text": row["contains_prompt_text"],
            "contains_response_text": row["contains_response_text"],
        }
        for row in files
    ]


def handoff_summary(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in events:
        groups[(str(row["quant"]), str(row["event"]))].append(row)

    output: list[dict[str, Any]] = []
    for (quant, event), group in sorted(groups.items()):
        output.append(
            {
                "quant": quant,
                "event": event,
                "n_reps": len(group),
                "trigger_task": group[0]["previous_task_id"],
                "trigger_prompt_tokens": rounded(
                    mean(float(row["previous_task_prompt_tokens"]) for row in group)
                ),
                "handoff_prompt_tokens": rounded(
                    mean(float(row["compaction_prompt_tokens"]) for row in group)
                ),
                "handoff_completion_tokens": rounded(
                    mean(float(row["handoff_completion_tokens"]) for row in group)
                ),
                "handoff_energy_mean_j": rounded(
                    mean(float(row["compaction_energy_j"]) for row in group)
                ),
                "handoff_energy_sd_j": rounded(
                    sample_sd(float(row["compaction_energy_j"]) for row in group)
                ),
            }
        )
    return output


def token_summary(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    per_run: list[dict[str, Any]] = []
    for (quant, arm, rep), run_rows in sorted(grouped.items()):
        ordinary = [row for row in run_rows if not bool(row["is_compaction"])]
        handoffs = [row for row in run_rows if bool(row["is_compaction"])]
        per_run.append(
            {
                "quant": quant,
                "arm": arm,
                "rep": rep,
                "ordinary_prompt_tokens": sum(
                    int(row["accumulated_prompt_tokens"]) for row in ordinary
                ),
                "ordinary_completion_tokens": sum(
                    int(row["completion_tokens"]) for row in ordinary
                ),
                "handoff_prompt_tokens": sum(
                    int(row["accumulated_prompt_tokens"]) for row in handoffs
                ),
                "handoff_completion_tokens": sum(
                    int(row["completion_tokens"]) for row in handoffs
                ),
            }
        )

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_run:
        groups[(str(row["quant"]), str(row["arm"]))].append(row)

    output: list[dict[str, Any]] = []
    for (quant, arm), group in sorted(groups.items()):
        output.append(
            {
                "quant": quant,
                "arm": arm,
                "n_reps": len(group),
                "ordinary_prompt_tokens_mean": rounded(
                    mean(float(row["ordinary_prompt_tokens"]) for row in group), 1
                ),
                "ordinary_completion_tokens_mean": rounded(
                    mean(float(row["ordinary_completion_tokens"]) for row in group), 1
                ),
                "handoff_prompt_tokens_mean": rounded(
                    mean(float(row["handoff_prompt_tokens"]) for row in group), 1
                ),
                "handoff_completion_tokens_mean": rounded(
                    mean(float(row["handoff_completion_tokens"]) for row in group), 1
                ),
                "all_prompt_tokens_mean": rounded(
                    mean(
                        float(row["ordinary_prompt_tokens"])
                        + float(row["handoff_prompt_tokens"])
                        for row in group
                    ),
                    1,
                ),
                "all_completion_tokens_mean": rounded(
                    mean(
                        float(row["ordinary_completion_tokens"])
                        + float(row["handoff_completion_tokens"])
                        for row in group
                    ),
                    1,
                ),
            }
        )
    return output


def quality_programmatic_summary(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for quant in QUANTS:
        for arm in ARMS:
            values: list[float] = []
            for rep in REPS:
                ordinary = [
                    row
                    for row in grouped[(quant, arm, rep)]
                    if not bool(row["is_compaction"])
                    and row.get("quality") is not None
                ]
                values.append(mean(float(row["quality"]) for row in ordinary))
            output.append(
                {
                    "quant": quant,
                    "arm": arm,
                    "n_reps": len(values),
                    "stored_quality_mean": rounded(mean(values)),
                    "stored_quality_sd": rounded(sample_sd(values)),
                    "auditability": (
                        "Descriptivo historico no reauditable: los raws "
                        "principales no conservan respuestas ni handoffs."
                    ),
                }
            )
    return output


def cycle_accounting(
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """Asigna a cada handoff su intervalo de reutilización y deuda residual.

    El ciclo k comienza con el handoff k y termina en la tarea que dispara el
    siguiente handoff; el último termina en T16. La deuda es la diferencia
    acumulada compactación - historial completo.
    """
    output: list[dict[str, Any]] = []
    for quant in QUANTS:
        for rep in REPS:
            naive = grouped[(quant, "naive", rep)]
            compact = grouped[(quant, "compaction", rep)]
            events = [row for row in compact if bool(row["is_compaction"])]
            ordinary_naive = {
                int(float(row["task_index"])): row
                for row in naive
                if not bool(row["is_compaction"])
            }
            ordinary_compact = {
                int(float(row["task_index"])): row
                for row in compact
                if not bool(row["is_compaction"])
            }

            def cumulative(rows: list[dict[str, Any]], limit: float) -> float:
                return sum(
                    float(row["energy_j"])
                    for row in rows
                    if float(row["task_index"]) <= limit
                )

            for position, event in enumerate(events):
                trigger_index = int(math.floor(float(event["task_index"])))
                next_trigger_index = (
                    int(math.floor(float(events[position + 1]["task_index"])))
                    if position + 1 < len(events)
                    else max(ordinary_naive)
                )
                recovery_indices = list(range(trigger_index + 1, next_trigger_index + 1))
                ordinary_savings = sum(
                    float(ordinary_naive[index]["energy_j"])
                    - float(ordinary_compact[index]["energy_j"])
                    for index in recovery_indices
                )
                debt_after_handoff = (
                    cumulative(compact, float(event["task_index"]))
                    - cumulative(naive, float(event["task_index"]))
                )
                debt_end_cycle = (
                    cumulative(compact, float(next_trigger_index))
                    - cumulative(naive, float(next_trigger_index))
                )
                output.append(
                    {
                        "quant": quant,
                        "rep": rep,
                        "cycle": position + 1,
                        "handoff_event": event["task_id"],
                        "trigger_task": ordinary_naive[trigger_index]["task_id"],
                        "handoff_energy_j": rounded(float(event["energy_j"])),
                        "recovery_start_task": (
                            ordinary_naive[recovery_indices[0]]["task_id"]
                            if recovery_indices
                            else None
                        ),
                        "recovery_end_task": ordinary_naive[next_trigger_index]["task_id"],
                        "ordinary_savings_in_cycle_j": rounded(ordinary_savings),
                        "residual_debt_after_handoff_j": rounded(debt_after_handoff),
                        "residual_debt_end_cycle_j": rounded(debt_end_cycle),
                        "amortized_by_cycle_end": debt_end_cycle <= 0,
                    }
                )
    return output


def cumulative_curve_mean(
    cumulative_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in cumulative_rows:
        groups[(str(row["quant"]), int(float(row["task_index"])))].append(row)

    output: list[dict[str, Any]] = []
    for quant in QUANTS:
        output.append(
            {
                "quant": quant,
                "step": 0,
                "task_index": -1,
                "task_id": "INICIO",
                "compact_minus_naive_mean_j": 0.0,
                "compact_minus_naive_sd_j": 0.0,
            }
        )
        for index in range(EXPECTED_TASKS):
            group = groups[(quant, index)]
            values = [float(row["compact_minus_naive_j"]) for row in group]
            output.append(
                {
                    "quant": quant,
                    "step": index + 1,
                    "task_index": index,
                    "task_id": group[0]["task_id"],
                    "compact_minus_naive_mean_j": rounded(mean(values)),
                    "compact_minus_naive_sd_j": rounded(sample_sd(values)),
                }
            )
    return output


def _font(size: int, bold: bool = False):
    if ImageFont is None:
        raise RuntimeError("Pillow no está disponible")
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _draw_vertical_label(
    image, text: str, center_x: float, center_y: float, font, fill: str
) -> None:
    probe = ImageDraw.Draw(image)
    box = probe.textbbox((0, 0), text, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    layer = Image.new(
        "RGBA", (text_width + 24, text_height + 24), (255, 255, 255, 0)
    )
    ImageDraw.Draw(layer).text((12, 12), text, font=font, fill=fill)
    rotated = layer.rotate(90, expand=True)
    image.paste(
        rotated,
        (
            int(center_x - rotated.width / 2),
            int(center_y - rotated.height / 2),
        ),
        rotated,
    )


def _save_energy_delta_plot(
    path: Path, curve: list[dict[str, Any]]
) -> None:
    width, height = 1800, 1000
    left, right, top, bottom = 170, 80, 100, 150
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(42, True), _font(30), _font(24)

    series = {
        quant: [row for row in curve if row["quant"] == quant]
        for quant in QUANTS
    }
    all_values = [
        float(row["compact_minus_naive_mean_j"]) for row in curve
    ]
    y_min, y_max = min(0.0, min(all_values)), max(all_values)
    y_max = math.ceil(y_max / 1000.0) * 1000.0
    y_min = math.floor(y_min / 1000.0) * 1000.0
    if math.isclose(y_max, y_min):
        y_max = y_min + 1.0

    def x_pos(step: float) -> float:
        return left + (step / max(float(EXPECTED_TASKS), 1.0)) * plot_w

    def y_pos(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    draw.text(
        (width / 2, 35),
        "Deuda energética acumulada de la compactación",
        font=title_font,
        fill="#172033",
        anchor="ma",
    )
    for tick in range(int(y_min), int(y_max) + 1, 1000):
        y = y_pos(float(tick))
        draw.line((left, y, width - right, y), fill="#D9DEE8", width=2)
        draw.text(
            (left - 20, y),
            f"{tick:,}".replace(",", "."),
            font=tick_font,
            fill="#4B5563",
            anchor="rm",
        )
    zero_y = y_pos(0.0)
    draw.line((left, zero_y, width - right, zero_y), fill="#111827", width=4)
    draw.line((left, top, left, height - bottom), fill="#111827", width=3)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="#111827",
        width=3,
    )

    tick_step = max(1, math.ceil(EXPECTED_TASKS / 6))
    for step in range(0, EXPECTED_TASKS + 1, tick_step):
        x = x_pos(step)
        draw.line((x, height - bottom, x, height - bottom + 12), fill="#111827", width=2)
        draw.text(
            (x, height - bottom + 25),
            str(step),
            font=tick_font,
            fill="#4B5563",
            anchor="ma",
        )

    palette = ("#C56A1A", "#2463A6", "#2F855A", "#805AD5")
    colors = {
        quant: palette[index % len(palette)]
        for index, quant in enumerate(QUANTS)
    }
    for quant, rows in series.items():
        points = [
            (
                x_pos(float(row["step"])),
                y_pos(float(row["compact_minus_naive_mean_j"])),
            )
            for row in rows
        ]
        draw.line(points, fill=colors[quant], width=7, joint="curve")
        for point in points:
            draw.ellipse(
                (point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5),
                fill=colors[quant],
            )
        final = rows[-1]
        draw.text(
            (points[-1][0] - 10, points[-1][1] - 12),
            f"{quant}: +{float(final['compact_minus_naive_mean_j']):,.0f} J".replace(",", "."),
            font=label_font,
            fill=colors[quant],
            anchor="rs",
        )

    draw.text(
        (width / 2, height - 55),
        "Tareas ordinarias completadas (0 = antes de la intervención)",
        font=label_font,
        fill="#172033",
        anchor="ms",
    )
    _draw_vertical_label(
        image,
        "Compactación − historial completo (J)",
        45,
        height / 2,
        label_font,
        "#172033",
    )
    draw.text(
        (width - right, top + 5),
        f"Media de {len(REPS)} réplicas; punto de equilibrio si ΔE ≤ 0",
        font=tick_font,
        fill="#4B5563",
        anchor="ra",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(200, 200))


def _save_energy_accounting_plot(
    path: Path, aggregate: list[dict[str, Any]]
) -> None:
    width, height = 1700, 1050
    left, right, top, bottom = 170, 80, 120, 190
    plot_w, plot_h = width - left - right, height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(42, True), _font(28), _font(23)
    data = {(row["quant"], row["arm"]): row for row in aggregate}
    labels = [
        (
            quant,
            arm,
            f"{quant}\n{'Completo' if arm == 'naive' else 'Compactación'}",
        )
        for quant in QUANTS
        for arm in ARMS
    ]
    largest_total = max(
        float(row["ordinary_energy_mean_j"])
        + float(row["compaction_tax_mean_j"])
        for row in aggregate
    )
    y_step = max(1000.0, math.ceil(largest_total / 5.0 / 1000.0) * 1000.0)
    y_max = y_step * 5.0

    def y_pos(value: float) -> float:
        return top + (y_max - value) / y_max * plot_h

    draw.text(
        (width / 2, 38),
        "Contabilidad energética por política",
        font=title_font,
        fill="#172033",
        anchor="ma",
    )
    for index in range(6):
        tick = y_step * index
        y = y_pos(tick)
        draw.line((left, y, width - right, y), fill="#D9DEE8", width=2)
        draw.text(
            (left - 20, y),
            f"{tick:,.0f}".replace(",", "."),
            font=tick_font,
            fill="#4B5563",
            anchor="rm",
        )
    draw.line((left, top, left, height - bottom), fill="#111827", width=3)
    draw.line(
        (left, height - bottom, width - right, height - bottom),
        fill="#111827",
        width=3,
    )

    gap = plot_w / len(labels)
    bar_w = 170
    for position, (quant, arm, label) in enumerate(labels):
        row = data[(quant, arm)]
        ordinary = float(row["ordinary_energy_mean_j"])
        tax = float(row["compaction_tax_mean_j"])
        x = left + gap * (position + 0.5)
        draw.rectangle(
            (x - bar_w / 2, y_pos(ordinary), x + bar_w / 2, y_pos(0)),
            fill="#356FA8",
        )
        if tax > 0:
            draw.rectangle(
                (
                    x - bar_w / 2,
                    y_pos(ordinary + tax),
                    x + bar_w / 2,
                    y_pos(ordinary),
                ),
                fill="#D6852D",
            )
        total = ordinary + tax
        draw.text(
            (x, y_pos(total) - 14),
            f"{total:,.0f} J".replace(",", "."),
            font=label_font,
            fill="#172033",
            anchor="ms",
        )
        label_lines = label.split("\n")
        draw.text(
            (x, height - bottom + 28),
            label_lines[0],
            font=label_font,
            fill="#172033",
            anchor="ma",
        )
        draw.text(
            (x, height - bottom + 64),
            label_lines[1],
            font=tick_font,
            fill="#4B5563",
            anchor="ma",
        )

    legend_y = height - 50
    draw.rectangle((520, legend_y - 22, 555, legend_y + 13), fill="#356FA8")
    draw.text((570, legend_y), "Llamadas ordinarias", font=tick_font, fill="#172033", anchor="lm")
    draw.rectangle((910, legend_y - 22, 945, legend_y + 13), fill="#D6852D")
    draw.text((960, legend_y), "Resúmenes", font=tick_font, fill="#172033", anchor="lm")
    _draw_vertical_label(
        image,
        "Energía media por sesión (J)",
        45,
        height / 2,
        label_font,
        "#172033",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(200, 200))


def _save_token_plot(path: Path, tokens: list[dict[str, Any]]) -> None:
    width, height = 1800, 1050
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font, label_font, tick_font = _font(42, True), _font(26), _font(22)
    draw.text(
        (width / 2, 38),
        "Tokens procesados por sesión",
        font=title_font,
        fill="#172033",
        anchor="ma",
    )
    data = {(row["quant"], row["arm"]): row for row in tokens}
    labels = [
        (
            quant,
            arm,
            f"{quant} {'completo' if arm == 'naive' else 'compactación'}",
        )
        for quant in QUANTS
        for arm in ARMS
    ]
    panels = [
        (
            110,
            160,
            850,
            760,
            "Tokens de prompt",
            "ordinary_prompt_tokens_mean",
            "handoff_prompt_tokens_mean",
        ),
        (
            950,
            160,
            1690,
            760,
            "Tokens de salida",
            "ordinary_completion_tokens_mean",
            "handoff_completion_tokens_mean",
        ),
    ]
    for x0, y0, x1, y1, heading, ordinary_key, handoff_key in panels:
        draw.text(
            ((x0 + x1) / 2, y0 - 40),
            heading,
            font=label_font,
            fill="#172033",
            anchor="ma",
        )
        max_value = max(
            float(data[(q, a)][ordinary_key]) + float(data[(q, a)][handoff_key])
            for q, a, _ in labels
        )
        max_value *= 1.12
        draw.line((x0, y0, x0, y1), fill="#111827", width=3)
        draw.line((x0, y1, x1, y1), fill="#111827", width=3)
        gap = (x1 - x0) / len(labels)
        bar_w = 110
        for position, (quant, arm, label) in enumerate(labels):
            row = data[(quant, arm)]
            ordinary = float(row[ordinary_key])
            handoff = float(row[handoff_key])
            x = x0 + gap * (position + 0.5)

            def py(value: float) -> float:
                return y1 - value / max_value * (y1 - y0)

            draw.rectangle(
                (x - bar_w / 2, py(ordinary), x + bar_w / 2, y1),
                fill="#356FA8",
            )
            if handoff:
                draw.rectangle(
                    (
                        x - bar_w / 2,
                        py(ordinary + handoff),
                        x + bar_w / 2,
                        py(ordinary),
                    ),
                    fill="#D6852D",
                )
            total = ordinary + handoff
            draw.text(
                (x, py(total) - 10),
                f"{total:,.0f}".replace(",", "."),
                font=tick_font,
                fill="#172033",
                anchor="ms",
            )
            first, second = label.split()
            draw.text(
                (x, y1 + 24),
                first,
                font=tick_font,
                fill="#172033",
                anchor="ma",
            )
            draw.text(
                (x, y1 + 54),
                second,
                font=tick_font,
                fill="#4B5563",
                anchor="ma",
            )

    draw.rectangle((520, 900, 555, 935), fill="#356FA8")
    draw.text((570, 918), "Tareas ordinarias", font=tick_font, fill="#172033", anchor="lm")
    draw.rectangle((910, 900, 945, 935), fill="#D6852D")
    draw.text((960, 918), "Resúmenes", font=tick_font, fill="#172033", anchor="lm")
    draw.text(
        (width / 2, 995),
        f"Media de {len(REPS)} sesiones por condición",
        font=tick_font,
        fill="#4B5563",
        anchor="ms",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(200, 200))


def write_authoritative_report(
    path: Path,
    validation: dict[str, Any],
    aggregate: list[dict[str, Any]],
    accounting: list[dict[str, Any]],
) -> None:
    lookup = {(row["quant"], row["arm"]): row for row in aggregate}
    mean_accounting: dict[str, dict[str, float]] = {}
    for quant in QUANTS:
        group = [row for row in accounting if row["quant"] == quant]
        mean_accounting[quant] = {
            "ordinary_savings": mean(
                float(row["ordinary_call_savings_j"]) for row in group
            ),
            "tax": mean(float(row["compaction_tax_j"]) for row in group),
            "net": mean(float(row["net_extra_compaction_j"]) for row in group),
        }
    session_label = (
        "conjunto de referencia" if SESSION_TAG == "v3" else SESSION_TAG
    )
    lines = [
        f"# Resultados: {session_label}",
        "",
        (
            "Este informe fue generado exclusivamente desde los "
            f"{len(QUANTS) * len(ARMS) * len(REPS)} JSONL declarados."
        ),
        "No incluye archivos fuera del patrón y las condiciones declaradas.",
        "",
        "## Integridad",
        "",
        f"- Validación: {'correcta' if validation['ok'] else 'fallida'}.",
        f"- Sesiones: {validation['runs']}.",
        f"- Filas: {validation['rows']}.",
        f"- Tareas ordinarias por sesión: {EXPECTED_TASKS}.",
        (
            f"- Resúmenes esperados por sesión compactada: {EXPECTED_COMPACTIONS}."
            if EXPECTED_COMPACTIONS >= 0
            else "- Resúmenes por sesión compactada: validados sin conteo predeclarado."
        ),
        "",
        "## Totales medios por sesión",
        "",
    ]
    for quant in QUANTS:
        naive = float(lookup[(quant, "naive")]["total_energy_mean_j"])
        compact = float(lookup[(quant, "compaction")]["total_energy_mean_j"])
        lines.append(
            f"- {quant}: historial completo {naive:.2f} J; compactación "
            f"{compact:.2f} J; diferencia {compact - naive:+.2f} J."
        )
    lines.extend(["", "## Contabilidad de la compactación", ""])
    for quant in QUANTS:
        row = mean_accounting[quant]
        if row["ordinary_savings"] >= 0:
            ordinary_text = (
                f"ahorró {row['ordinary_savings']:.2f} J en llamadas ordinarias"
            )
        else:
            ordinary_text = (
                f"consumió {-row['ordinary_savings']:.2f} J adicionales en "
                "llamadas ordinarias"
            )
        lines.append(
            f"- {quant}: {ordinary_text}; resúmenes {row['tax']:.2f} J; "
            f"sobrecosto neto {row['net']:.2f} J."
        )
    break_even_count = sum(bool(row["break_even_observed"]) for row in accounting)
    if break_even_count:
        lines.append(
            f"- {break_even_count} de {len(accounting)} parejas alcanzaron el "
            "punto de equilibrio al cierre."
        )
    else:
        lines.append("- Ninguna pareja alcanzó el punto de equilibrio observado.")
    lines.extend(["", "## Salvedad de calidad", ""])
    if validation["contains_response_text"]:
        lines.extend(
            [
                "Los JSONL conservan las respuestas y permiten reauditar las reglas",
                "programáticas. Esto no equivale a una evaluación humana de calidad",
                "ni demuestra fidelidad semántica completa.",
            ]
        )
    else:
        lines.extend(
            [
                "Los puntajes programáticos se preservan como datos históricos. No",
                "pueden reauditarse desde estos raws porque faltan las respuestas y",
                "los contenidos de los resúmenes; no sustentan calidad humana ni",
                "fidelidad semántica.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretación",
            "",
            "El resultado describe únicamente la configuración, la política y el",
            "horizonte observados. No establece un umbral óptimo ni una ley universal.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["quant"]),
            str(item["arm"]),
            int(item["rep"]),
            float(item["task_index"]),
        ),
    ):
        clean = {
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
            if not key.startswith("_")
        }
        clean["source_file"] = row["_source_file"]
        clean["source_line"] = row["_source_line"]
        clean["source_sha256"] = row["_source_sha256"]
        output.append(clean)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analiza una comparación historial completo/compactación "
            "sin modificar los JSONL."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/reference/raw"),
        help=(
            "Directorio que contiene los JSONL. Por omisión se usa el "
            "conjunto de referencia publicado."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results/reference/expected"),
        help="Directorio para tablas, figuras, informe y manifiesto.",
    )
    parser.add_argument(
        "--session-tag",
        default="v3",
        help="Etiqueta usada en run_<etiqueta>_<quant>_<arm>_repN.jsonl.",
    )
    parser.add_argument(
        "--quants",
        default="AWQ,FP16",
        help="Etiquetas de representación separadas por comas.",
    )
    parser.add_argument(
        "--reps",
        default="1,2,3",
        help="Réplicas esperadas separadas por comas.",
    )
    parser.add_argument(
        "--expected-tasks",
        type=int,
        default=29,
        help="Cantidad esperada de tareas ordinarias por sesión.",
    )
    parser.add_argument(
        "--expected-compactions",
        type=int,
        default=3,
        help=(
            "Cantidad esperada de resúmenes por sesión compactada; "
            "usar -1 para no fijarla."
        ),
    )
    return parser.parse_args()


def main() -> int:
    global QUANTS, REPS, SESSION_TAG, EXPECTED_TASKS, EXPECTED_COMPACTIONS
    args = parse_args()
    QUANTS = tuple(value.strip() for value in args.quants.split(",") if value.strip())
    REPS = tuple(
        int(value.strip()) for value in args.reps.split(",") if value.strip()
    )
    SESSION_TAG = args.session_tag.strip()
    EXPECTED_TASKS = args.expected_tasks
    EXPECTED_COMPACTIONS = args.expected_compactions
    if not QUANTS or not REPS or not SESSION_TAG:
        raise SystemExit("quants, reps y session-tag no pueden quedar vacíos")
    if EXPECTED_TASKS <= 0 or EXPECTED_COMPACTIONS < -1:
        raise SystemExit(
            "expected-tasks debe ser positivo y expected-compactions >= -1"
        )
    rows, files, read_errors = read_runs(args.source)
    grouped = group_runs(rows)
    validation = validate_rows(rows, grouped)
    validation["errors"] = read_errors + validation["errors"]
    validation["ok"] = not validation["errors"]

    run_summary = per_run_summary(grouped) if validation["ok"] else []
    aggregate = aggregate_summary(run_summary) if run_summary else []
    accounting = compaction_accounting(grouped) if validation["ok"] else []
    cumulative = cumulative_by_task(grouped) if validation["ok"] else []
    paired = paired_task_effects(grouped) if validation["ok"] else []
    triggers = trigger_events(grouped) if validation["ok"] else []
    integrity = integrity_summary(files)
    handoffs = handoff_summary(triggers) if triggers else []
    tokens = token_summary(grouped) if validation["ok"] else []
    quality_summary = (
        quality_programmatic_summary(grouped) if validation["ok"] else []
    )
    cycles = cycle_accounting(grouped) if validation["ok"] else []
    curve_mean = cumulative_curve_mean(cumulative) if cumulative else []

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "integrity_provenance.csv", integrity)
    write_csv(args.out / "normalized_rows.csv", normalized_rows(rows))
    write_csv(args.out / "per_run_summary.csv", run_summary)
    write_csv(args.out / "aggregate_summary.csv", aggregate)
    write_csv(args.out / "compaction_accounting.csv", accounting)
    write_csv(args.out / "cycle_accounting.csv", cycles)
    write_csv(args.out / "cumulative_by_task.csv", cumulative)
    write_csv(args.out / "cumulative_curve_mean.csv", curve_mean)
    write_csv(args.out / "paired_task_effects.csv", paired)
    write_csv(args.out / "compaction_events.csv", triggers)
    write_csv(args.out / "handoff_summary.csv", handoffs)
    write_csv(args.out / "token_summary.csv", tokens)
    write_csv(args.out / "quality_programmatic_summary.csv", quality_summary)

    if validation["ok"]:
        if Image is None:
            validation["warnings"].append(
                "Pillow no está instalado; no se generaron figuras PNG."
            )
        else:
            _save_energy_delta_plot(
                args.out / "figura_delta_energia_acumulada.png", curve_mean
            )
            _save_energy_accounting_plot(
                args.out / "figura_contabilidad_energia.png", aggregate
            )
            _save_token_plot(args.out / "figura_tokens_por_politica.png", tokens)
        write_authoritative_report(
            args.out / "analysis_summary.md",
            validation,
            aggregate,
            accounting,
        )

    manifest = {
        "schema_version": 3,
        "source_directory": str(args.source),
        "declared_scope": {
            "session": SESSION_TAG,
            "quants": list(QUANTS),
            "arms": list(ARMS),
            "reps": list(REPS),
            "expected_tasks_per_session": EXPECTED_TASKS,
            "expected_compactions_per_compact_session": EXPECTED_COMPACTIONS,
        },
        "source_files": files,
        "validation": validation,
        "derived_files": [
            "integrity_provenance.csv",
            "normalized_rows.csv",
            "per_run_summary.csv",
            "aggregate_summary.csv",
            "compaction_accounting.csv",
            "cycle_accounting.csv",
            "cumulative_by_task.csv",
            "cumulative_curve_mean.csv",
            "paired_task_effects.csv",
            "compaction_events.csv",
            "handoff_summary.csv",
            "token_summary.csv",
            "quality_programmatic_summary.csv",
            "figura_delta_energia_acumulada.png",
            "figura_contabilidad_energia.png",
            "figura_tokens_por_politica.png",
            "analysis_summary.md",
        ],
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "ok": validation["ok"],
                "rows": validation["rows"],
                "runs": validation["runs"],
                "warnings": validation["warnings"],
                "out": str(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
