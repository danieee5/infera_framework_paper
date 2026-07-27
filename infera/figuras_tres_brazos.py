#!/usr/bin/env python3
"""Genera figuras reproducibles de la campaña de tres brazos, sin GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics as st
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any


ARMS = ("descarte", "resumen", "completo")
QUANTS = ("AWQ", "FP16")
REPS = (1, 2, 3)
EVENT_TYPES = {"COMPACTION", "DESCARTE"}
ARM_LABELS = {
    "descarte": "Descarte",
    "resumen": "Resumen",
    "completo": "Completo",
}
COLORS = {
    "descarte": "#2A9D8F",
    "resumen": "#E76F51",
    "completo": "#457B9D",
    "mechanism": "#F4A261",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: se esperaba un objeto JSON")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = path.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise ValueError(f"{path}: vacío o posiblemente truncado")
    return [json.loads(line) for line in raw.decode("utf-8").splitlines()]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_inputs(campaign: Path) -> None:
    campaign_manifest = load_json(campaign / "manifiesto_campana.json")
    declared_analysis = campaign_manifest.get("analysis_artifacts")
    if not isinstance(declared_analysis, list) or len(declared_analysis) != 6:
        raise ValueError("la campaña no declara 6 artefactos de análisis")
    for entry in declared_analysis:
        if not isinstance(entry, dict):
            raise ValueError("entrada inválida en artefactos de análisis")
        name = entry.get("name")
        digest = entry.get("sha256")
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not isinstance(digest, str)
        ):
            raise ValueError("artefacto de análisis inseguro/incompleto")
        path = campaign / "analisis" / name
        if (
            not path.is_file()
            or sha256_file(path) != digest
            or path.stat().st_size != int(entry.get("bytes", -1))
        ):
            raise ValueError(f"artefacto de análisis ausente o alterado: {name}")
    analysis_manifest = load_json(campaign / "analisis/manifiesto_analisis.json")
    expected = analysis_manifest.get("raw_sha256")
    if not isinstance(expected, dict) or len(expected) != 18:
        raise ValueError("el manifiesto de análisis no identifica 18 raws")
    for name, digest in expected.items():
        path = campaign / "raw" / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"raw ausente o alterado: {name}")
    aggregate = load_csv(campaign / "analisis/agregado.csv")
    sessions = load_csv(campaign / "analisis/por_sesion.csv")
    if len(aggregate) != 6 or len(sessions) != 18:
        raise ValueError("los CSV de análisis no tienen 6/18 filas")


def load_sessions(
    campaign: Path,
) -> dict[tuple[str, str, int], list[dict[str, Any]]]:
    output = {}
    for quant in QUANTS:
        for arm in ARMS:
            for rep in REPS:
                path = campaign / "raw" / f"run_{quant}_{arm}_rep{rep}.jsonl"
                output[quant, arm, rep] = load_jsonl(path)
    return output


def save_figure(fig: Any, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")


def figure_energy(
    plt: Any,
    campaign: Path,
    directory: Path,
) -> None:
    aggregate = {
        (row["quant"], row["brazo"]): row
        for row in load_csv(campaign / "analisis/agregado.csv")
    }
    session_rows = load_csv(campaign / "analisis/por_sesion.csv")
    by_condition: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in session_rows:
        by_condition[row["quant"], row["brazo"]].append(
            float(row["energia_total_j"])
        )

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, quant in zip(axes, QUANTS):
        x = list(range(len(ARMS)))
        task = [
            float(aggregate[quant, arm]["energia_tareas_media_j"])
            for arm in ARMS
        ]
        mechanism = [
            float(aggregate[quant, arm]["energia_mecanismo_media_j"])
            for arm in ARMS
        ]
        axis.bar(
            x,
            task,
            color=COLORS["completo"],
            label="Energía de tareas",
        )
        axis.bar(
            x,
            mechanism,
            bottom=task,
            color=COLORS["mechanism"],
            label="Mecanismo de resumen",
        )
        for index, arm in enumerate(ARMS):
            jitter = (-0.07, 0.0, 0.07)
            for offset, value in zip(jitter, by_condition[quant, arm]):
                axis.scatter(
                    index + offset,
                    value,
                    s=22,
                    facecolors="white",
                    edgecolors="#202020",
                    linewidths=0.8,
                    zorder=5,
                )
        axis.set_title(quant)
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS])
        axis.set_ylabel("Energía por sesión (J)")
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=2,
        frameon=False,
    )
    fig.suptitle("Energía end-to-end por política", y=0.99)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, directory, "figura_1_energia_por_politica")
    plt.close(fig)


def task_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["task_type"] not in EVENT_TYPES]


def figure_cumulative(
    plt: Any,
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
    directory: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, quant in zip(axes, QUANTS):
        for arm in ARMS:
            repetitions = [
                task_rows(sessions[quant, arm, rep])
                for rep in REPS
            ]
            x = [int(row["task_index"]) + 1 for row in repetitions[0]]
            curves = [
                [float(row["cumulative_energy_j"]) for row in rows]
                for rows in repetitions
            ]
            mean = [st.mean(values) for values in zip(*curves)]
            low = [min(values) for values in zip(*curves)]
            high = [max(values) for values in zip(*curves)]
            axis.plot(
                x,
                mean,
                color=COLORS[arm],
                linewidth=2.0,
                label=ARM_LABELS[arm],
            )
            axis.fill_between(x, low, high, color=COLORS[arm], alpha=0.12)
        axis.set_title(quant)
        axis.set_xlabel("Tarea en la trayectoria")
        axis.set_ylabel("Energía acumulada (J)")
        axis.set_xlim(1, 29)
        axis.grid(alpha=0.22)
    axes[1].legend(frameon=False)
    fig.suptitle("Acumulación de energía durante la sesión")
    fig.tight_layout()
    save_figure(fig, directory, "figura_2_energia_acumulada")
    plt.close(fig)


def figure_occupancy(
    plt: Any,
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
    directory: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, quant in zip(axes, QUANTS):
        for arm in ARMS:
            repetitions = [
                task_rows(sessions[quant, arm, rep])
                for rep in REPS
            ]
            x = [int(row["task_index"]) + 1 for row in repetitions[0]]
            curves = [
                [int(row["accumulated_prompt_tokens"]) for row in rows]
                for rows in repetitions
            ]
            mean = [st.mean(values) for values in zip(*curves)]
            axis.plot(
                x,
                mean,
                color=COLORS[arm],
                linewidth=2.0,
                label=ARM_LABELS[arm],
            )
            for event in (
                row
                for row in sessions[quant, arm, 1]
                if row["task_type"] in EVENT_TYPES
            ):
                trigger = event.get(
                    "trigger_prompt_tokens",
                    event.get("context_tokens_before_intervention"),
                )
                axis.scatter(
                    float(event["task_index"]) + 1.0,
                    int(trigger),
                    s=28,
                    marker="D",
                    color=COLORS[arm],
                    zorder=5,
                )
        axis.axhline(
            4500,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label="Umbral 4.500",
        )
        axis.set_title(quant)
        axis.set_xlabel("Tarea en la trayectoria")
        axis.set_ylabel("Tokens del prompt completo")
        axis.set_xlim(1, 29)
        axis.grid(alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[1].legend(unique.values(), unique.keys(), frameon=False)
    fig.suptitle("Ocupación realizada del contexto")
    fig.tight_layout()
    save_figure(fig, directory, "figura_3_ocupacion_contexto")
    plt.close(fig)


def figure_programmatic_score(
    plt: Any,
    sessions: dict[tuple[str, str, int], list[dict[str, Any]]],
    directory: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True)
    for axis, quant in zip(axes, QUANTS):
        programmatic = []
        nontruncated = []
        for arm in ARMS:
            rows = task_rows(sessions[quant, arm, 1])
            programmatic.append(sum(float(row["quality"]) >= 1.0 for row in rows))
            nontruncated.append(
                sum(
                    float(row["quality"]) >= 1.0
                    and row.get("finish_reason") != "length"
                    for row in rows
                )
            )
        x = list(range(len(ARMS)))
        axis.bar(
            [value - 0.18 for value in x],
            programmatic,
            width=0.36,
            color=[COLORS[arm] for arm in ARMS],
            label="Score programático = 1",
        )
        axis.bar(
            [value + 0.18 for value in x],
            nontruncated,
            width=0.36,
            facecolor="white",
            edgecolor=[COLORS[arm] for arm in ARMS],
            linewidth=1.6,
            hatch="//",
            label="Score = 1 y no truncada",
        )
        axis.set_title(quant)
        axis.set_xticks(x, [ARM_LABELS[arm] for arm in ARMS])
        axis.set_ylabel("Tareas de 29")
        axis.set_ylim(0, 29)
        axis.grid(axis="y", alpha=0.22)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncols=2,
        frameon=False,
    )
    fig.suptitle("Resultado de la verificación predeclarada")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    save_figure(fig, directory, "figura_4_score_programatico")
    plt.close(fig)


def write_manifest(directory: Path, campaign: Path) -> None:
    artifacts = []
    for path in sorted(directory.iterdir()):
        if path.name == "manifiesto_figuras.json":
            continue
        artifacts.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    value = {
        "schema_version": 1,
        "campaign": str(campaign),
        "source_analysis_manifest_sha256": sha256_file(
            campaign / "analisis/manifiesto_analisis.json"
        ),
        "source_raw_sha256": load_json(
            campaign / "analisis/manifiesto_analisis.json"
        )["raw_sha256"],
        "figures": artifacts,
        "notes": [
            "Las bandas muestran rango de tres repeticiones instrumentales.",
            "Score programático no equivale a evaluación humana integral.",
            "No se estiman valores p ni inferencia poblacional.",
        ],
    }
    (directory / "manifiesto_figuras.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate(campaign: Path, output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib no está instalado; usa el entorno de requirements-gpu.txt"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 120,
        }
    )
    validate_inputs(campaign)
    sessions = load_sessions(campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.{uuid.uuid4().hex}.partial"
    temporary.mkdir()
    try:
        figure_energy(plt, campaign, temporary)
        figure_cumulative(plt, sessions, temporary)
        figure_occupancy(plt, sessions, temporary)
        figure_programmatic_score(plt, sessions, temporary)
        write_manifest(temporary, campaign)
        os.rename(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera cuatro figuras auditables sin ejecutar inferencia."
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.campaign.is_dir():
        raise SystemExit(f"no existe la campaña: {args.campaign}")
    if args.output.exists():
        raise SystemExit(f"no se sobrescribirá la salida: {args.output}")
    try:
        generate(args.campaign.resolve(), args.output.resolve())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FIGURAS_NO_GENERADAS: {exc}") from exc
    print(f"Figuras completas: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
