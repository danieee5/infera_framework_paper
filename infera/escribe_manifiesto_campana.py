#!/usr/bin/env python3
"""Mantiene el manifiesto transaccional de la campaña de tres brazos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gpu_power_monitor import GPUPowerMonitor
from infera_session_runner import atomic_write_json, sha256_file, utc_now


SCHEDULE = (
    ("AWQ", 1, ("completo", "resumen", "descarte")),
    ("FP16", 1, ("descarte", "resumen", "completo")),
    ("FP16", 2, ("completo", "descarte", "resumen")),
    ("AWQ", 2, ("resumen", "descarte", "completo")),
    ("AWQ", 3, ("descarte", "completo", "resumen")),
    ("FP16", 3, ("resumen", "completo", "descarte")),
)


def flattened_schedule() -> list[dict[str, object]]:
    return [
        {"position": position, "quant": quant, "rep": rep, "arm": arm}
        for position, (quant, rep, arm) in enumerate(
            (
                (quant, rep, arm)
                for quant, rep, arms in SCHEDULE
                for arm in arms
            ),
            start=1,
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument(
        "--status",
        choices=("preflight_failed", "running", "complete", "failed"),
        required=True,
    )
    parser.add_argument("--exit-code", type=int, default=0)
    return parser.parse_args()


def scan_artifacts(raw_dir: Path) -> dict[str, object]:
    raws = sorted(raw_dir.glob("run_*.jsonl"))
    partials = sorted(raw_dir.glob("*.partial"))
    manifests = sorted(raw_dir.glob("*.jsonl.manifiesto.json"))
    return {
        "raws": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in raws
        ],
        "session_manifests": [
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "status": json.loads(path.read_text(encoding="utf-8")).get(
                    "status"
                ),
            }
            for path in manifests
        ],
        "partials": [path.name for path in partials],
    }


def scan_tree(root: Path, excluded_names: set[str] | None = None) -> list[dict[str, object]]:
    excluded_names = excluded_names or set()
    if not root.is_dir():
        return []
    return [
        {
            "name": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in excluded_names
    ]


def gpu_metadata() -> dict[str, object]:
    monitor = GPUPowerMonitor(device_index=0)
    try:
        if not monitor.available:
            raise SystemExit("NVML no está disponible para la GPU 0")
        metadata = monitor.device_metadata()
        if not metadata.get("uuid"):
            raise SystemExit("NVML no devolvió UUID de la GPU")
        metadata["telemetry_probe"] = monitor.telemetry_probe(
            require_complete=True
        )
        return metadata
    finally:
        monitor.cleanup()


def main() -> int:
    args = parse_args()
    if args.status == "preflight_failed":
        if args.manifest.exists():
            raise SystemExit(
                f"no se sobrescribirá el manifiesto: {args.manifest}"
            )
        document = {
            "schema_version": 1,
            "campaign": "tres_brazos",
            "status": "preflight_failed",
            "finished_utc": utc_now(),
            "exit_code": args.exit_code,
            "preflight_attempted": str(args.preflight.resolve()),
            "preflight_exists": args.preflight.is_file(),
            "preflight_sha256": (
                sha256_file(args.preflight)
                if args.preflight.is_file()
                else None
            ),
            "schedule": flattened_schedule(),
        }
    else:
        if not args.preflight.is_file():
            raise SystemExit(f"falta preflight: {args.preflight}")
        preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
        if preflight.get("status") != "ready_without_gpu_inference":
            raise SystemExit("el preflight no está listo")

    if args.status == "running":
        if args.manifest.exists():
            raise SystemExit(
                f"no se sobrescribirá el manifiesto: {args.manifest}"
            )
        document = {
            "schema_version": 1,
            "campaign": "tres_brazos",
            "status": "running",
            "started_utc": utc_now(),
            "preflight": str(args.preflight.resolve()),
            "preflight_sha256": sha256_file(args.preflight),
            "schedule": flattened_schedule(),
            "gpu": gpu_metadata(),
        }
    elif args.status in ("complete", "failed"):
        if not args.manifest.is_file():
            raise SystemExit("no existe manifiesto running que finalizar")
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        if document.get("campaign") != "tres_brazos":
            raise SystemExit("manifiesto de otra campaña")
        if document.get("status") == "complete":
            raise SystemExit("una campaña completa es inmutable")
        if document.get("preflight") != str(args.preflight.resolve()):
            raise SystemExit("se intentó finalizar con otro preflight")
        if document.get("preflight_sha256") != sha256_file(args.preflight):
            raise SystemExit("el preflight cambió durante la campaña")
        artifacts = scan_artifacts(args.raw_dir)
        if args.status == "complete":
            if len(artifacts["raws"]) != 18:
                raise SystemExit("no hay exactamente 18 JSONL finales")
            if len(artifacts["session_manifests"]) != 18:
                raise SystemExit("no hay exactamente 18 manifiestos de sesión")
            if artifacts["partials"]:
                raise SystemExit("quedaron JSONL parciales")
            if any(
                item["status"] != "complete"
                for item in artifacts["session_manifests"]
            ):
                raise SystemExit("hay manifiestos de sesión no completos")
            if not (args.analysis_dir / "informe.md").is_file():
                raise SystemExit("falta el análisis validado")
        document.update({
            "status": args.status,
            "finished_utc": utc_now(),
            "exit_code": args.exit_code,
            "artifacts": artifacts,
            "log_artifacts": scan_tree(
                args.raw_dir.parent / "logs",
                excluded_names={"campana.log"},
            ),
            "analysis_artifacts": scan_tree(args.analysis_dir),
            "analysis_dir": (
                str(args.analysis_dir.resolve())
                if args.analysis_dir.exists()
                else None
            ),
        })

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.manifest, document)
    print(f"{args.manifest}: {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
