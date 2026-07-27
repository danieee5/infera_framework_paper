#!/usr/bin/env python3
"""Audita una campaña descargada sin modificarla ni usar GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any


STAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = STAGE_ROOT.parent
sys.path.insert(0, str(STAGE_ROOT))

from infera_kb import build_fixed_context  # noqa: E402


SUBSTANTIVE_ANALYSIS = (
    "agregado.csv",
    "efecto_por_tarea.csv",
    "ocupacion_post_intervencion.csv",
    "por_sesion.csv",
    "informe.md",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: se esperaba un objeto JSON")
    return value


def verify_artifacts(
    campaign_dir: Path,
    entries: Any,
    subdirectory: str,
    *,
    require_bytes: bool,
) -> int:
    if not isinstance(entries, list):
        raise ValueError(f"lista de artefactos inválida para {subdirectory}")
    checked = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"entrada de artefacto inválida en {subdirectory}")
        name = entry.get("name")
        expected_hash = entry.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(expected_hash, str)
        ):
            raise ValueError(f"artefacto inseguro/incompleto en {subdirectory}")
        path = campaign_dir / subdirectory / name
        if not path.is_file():
            raise ValueError(f"falta {path}")
        observed_hash = sha256_file(path)
        if observed_hash != expected_hash:
            raise ValueError(
                f"SHA-256 no coincide para {path}: "
                f"{observed_hash} != {expected_hash}"
            )
        if require_bytes and path.stat().st_size != int(entry.get("bytes", -1)):
            raise ValueError(f"tamaño no coincide para {path}")
        checked += 1
    return checked


def parse_checksum(path: Path) -> str:
    fields = path.read_text(encoding="utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise ValueError(f"checksum externo inválido: {path}")
    return fields[0].lower()


def verify_archive_contents(
    archive_path: Path,
    campaign_dir: Path,
) -> dict[str, int | bool]:
    archived_files: set[Path] = set()
    entries = 0
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            entries += 1
            relative = PurePosixPath(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] != campaign_dir.name
            ):
                raise ValueError(f"ruta insegura/inesperada en archivo: {member.name}")
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(
                    f"tipo de entrada no permitido en archivo: {member.name}"
                )
            local_relative = Path(*relative.parts[1:])
            local_path = campaign_dir / local_relative
            extracted = archive.extractfile(member)
            if extracted is None or not local_path.is_file():
                raise ValueError(f"falta archivo descargado: {local_relative}")
            digest = hashlib.sha256()
            for block in iter(lambda: extracted.read(1024 * 1024), b""):
                digest.update(block)
            if (
                digest.hexdigest() != sha256_file(local_path)
                or member.size != local_path.stat().st_size
            ):
                raise ValueError(
                    f"el contenedor no coincide con la carpeta: {local_relative}"
                )
            archived_files.add(local_relative)
    local_files = {
        path.relative_to(campaign_dir)
        for path in campaign_dir.rglob("*")
        if path.is_file()
    }
    if archived_files != local_files:
        missing = sorted(str(path) for path in local_files - archived_files)
        extra = sorted(str(path) for path in archived_files - local_files)
        raise ValueError(
            f"inventario archivo/carpeta diferente; faltan={missing}, sobran={extra}"
        )
    return {
        "entries": entries,
        "regular_files": len(archived_files),
        "paths_safe": True,
        "matches_campaign_directory": True,
    }


def prepare_relocated_copy(
    campaign_dir: Path,
    source_root: Path,
    destination: Path,
) -> Path:
    relocated = destination / "campaign"
    relocated.mkdir()
    shutil.copy2(campaign_dir / "preflight.json", relocated / "preflight.json")
    shutil.copy2(
        campaign_dir / "manifiesto_campana.json",
        relocated / "manifiesto_campana.json",
    )
    shutil.copytree(campaign_dir / "raw", relocated / "raw")

    preflight_path = relocated / "preflight.json"
    preflight = load_json(preflight_path)
    preflight["tasks"] = str(
        (source_root / "infera/config/reference/session_tasks.json").resolve()
    )
    preflight["kb_dir"] = str((source_root / "infera/kb").resolve())
    preflight_path.write_text(
        json.dumps(preflight, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    campaign_path = relocated / "manifiesto_campana.json"
    campaign = load_json(campaign_path)
    campaign["preflight"] = str(preflight_path.resolve())
    campaign["preflight_sha256"] = sha256_file(preflight_path)
    campaign_path.write_text(
        json.dumps(campaign, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return relocated


def rerun_analysis(
    campaign_dir: Path,
    source_root: Path,
    output: Path,
) -> list[str]:
    if output.exists():
        raise ValueError(f"no se sobrescribirá el reanálisis: {output}")
    analyzer = source_root / "infera/analiza_tres_brazos.py"
    if not analyzer.is_file():
        raise ValueError(f"no existe el analyzer: {analyzer}")
    with tempfile.TemporaryDirectory(prefix="infera_relocated_") as temporary:
        relocated = prepare_relocated_copy(
            campaign_dir,
            source_root,
            Path(temporary),
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(analyzer),
                "--crudos",
                str(relocated / "raw"),
                "--salida",
                str(output),
                "--expected-tasks",
                "29",
                "--max-model-len",
                "8192",
                "--pairs-kept",
                "4",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ValueError(f"el analyzer rechazó la copia reubicada: {detail}")
    differences = []
    existing = campaign_dir / "analisis"
    for name in SUBSTANTIVE_ANALYSIS:
        if (existing / name).read_bytes() != (output / name).read_bytes():
            differences.append(name)
    return differences


def audit(args: argparse.Namespace) -> dict[str, Any]:
    campaign_dir = args.campaign.resolve()
    source_root = args.source_root.resolve()
    campaign_path = campaign_dir / "manifiesto_campana.json"
    preflight_path = campaign_dir / "preflight.json"
    campaign = load_json(campaign_path)
    preflight = load_json(preflight_path)
    if campaign.get("status") != "complete" or campaign.get("exit_code") != 0:
        raise ValueError("la campaña no declara finalización exitosa")
    if sha256_file(preflight_path) != campaign.get("preflight_sha256"):
        raise ValueError("el preflight local no coincide con la campaña")

    artifacts = campaign.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("la campaña no contiene el inventario de artefactos")
    checked = {
        "raws": verify_artifacts(
            campaign_dir,
            artifacts.get("raws"),
            "raw",
            require_bytes=True,
        ),
        "session_manifests": verify_artifacts(
            campaign_dir,
            artifacts.get("session_manifests"),
            "raw",
            require_bytes=False,
        ),
        "analysis": verify_artifacts(
            campaign_dir,
            campaign.get("analysis_artifacts"),
            "analisis",
            require_bytes=True,
        ),
        "logs": verify_artifacts(
            campaign_dir,
            campaign.get("log_artifacts"),
            "logs",
            require_bytes=True,
        ),
    }
    expected_counts = {
        "raws": 18,
        "session_manifests": 18,
        "analysis": 6,
        "logs": 11,
    }
    if checked != expected_counts:
        raise ValueError(
            f"inventario incompleto/inesperado: {checked} != {expected_counts}"
        )
    if artifacts.get("partials") != []:
        raise ValueError("la campaña declara artefactos parciales")
    if list((campaign_dir / "raw").glob("*.partial")):
        raise ValueError("se encontraron raws parciales")

    tasks = source_root / "infera/config/reference/session_tasks.json"
    kb_dir = source_root / "infera/kb"
    if sha256_file(tasks) != preflight.get("tasks_sha256"):
        raise ValueError("el escenario local no coincide con el preflight")
    if sha256_text(build_fixed_context(str(kb_dir))) != preflight.get("kb_sha256"):
        raise ValueError("la KB local no coincide con el preflight")

    frozen_code = preflight.get("code_sha256")
    if not isinstance(frozen_code, dict):
        raise ValueError("el preflight no congeló el código")
    code_drift = []
    for name, expected_hash in sorted(frozen_code.items()):
        path = source_root / "infera" / name
        observed_hash = sha256_file(path) if path.is_file() else None
        if observed_hash != expected_hash:
            code_drift.append(
                {
                    "name": name,
                    "preflight_sha256": expected_hash,
                    "local_sha256": observed_hash,
                }
            )

    archive = None
    if args.archive is not None:
        expected_archive_hash = parse_checksum(args.checksum)
        observed_archive_hash = sha256_file(args.archive)
        if observed_archive_hash != expected_archive_hash:
            raise ValueError(
                "el archivo externo no coincide con su .sha256: "
                f"{observed_archive_hash} != {expected_archive_hash}"
            )
        archive = {
            "path": str(args.archive.resolve()),
            "sha256": observed_archive_hash,
            "bytes": args.archive.stat().st_size,
            **verify_archive_contents(args.archive, campaign_dir),
        }

    differences = None
    if args.reanalysis is not None:
        differences = rerun_analysis(
            campaign_dir,
            source_root,
            args.reanalysis.resolve(),
        )
        if differences:
            raise ValueError(
                "el reanálisis difiere de la descarga: "
                + ", ".join(differences)
            )

    return {
        "ok": True,
        "campaign": str(campaign_dir),
        "status": campaign.get("status"),
        "exit_code": campaign.get("exit_code"),
        "preflight_sha256": campaign.get("preflight_sha256"),
        "artifacts_checked": checked,
        "tasks": preflight.get("task_count"),
        "sessions": len(artifacts["raws"]),
        "partials": 0,
        "code_drift": code_drift,
        "archive": archive,
        "reanalysis_matches_download": (
            differences == [] if differences is not None else None
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audita una campaña descargada y, opcionalmente, repite su "
            "análisis sobre una copia temporal reubicada."
        )
    )
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--checksum", type=Path)
    parser.add_argument("--reanalysis", type=Path)
    args = parser.parse_args()
    if args.archive is not None and args.checksum is None:
        parser.error("--archive exige --checksum")
    if args.checksum is not None and args.archive is None:
        parser.error("--checksum exige --archive")
    return args


def main() -> int:
    args = parse_args()
    try:
        report = audit(args)
    except (OSError, ValueError, json.JSONDecodeError, tarfile.TarError) as exc:
        raise SystemExit(f"PAQUETE_INVALIDO: {exc}") from exc
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
