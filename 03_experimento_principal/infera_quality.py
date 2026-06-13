"""
infera_quality.py
Puntaje de calidad por tarea (0.0 - 1.0). DV nueva del Protocolo C.

Filosofia (defendible en sustentacion):
  - Nivel 1/2 = verificable POR CODIGO (objetivo, no subjetivo):
      contains_all, contains_any, forbidden, required_fields, rota
  - Nivel 3 = juez LLM con rubrica (OPCIONAL, desactivado por defecto).
      Solo se usa para tareas abiertas (SUMMARIZE) y DEBE validarse contra
      anotacion humana en una muestra (reportar acuerdo).

Cada metodo devuelve un sub-puntaje en [0,1]. El puntaje final combina los
sub-puntajes presentes (media), y 'forbidden' actua como penalizacion
multiplicativa (alucinacion = castigo fuerte).
"""

import json
import re
from typing import Optional


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _score_contains_all(text: str, keys: list) -> float:
    t = _norm(text)
    if not keys:
        return 1.0
    hits = sum(1 for k in keys if _norm(k) in t)
    return hits / len(keys)


def _score_contains_any(text: str, groups: list) -> float:
    """groups = lista de grupos; cada grupo aporta 1 si AL MENOS una variante aparece."""
    t = _norm(text)
    if not groups:
        return 1.0
    hits = 0
    for group in groups:
        if any(_norm(v) in t for v in group):
            hits += 1
    return hits / len(groups)


def _penalty_forbidden(text: str, forbidden: list) -> float:
    """Devuelve factor multiplicativo: 1.0 si no aparece nada prohibido; baja si si."""
    t = _norm(text)
    if not forbidden:
        return 1.0
    bad = sum(1 for f in forbidden if _norm(f) in t)
    if bad == 0:
        return 1.0
    return max(0.0, 1.0 - 0.5 * bad)  # cada entidad prohibida resta 0.5


def _score_required_fields(text: str, fields: list) -> float:
    t = _norm(text)
    if not fields:
        return 1.0
    hits = sum(1 for f in fields if _norm(f) in t)
    return hits / len(fields)


def _extract_json_array(text: str):
    """Extrae el primer array JSON del texto (para tareas de rota). None si falla."""
    # bloque ```json ... ```
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"(\[\s*\{.*\}\s*\])", text, re.DOTALL)
        raw = m.group(1) if m else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _score_rota(text: str, spec: dict) -> float:
    """
    Constraint-satisfaction objetivo. spec: {posts, days, shifts, guards}.
    Verifica 3 restricciones y devuelve la fraccion satisfecha:
      C1 cobertura: cada (dia, turno, puesto) tiene exactamente 1 guardia valido.
      C2 no doble turno el mismo dia: ningun guardia en dos turnos del mismo dia.
      C3 descanso: ningun guardia NOCTURNO dia d y DIURNO dia d+1 (solo si shifts incluye DIURNO/NOCTURNO).
    Si el JSON no parsea -> 0.0 (el modelo no produjo salida verificable).
    """
    arr = _extract_json_array(text)
    if not arr or not isinstance(arr, list):
        return 0.0

    posts = spec["posts"]
    days = spec["days"]
    shifts = [s.upper() for s in spec["shifts"]]
    guards = set(spec["guards"])

    entries = []
    for e in arr:
        if not isinstance(e, dict):
            continue
        try:
            g = str(e.get("guardia", "")).upper().strip()
            d = int(e.get("dia"))
            t = str(e.get("turno", "")).upper().strip()
            p = int(e.get("puesto"))
        except Exception:
            continue
        entries.append((g, d, t, p))

    # C1 cobertura
    needed = {(d, t, p) for d in range(1, days + 1) for t in shifts for p in range(1, posts + 1)}
    covered = {(d, t, p) for (g, d, t, p) in entries if g in guards and (d, t, p) in needed}
    c1 = len(covered) / len(needed) if needed else 1.0

    # C2 no doble turno el mismo dia
    by_guard_day = {}
    for (g, d, t, p) in entries:
        by_guard_day.setdefault((g, d), set()).add(t)
    viol_c2 = sum(1 for turns in by_guard_day.values() if len(turns) > 1)
    total_gd = max(1, len(by_guard_day))
    c2 = max(0.0, 1.0 - viol_c2 / total_gd)

    # C3 descanso nocturno->diurno
    c3 = 1.0
    if "NOCTURNO" in shifts and "DIURNO" in shifts and days >= 2:
        noct = {(g, d) for (g, d, t, p) in entries if t == "NOCTURNO"}
        diur = {(g, d) for (g, d, t, p) in entries if t == "DIURNO"}
        viol_c3 = sum(1 for (g, d) in noct if (g, d + 1) in diur)
        c3 = max(0.0, 1.0 - viol_c3 / max(1, len(noct)))

    return (c1 + c2 + c3) / 3.0


def score_task(response_text: str, verify: dict, judge_fn: Optional[callable] = None) -> dict:
    """
    Calcula el puntaje de calidad de una respuesta dada su spec 'verify'.
    judge_fn(response_text, rubric) -> float en [0,1], opcional (Nivel 3).
    Devuelve dict con sub-puntajes y 'quality' final.
    """
    subs = {}

    if "contains_all" in verify:
        subs["contains_all"] = _score_contains_all(response_text, verify["contains_all"])
    if "contains_any" in verify:
        subs["contains_any"] = _score_contains_any(response_text, verify["contains_any"])
    if "required_fields" in verify:
        subs["required_fields"] = _score_required_fields(response_text, verify["required_fields"])
    if "rota" in verify:
        subs["rota"] = _score_rota(response_text, verify["rota"])
    if "judge" in verify and judge_fn is not None:
        try:
            subs["judge"] = float(judge_fn(response_text, verify["judge"]))
        except Exception:
            pass  # juez opcional; si falla, se omite

    base = sum(subs.values()) / len(subs) if subs else 1.0
    penalty = _penalty_forbidden(response_text, verify.get("forbidden", []))
    quality = round(base * penalty, 4)

    return {"quality": quality, "subscores": subs, "forbidden_penalty": penalty}
