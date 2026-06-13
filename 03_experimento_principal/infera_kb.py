"""
infera_kb.py
Carga el conocimiento de proyecto (contexto fijo) y lo ensambla en un unico
bloque de sistema que se inyecta identico en TODAS las sesiones.

El contexto fijo = KB markdown + CSVs renderizados como tablas de texto.
Esto simula un "proyecto" tipo ChatGPT/Claude con conocimiento cargado.
"""

import csv
import io
from pathlib import Path


def _render_csv(path: Path, max_rows: int = 100) -> str:
    """Renderiza un CSV como tabla de texto plano legible por el modelo."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header, data = rows[0], rows[1:max_rows + 1]
    out = io.StringIO()
    out.write(" | ".join(header) + "\n")
    for r in data:
        out.write(" | ".join(r) + "\n")
    return out.getvalue()


def build_fixed_context(kb_dir: str) -> str:
    """
    Construye el bloque de contexto fijo (string) a partir de:
      - vigia_kb.md          (texto principal)
      - permisos_medicos.csv (tabla)
      - inventario_uniformes.csv (tabla)
    Devuelve el string completo que se usara como mensaje de sistema.
    """
    kb = Path(kb_dir)
    parts = []

    md = kb / "vigia_kb.md"
    if md.exists():
        parts.append(md.read_text(encoding="utf-8"))

    permisos = kb / "permisos_medicos.csv"
    if permisos.exists():
        parts.append("\n## ANEXO — PERMISOS MEDICOS (RR-HH)\n\n" + _render_csv(permisos))

    uniformes = kb / "inventario_uniformes.csv"
    if uniformes.exists():
        parts.append("\n## ANEXO — INVENTARIO DE UNIFORMES (RR-HH)\n\n" + _render_csv(uniformes))

    header = (
        "Eres el asistente interno de VIGIA Seguridad S.A. "
        "Responde unicamente con base en el siguiente conocimiento de proyecto. "
        "No inventes personas, clientes ni datos que no esten aqui.\n\n"
        "===== CONOCIMIENTO DE PROYECTO (CONTEXTO FIJO) =====\n\n"
    )
    return header + "\n".join(parts)


if __name__ == "__main__":
    import sys
    ctx = build_fixed_context(sys.argv[1] if len(sys.argv) > 1 else "kb")
    print(ctx)
    print(f"\n[INFO] Caracteres del contexto fijo: {len(ctx)}")
