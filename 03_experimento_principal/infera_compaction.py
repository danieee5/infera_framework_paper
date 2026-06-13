"""
infera_compaction.py
Operacion de compactacion para el brazo B del Protocolo C.

Cuando el contexto acumulado cruza el umbral (el codo), se compacta:
  1. Se le pide al MISMO modelo que destile la conversacion hasta ahora en un
     "handoff" estructurado (decisiones, hechos clave, estado pendiente).
  2. Se reinicia el contexto a: KB fijo + handoff. Las tareas restantes continuan.

El COSTO ENERGETICO de esta llamada de compactacion se mide en el runner
(el "impuesto de compactacion"). Aqui solo se construyen los prompts/mensajes.
"""

COMPACTION_INSTRUCTION = (
    "Vas a COMPACTAR esta sesion de trabajo. Resume todo lo realizado hasta ahora "
    "en un HANDOFF estructurado y conciso que permita continuar sin perder informacion "
    "critica. Incluye exactamente estas secciones:\n"
    "1. DECISIONES Y DOCUMENTOS GENERADOS (memos redactados: a quien, cliente, turno, falta).\n"
    "2. HECHOS CLAVE CONSULTADOS (datos de RR-HH, permisos, turnos, personas).\n"
    "3. ESTADO / PENDIENTES.\n"
    "Se breve pero no omitas nombres propios ni datos numericos relevantes. "
    "No inventes nada que no haya aparecido en la sesion."
)


def build_compaction_messages(kb_system: str, running_history: list) -> list:
    """
    Construye los mensajes para PEDIR el handoff.
    running_history = lista de turnos [{role, content}, ...] SIN el system KB.
    """
    return (
        [{"role": "system", "content": kb_system}]
        + running_history
        + [{"role": "user", "content": COMPACTION_INSTRUCTION}]
    )


def apply_handoff(kb_system: str, handoff_text: str) -> list:
    """
    Reconstruye el historial compactado: KB fijo + el handoff como contexto previo.
    A partir de aqui el contexto vuelve a ser corto (KB + handoff).
    """
    handoff_block = (
        "===== RESUMEN DE LA SESION ANTERIOR (HANDOFF) =====\n"
        + handoff_text.strip()
        + "\n===== FIN DEL HANDOFF — CONTINUA LA SESION ====="
    )
    return [
        {"role": "system", "content": kb_system},
        {"role": "assistant", "content": handoff_block},
    ]
