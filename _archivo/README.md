# Archivo

Esta carpeta conserva material que **no forma parte del estudio reportado**
en el paper, pero se mantiene en el repositorio por trazabilidad del proceso
de investigación.

---

## `diseno_exp2_moss_abandonado/`

Diseño preliminar de un "Experimento 2" multi-turno, sobre una empresa
ficticia de seguridad privada ("MOSS"), en el que cada turno de una
conversación dependía del historial acumulado y de documentos internos bajo
re-prefill completo. Este diseño fue **reemplazado** por el caso VIGÍA y el
Protocolo C (etapas 2 y 3 del repo), que usan una sesión incremental con
tareas heterogéneas en vez de turnos conversacionales homogéneos.

Contenido:
- `CHECKLIST_EXP2.md`, `DISEÑO_MULTITURN.md`, `EXP2_METODOLOGIA.md` — diseño y
  checklist de esa versión preliminar.
- `multiturn_runner.py`, `multiturn_analysis.py`,
  `build_multiturn_conversation.py`, `setup_exp2.sh` — scripts asociados.
- `data/conversations/conversation_flow_v2_1_corrected.json`,
  `data/multiturn/conversation_flow.json` — conversaciones de ejemplo
  diseñadas para ese experimento (empresa "MOSS").

> Nota: `data/conversations/conversation_histories.jsonl` (30 historiales
> genéricos de soporte técnico) **no** está aquí — sigue en uso en
> [`01_piloto_validacion_instrumento/data/conversations/`](../01_piloto_validacion_instrumento/data/conversations/)
> como insumo del Caso B del piloto.

---

## `borrador_paper_protocolo_c.md`

Borrador temprano del paper del Protocolo C, escrito solo hasta la
metodología (sin resultados). Superado por
[`INFERA_paper_pivote_hasta_resultados.md`](../INFERA_paper_pivote_hasta_resultados.md)
en la raíz del repositorio, que es la versión vigente.
