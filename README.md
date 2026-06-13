# INFERA — Energía y calidad en inferencia de LLM auto-hospedados

**Trabajo de titulación — Universidad de Especialidades Espíritu Santo (UEES), Ecuador, 2026**
*Autora: Daniela Mora*

---

## ¿Qué es esto?

Este repositorio es el entregable reproducible del estudio reportado en
[`INFERA_paper_pivote_hasta_resultados.md`](./INFERA_paper_pivote_hasta_resultados.md):
una caracterización de cómo varían el **consumo de energía** (medido vía
NVML) y la **calidad de respuesta** de un modelo LLaMA 3.1 8B Instruct
auto-hospedado, a medida que crece el contexto acumulado de una sesión de
chat, y de si **compactar el contexto** (resumir + reiniciar) es
energéticamente rentable.

El trabajo se ejecutó en tres etapas, cada una en su propia carpeta numerada.
Las tres comparten el mismo hardware (GPU NVIDIA RTX 4090, 24 GB) y el mismo
modelo servido con vLLM.

---

## Las tres etapas

| Carpeta | Nombre | Qué es | Relación con el paper |
|---|---|---|---|
| [`01_piloto_validacion_instrumento/`](./01_piloto_validacion_instrumento/) | Piloto de validación del instrumento | Diseño factorial 3⁴×3 = 243 corridas con peticiones aisladas (sin sesión incremental). Valida el protocolo de medición de energía (NVML, buffer 500 ms) y ancla la relación energía-vs-contexto corto. | §5.3, CV reportado en §6.1 |
| [`02_calibracion_sondas/`](./02_calibracion_sondas/) | Calibración de sondas | Sesión incremental de 19 tareas (3 de ellas "sondas" de calidad). Se usó para decidir el umbral de compactación (4000 tokens) y la posición de las sondas densas de la sesión final. | §5.4.1 |
| [`03_experimento_principal/`](./03_experimento_principal/) | Experimento principal (Protocolo C) | 12 sesiones incrementales de 29 tareas (2 esquemas × 2 brazos × 3 repeticiones), más una sesión de control causal ("filler"). Es el experimento reportado en Resultados. | §5.4.2–§5.5, §6 |

El orden de lectura recomendado es 01 → 02 → 03: cada etapa usa lo aprendido
en la anterior. Cada carpeta tiene su propio `README.md` con instrucciones de
ejecución.

---

## Estructura del repositorio

```
.
├── README.md                              ← este archivo
├── INFERA_paper_pivote_hasta_resultados.md  ← paper completo
├── INFERA_tablas_revision.docx            ← tablas del paper en formato Word
├── requirements.txt                       ← dependencias Python ancladas
│
├── 01_piloto_validacion_instrumento/      ← Etapa 1: validación del instrumento (243 corridas)
├── 02_calibracion_sondas/                 ← Etapa 2: calibración (19 tareas)
├── 03_experimento_principal/              ← Etapa 3: Protocolo C (experimento principal)
│
├── referencias/                           ← lecturas de referencia del marco teórico
└── _archivo/                              ← diseños anteriores, ya no vigentes (ver su README)
```

---

## Requisitos de hardware y software

- GPU NVIDIA con ≥ 24 GB de VRAM (RTX 4090 o equivalente), instancia dedicada
  (NVML mide la potencia total de la tarjeta; instancias compartidas
  contaminan la medición).
- CUDA 12.1, PyTorch 2.3.1, Python 3.10, vLLM 0.5.3.
- Ver [`requirements.txt`](./requirements.txt) para las versiones exactas.

Cada carpeta de etapa documenta sus propios pasos de ejecución; las etapas 02
y 03 comparten el mismo instrumento (`infera_session_runner.py`, dentro de
`03_experimento_principal/`).

---

## `_archivo/`

Contiene un diseño experimental anterior (multi-turno, empresa ficticia
"MOSS") que fue reemplazado por el diseño VIGÍA/Protocolo C usado en las
etapas 02 y 03, y un borrador previo del paper. Se conservan por
trazabilidad; no forman parte del estudio reportado. Ver
[`_archivo/README.md`](./_archivo/README.md).
