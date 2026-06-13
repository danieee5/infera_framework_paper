# Etapa 2 — Calibración de sondas

Sesión incremental de **19 tareas** (`session_id: vigia_rrhh_ops_v2`), corrida
con el mismo instrumento que la etapa 3
(`03_experimento_principal/infera_session_runner.py`), sobre los dos esquemas
de cuantización (FP16 y AWQ INT4) y los dos brazos (`naive` y `compaction`),
con 3 repeticiones cada uno (12 archivos de resultados).

Referencia en el paper: §5.4.1 (subfase 2.1).

---

## Para qué sirvió

Esta sesión incluye **3 sondas de calidad** (`SONDA_A`, `SONDA_B`, `SONDA_C`)
distribuidas a lo largo de la conversación, en aproximadamente 3300, 4650 y
5950 tokens de contexto acumulado. Son preguntas puntuales contra el
conocimiento de proyecto, repetidas en distintas posiciones para observar
cómo cambia la calidad de respuesta a medida que crece el contexto.

A partir de los datos de esta calibración se tomaron **dos decisiones de
diseño** para la sesión final (etapa 3):

1. **Umbral de compactación = 4000 tokens** (aproximadamente la mitad de la
   ventana de contexto del modelo, configurada en 8192 tokens).
2. **Posición de las sondas densas** de la sesión final: se concentraron en
   la franja de contexto cubierta por estas 3 sondas de calibración, para
   localizar el "codo" de degradación con mayor precisión.

---

## Cómo se ejecutó

Se usa el mismo runner que la etapa 3, apuntando a `session_tasks.json` de
esta carpeta. Ejecutar desde `03_experimento_principal/` (donde vive
`infera_session_runner.py` y la KB en `kb/`):

```bash
cd ../03_experimento_principal

python infera_session_runner.py \
  --vllm-url http://localhost:8000/v1/chat/completions \
  --model /models/llama3.1-8b-instruct \
  --quant FP16 --arm naive --rep 1 \
  --kb-dir kb \
  --tasks ../02_calibracion_sondas/session_tasks.json \
  --compaction-threshold 4000 \
  --out ../02_calibracion_sondas/results/run_FP16_naive_rep1.jsonl
```

Repetir variando `--quant` (FP16/AWQ), `--arm` (naive/compaction) y `--rep`
(1-3) para los 12 archivos de `results/`.

### Análisis

```bash
python infera_analysis.py \
  --results ../02_calibracion_sondas/results \
  --out ../02_calibracion_sondas/results/analisis
```

---

## Estructura

```
02_calibracion_sondas/
├── README.md
├── session_tasks.json           ← 19 tareas, incluye SONDA_A/B/C
└── results/
    ├── run_{AWQ,FP16}_{naive,compaction}_rep{1,2,3}.jsonl   (12 archivos)
    └── analisis/                ← figuras y tablas de esta calibración
```
