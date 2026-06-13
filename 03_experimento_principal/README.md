# Etapa 3 — Experimento principal (Protocolo C)

**Sobre conjunto Energía×Calidad y rentabilidad energética de la compactación de contexto en modelos LLM pequeños auto-hospedados.**

Marco reproducible: cualquiera puede clonar esto, servir un modelo con vLLM y reproducir el experimento en su propia GPU. Es el experimento reportado en la sección de Resultados del paper (§6); usa el mismo instrumento que la calibración de la [etapa 2](../02_calibracion_sondas/) (§5.4.1) y se apoya en las decisiones de diseño tomadas allí (umbral de compactación = 4000 tokens, posición de las sondas densas).

---

## Idea en una frase

Medir, sobre el MISMO eje de contexto acumulado, dos curvas: **energía por tarea** (NVML, joules reales) y **calidad de tarea** (verificable por código). Encontrar el **codo** donde pagas más joules por peor calidad. Luego probar si **compactar** (resumir + reiniciar contexto) recupera energía *y* calidad, midiendo el **impuesto de compactación**.

---

## Diseño experimental (lo que define la arquitectura)

- **Contexto fijo (KB):** `kb/` (empresa ficticia VIGÍA). Se inyecta idéntico como mensaje de sistema en **todas** las sesiones.
- **Sesión = UNA conversación incremental.** Una secuencia ordenada de tareas heterogéneas (`session_tasks_v3.json`, 29 tareas); el contexto crece turno a turno. **Cada sesión es aislada**: no hay memoria entre sesiones, no se pregunta por otros chats.
- **Unidad de medición:** una sesión por **(cuantización × brazo × repetición)**.
  - Cuantización: FP16, AWQ INT4.
  - Brazo: `naive` (contexto crece sin compactar) | `compaction` (compacta al cruzar el umbral de 4000 tokens, calibrado en la [etapa 2](../02_calibracion_sondas/)).
  - Repeticiones: 3, igual que el protocolo de medición validado en la [etapa 1](../01_piloto_validacion_instrumento/).
- Además, `session_tasks_v3_filler.json` define la sesión de **control causal** ("filler"): mismo largo, pero con historia conversacional neutral en vez de las dependencias semánticas de la sesión principal.
- **Tareas RECALL (T11, T14, T16):** detectores de *context rot* — miden si el modelo conserva información temprana bajo carga creciente.

---

## Estructura

```
03_experimento_principal/
├── README.md
├── GUIA_DESDE_CERO.md            # guia detallada de configuracion del pod
├── EXP2_PROTOCOLO_C_METODOLOGIA.md
├── kb/                           # CONTEXTO FIJO (anonimizado, LOPD)
│   ├── vigia_kb.md
│   ├── permisos_medicos.csv
│   └── inventario_uniformes.csv
├── session_tasks_v3.json         # 29 tareas, secuencia principal
├── session_tasks_v3_filler.json  # control causal (filler)
├── infera_kb.py                  # ensambla el contexto fijo
├── infera_quality.py             # puntaje de calidad (verificable por codigo + juez opcional)
├── infera_compaction.py          # operacion de compactacion / handoff
├── infera_session_runner.py      # RUNNER principal (mide energia+tokens+calidad por tarea)
├── infera_analysis.py            # sobre energia×calidad, codo, recuperacion, figuras
├── gpu_power_monitor.py          # medicion NVML (energia + VRAM)
├── setup_infera.sh               # entorno + dependencias
├── run_all.sh                    # orquesta naive+compaction x reps con cooling
├── overnight.sh                  # corrida desatendida completa (todas las cuantizaciones)
└── results/
    ├── run_v3_<quant>_<arm>_rep<n>.jsonl
    ├── run_v3_filler_<quant>_naive_rep<n>.jsonl
    └── analisis/                 # figuras y tablas usadas en el paper (§6)
```

---

## Pasos para correr

```bash
# 1. Entorno (una vez)
bash setup_infera.sh
source /workspace/venv/bin/activate

# 2. Servir el modelo (terminal aparte). Ej. FP16:
python -m vllm.entrypoints.openai.api_server \
  --model /models/llama3.1-8b-instruct \
  --dtype float16 --max-model-len 8192 --port 8000

# 3. Verificar el contexto fijo (opcional)
python infera_kb.py kb | tail -5

# 4. Correr las sesiones de esta cuantizacion (naive + compaction x 3 reps,
#    usando session_tasks_v3.json)
./run_all.sh FP16 /models/llama3.1-8b-instruct 3 4000

# 5. Repetir 2-4 para AWQ (sirviendo el modelo AWQ).

# 6. Analisis + figuras
python infera_analysis.py --results results --out results/analisis
```

### Alternativa: corrida desatendida completa

`overnight.sh` levanta vLLM para FP16 y AWQ, corre `session_tasks_v3.json`
(naive + compaction x 3 reps cada uno) y, si `FILLER=1`, también la sesión de
control causal `session_tasks_v3_filler.json` (solo naive). Al final corre
`infera_analysis.py` automáticamente:

```bash
FILLER=1 bash overnight.sh
```

### Calibrar el umbral de compactación

El umbral usado en la sesión principal (4000 tokens) ya fue calibrado en la
[etapa 2](../02_calibracion_sondas/) con `session_tasks.json` (19 tareas). Para
recalibrarlo desde cero:
1. Corre primero **solo el brazo naive** de FP16.
2. `python infera_analysis.py` reporta el **codo** (tokens acumulados).
3. Usa ese valor como `THRESHOLD` en `run_all.sh`/`overnight.sh` para el brazo compaction.

Si el codo no se detecta, la sesión es muy corta: aumenta el número de tareas en `session_tasks_v3.json` (puedes duplicar el bloque de tareas dependientes para forzar más acumulación).

---

## Salidas

- `results/run_v3_<quant>_<arm>_rep<n>.jsonl` — un registro por tarea (energía, tokens, calidad, eventos de compactación).
- `results/analisis/envelope_<quant>.png` — figura insignia (doble eje).
- `results/analisis/recovery_naive_vs_compaction.csv` — energía total, calidad media, impuesto de compactación.
- `results/analisis/all_runs_long.csv` — datos crudos consolidados.

---

## Métrica de calidad (defendible)

- **Verificable por código (objetivo):** `contains_all`, `contains_any`, `forbidden` (anti-alucinación), `required_fields` (memos), `rota` (constraint-satisfaction de turnos).
- **Juez LLM (opcional, Nivel 3):** desactivado por defecto. Si lo activas para tareas SUMMARIZE, **valida el acuerdo juez-humano** en una muestra (Cohen's κ) y repórtalo.

---

## Reproducibilidad

Mismo protocolo de medición validado en la [etapa 1](../01_piloto_validacion_instrumento/): NVML 500 ms buffer / 100 ms / integración trapezoidal · warmup 5 requests · cooling 120 s entre corridas · seed=42 · temperatura 0.0 · `max-model-len` documentado · cache HF en volumen de red · ediciones con Python `open()`, nunca `sed -i`.
