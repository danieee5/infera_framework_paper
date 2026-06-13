# CHECKLIST DE VERIFICACIÓN — EXPERIMENTO 2 (EXP2)
**Revisado:** 2026-06-04  
**Estado general:** ✅ LISTO PARA CORRER (tras correcciones aplicadas)

---

## CORRECCIONES APLICADAS (antes inexistentes)

Estas correcciones fueron realizadas automáticamente durante esta revisión. Deben incluirse en el próximo `git commit`:

| Problema | Solución aplicada |
|----------|-------------------|
| `build_multiturn_conversation.py` estaba en la raíz, no en `scripts/` | Movido a `scripts/` |
| `multiturn_runner.py` estaba en la raíz, no en `scripts/` | Movido a `scripts/` |
| `multiturn_analysis.py` estaba en la raíz, no en `scripts/` | Movido a `scripts/` |
| `conversation_flow.json` estaba en la raíz, no en `data/multiturn/` | Movido a `data/multiturn/` |
| `DISEÑO_MULTITURN.md` estaba en la raíz, no en `docs/` | Movido a `docs/` |
| `data/multiturn/` no existía | Directorio creado |
| `data/multiturn/conversation_history.json` no estaba en `.gitignore` | Añadido al `.gitignore` |

**Commit de cierre recomendado:**
```bash
cd /workspace/infera
git add scripts/build_multiturn_conversation.py scripts/multiturn_runner.py scripts/multiturn_analysis.py
git add data/multiturn/conversation_flow.json
git add docs/DISEÑO_MULTITURN.md docs/CHECKLIST_EXP2.md
git add .gitignore
git rm build_multiturn_conversation.py multiturn_runner.py multiturn_analysis.py 2>/dev/null || true
git rm conversation_flow.json DISEÑO_MULTITURN.md 2>/dev/null || true
git commit -m "feat: EXP2 scripts y datos en rutas correctas; actualizar .gitignore"
git push origin main
```

---

## T1 — ESTRUCTURA DEL REPO

```
titan_framework_paper/
├── scripts/
│   ├── build_multiturn_conversation.py  ✓ (movido desde raíz)
│   ├── multiturn_runner.py              ✓ (movido desde raíz)
│   ├── multiturn_analysis.py            ✓ (movido desde raíz)
│   ├── benchmark_runner.py              ✓
│   ├── gpu_power_monitor.py             ✓
│   ├── consolidate_results.py           ✓
│   ├── build_prompt_dataset.py          ✓ (renombrado vs diseño: OK para EXP2)
│   ├── start_vllm_fp16.sh               ✓
│   ├── start_vllm_int8.sh               ✓
│   └── start_vllm_awq.sh                ✓
├── data/
│   ├── prompts/                         ✓ (EXP1)
│   └── multiturn/
│       └── conversation_flow.json       ✓ (movido desde raíz)
├── results/
│   ├── infera_results_raw.csv           ✓ (EXP1, commiteado)
│   └── infera_summary.csv              ✓ (EXP1, commiteado)
├── docs/
│   ├── DISEÑO_MULTITURN.md             ✓ (movido desde raíz)
│   └── CHECKLIST_EXP2.md              ✓ (este archivo)
├── INFERA_paper_v3.md                  ✗ NO encontrado (no crítico para EXP2)
└── README.md                           ✓
```

**Nota:** `results/multiturn/` no existe aún — se crea automáticamente cuando corre `multiturn_runner.py`. ✓

---

## T2 — `scripts/build_multiturn_conversation.py`

| Verificación | Estado | Detalle |
|---|---|---|
| Sin `run_in_executor` dentro de `build_conversation()` | ✓ OK | Solo hay `result = await send_turn_streaming(vllm_messages)` |
| Sin `asyncio.run(...)` dentro de `build_conversation()` | ✓ OK | El único `asyncio.run` está en `__main__` (línea 238) |
| `FLOW_PATH = Path("data/multiturn/conversation_flow.json")` | ✓ OK | Correcto, relativo al CWD `/workspace/infera/` |
| `OUTPUT_PATH = Path("data/multiturn/conversation_history.json")` | ✓ OK | |
| `OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)` | ✓ OK | Línea 136 |
| `stream=True` con TTFT al primer chunk no vacío | ✓ OK | Líneas 59 y 89–90 |
| `stream_options: {include_usage: True}` | ✓ OK | Línea 60 |
| Salida incluye `system_prompt`, `document_content`, `turns` | ✓ OK | Líneas 206–223 |
| Campos por turno: `user_content`, `assistant_content`, `ttft_ms`, `total_time_s` | ✓ OK | Líneas 184–193 |

---

## T3 — `scripts/multiturn_runner.py`

| Verificación | Estado | Detalle |
|---|---|---|
| `from gpu_power_monitor import GPUPowerMonitor` con `sys.path.insert` | ✓ OK | Líneas 58–59: inserta `Path(__file__).parent` = `scripts/` → encuentra `gpu_power_monitor.py` |
| `HISTORY_PATH = Path("data/multiturn/conversation_history.json")` | ✓ OK | Línea 75, relativo a CWD `/workspace/infera/` |
| `RESULTS_DIR = Path("results/multiturn")` con `mkdir(parents=True, exist_ok=True)` | ✓ OK | Líneas 76 y 426 |
| `tpot_ms`: `wall_time / ct_total * 1000` (throughput servidor) | ✓ OK | Línea 369 |
| `tpot_ms_per_request`: `wall_time / ct_per_req * 1000` (latencia por usuario) | ✓ OK | Línea 370 |
| `ct_per_req = max(1, ct_total // n_req)` con `n_req = max(1, vr.get("n_success", batch_size))` | ✓ OK | Líneas 357–360 |
| `prompt_tokens_per_request` | ✓ OK | Línea 384 |
| `completion_tokens_per_request` | ✓ OK | Línea 385 |
| `total_context_tokens` (= pt_per_req) | ✓ OK | Línea 386 |
| `vram_delta_mb` (peak − start) | ✓ OK | Líneas 331 y 371 |
| `COOLING_S = 120` aplicado entre turnos | ✓ OK | Líneas 504–506 |
| Cooling también entre reps | ✓ OK | Líneas 508–510 |
| Cooling 2× entre batch sizes | ✓ OK | Líneas 512–514 |
| `build_messages_for_turn`: estructura correcta [sys, u1, a1, ..., uN] | ✓ OK | Líneas 94–101 — el último mensaje es siempre el user sin respuesta |
| `asyncio.gather` envía `batch_size` copias idénticas | ✓ OK | Líneas 210–214 |
| `--pilot` → solo `rep=1` | ✓ OK | Línea 422 |
| `--batch-sizes 1 --pilot` → 7 mediciones | ✓ OK | Combinación correcta |

---

## T4 — `data/multiturn/conversation_flow.json`

| Verificación | Estado | Detalle |
|---|---|---|
| 7 turnos exactos | ✓ OK | `turns: [1,2,3,4,5,6,7]` |
| Validez JSON (`json.load`) | ✓ OK | Parseado sin errores |
| Campos por turno: `turn_number`, `role`, `content`, `expected_input_tokens_approx` | ✓ OK | Todos presentes |
| Turnos cubren: resumen → seguimiento → comparación → tabla → capacitación → correo → meta-resumen | ✓ OK | Confirmado en los 7 turnos |
| `documents.combined_content` longitud | ⚠ NOTA | **3.953 chars (~988 tokens)**, NO ~6.000 chars como indicaba el diseño inicial. El contexto real en T1 será ~1.100 tokens (no ~1.600 como estima `expected_input_tokens_approx`). Ver nota abajo. |

**⚠ Nota sobre longitud de documentos:**  
El contenido combinado tiene ~988 tokens estimados (3.953 chars ÷ 4), no ~1.500 tokens. En consecuencia:
- `expected_input_tokens_approx` en todos los turnos sobreestima ~30–40%.
- Los valores reales se confirmarán durante Fase 1 si vLLM devuelve `stream_options.include_usage`.
- El experimento sigue siendo válido: el contexto crece orgánicamente de T1 a T7, y las hipótesis H1–H4 son comprobables. Solo el bridge exacto con Case_B/Case_C del EXP1 estará en el rango medio (~1.100–~2.600 tokens), no en el extremo.
- **Acción recomendada:** después de correr Fase 1, actualizar los `expected_input_tokens_approx` con los valores reales del API.

---

## T5 — CONSISTENCIA CON EXP1

| Verificación | Estado | Detalle |
|---|---|---|
| `gpu_power_monitor.py` no modificado | ✓ OK | Intacto en `scripts/` |
| `BUFFER_MS = 500` | ✓ OK | |
| `SAMPLING_MS = 100` | ✓ OK | 10 Hz |
| Patrón MELODI idéntico: `start_monitoring()` → inference → `stop_monitoring()` | ✓ OK | Líneas 277–280 en `multiturn_runner.py` |
| `VLLM_URL = "http://localhost:8000/v1/chat/completions"` | ✓ OK | Igual en ambos runners |
| `MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"` | ✓ OK | Igual en ambos runners |
| `COOLING_S = 120` | ✓ OK | Igual en ambos runners |

---

## T6 — LIMPIEZA DEL REPO

| Verificación | Estado | Detalle |
|---|---|---|
| Sin `__pycache__` o `.pyc` fuera de venv | ✓ OK | Ninguno encontrado |
| `.gitignore` excluye `results/*/` (raw runs EXP1) | ✓ OK | Ya estaba |
| `.gitignore` excluye `__pycache__/`, `*.pyc` | ✓ OK | Ya estaba |
| `.gitignore` excluye `data/multiturn/conversation_history.json` | ✓ AÑADIDO | Añadido durante esta revisión |
| EXP1 CSVs commiteados (`infera_summary.csv`, `infera_results_raw.csv`) | ✓ OK | Presentes en `results/` |

---

## T7 — SINTAXIS DE SCRIPTS PYTHON

| Script | Estado |
|---|---|
| `scripts/benchmark_runner.py` | ✓ OK |
| `scripts/build_multiturn_conversation.py` | ✓ OK |
| `scripts/build_prompt_dataset.py` | ✓ OK |
| `scripts/consolidate_results.py` | ✓ OK |
| `scripts/generate_reproducibility_info.py` | ✓ OK |
| `scripts/gpu_power_monitor.py` | ✓ OK |
| `scripts/multiturn_analysis.py` | ✓ OK |
| `scripts/multiturn_runner.py` | ✓ OK |

**Todos los scripts pasan `ast.parse` sin errores.**

---

## COMANDOS EXACTOS PARA CORRER EXP2 (desde cero en RunPod)

```bash
# ═══════════════════════════════════════════════════════════════
# PASO 0: Reiniciar pod desde RunPod dashboard → botón Start
# ═══════════════════════════════════════════════════════════════

# PASO 1: Verificar entorno
source /workspace/venv/bin/activate
python3 -c "import vllm, pynvml, httpx; print('OK — vllm, pynvml, httpx disponibles')"
ls /models/

# PASO 2: Pull del repo (incluye las correcciones de esta revisión)
cd /workspace/infera
git pull origin main

# Verificar estructura post-pull:
ls scripts/build_multiturn_conversation.py scripts/multiturn_runner.py scripts/multiturn_analysis.py
ls data/multiturn/conversation_flow.json

# ═══════════════════════════════════════════════════════════════
# FASE 1: Pre-generar conversación (UNA SOLA VEZ para todo EXP2)
# ═══════════════════════════════════════════════════════════════

# PASO 3: Levantar vLLM FP16
bash scripts/start_vllm_fp16.sh
# Esperar 60–90s que vLLM levante. Verificar:
# curl -s http://localhost:8000/health && echo "vLLM OK"

# PASO 4: Generar la historia de conversación
cd /workspace/infera
python scripts/build_multiturn_conversation.py

# Verificar salida:
python3 -c "
import json
h = json.load(open('data/multiturn/conversation_history.json'))
print(f'Turns: {len(h[\"turns\"])} (esperado: 7)')
print(f'Token counts: {\"API (exact)\" if h.get(\"token_counts_from_api\") else \"estimated\"}')
for t in h['turns']:
    print(f'  T{t[\"turn_number\"]}: ct={t[\"completion_tokens\"]} ttft={t[\"ttft_ms\"]}ms')
"
# Esperado: 7 turnos, todos con assistant_content no vacío, ttft_ms > 0

# ═══════════════════════════════════════════════════════════════
# FASE 2: Mediciones
# ═══════════════════════════════════════════════════════════════

# PASO 5: PILOTO FP16 (~15–20 min)
python scripts/multiturn_runner.py --quantization fp16 --batch-sizes 1 --pilot

# SEÑALES DE ÉXITO DEL PILOTO:
# ✓ 7 líneas con "✓" y status=success
# ✓ energy_j crece de T1 a T7
# ✓ vram_peak_mb crece de T1 a T7
# ✓ ttft_ms crece de T1 a T7
# ✓ nvml_samples ≥ 10 por turno
# ✓ total_context_tokens sube en cada turno
# Si token_count_source = "estimated": aceptable, pero notar en paper
# SI ALGUNA señal falla → NO continuar. Revisar logs.

# PASO 6: FP16 corrida completa (~3h)
python scripts/multiturn_runner.py --quantization fp16

# PASO 7: INT8
pkill -f "python.*vllm"  # o Ctrl+C en la terminal vLLM
sleep 30
bash scripts/start_vllm_int8.sh
# Esperar 60–90s
python scripts/multiturn_runner.py --quantization int8_w8a16

# PASO 8: AWQ
pkill -f "python.*vllm"
sleep 30
bash scripts/start_vllm_awq.sh
# Esperar 60–90s
python scripts/multiturn_runner.py --quantization int4_awq

# ═══════════════════════════════════════════════════════════════
# FASE 3: Análisis
# ═══════════════════════════════════════════════════════════════

# PASO 9: Análisis de hipótesis
python scripts/multiturn_analysis.py --results-dir results/multiturn/

# CSVs exportados a results/multiturn/analysis/
```

---

## SEÑALES DE ÉXITO DEL PILOTO (7 mediciones FP16 batch=1)

| Métrica | Señal esperada | Si no se cumple |
|---|---|---|
| `status` | `"success"` en los 7 turnos | Error de vLLM o timeout → revisar `bash scripts/start_vllm_fp16.sh` |
| `energy_j` | Crece T1 → T7 | KV-cache no contabilizado → revisar MELODI buffer |
| `vram_peak_mb` | Crece T1 → T7 | Acumulación de KV-cache no observable → posible restart de vLLM |
| `ttft_ms` | Crece T1 → T7 | Prefill no crece → verificar `build_messages_for_turn` |
| `nvml_samples` | ≥ 10 por turno | GPU demasiado rápida (improbable en T7) o pynvml falla |
| `total_context_tokens` | Sube cada turno | `pt_total = 0` (stream_options no soportado) → usar `input_tokens_approx` como fallback |

**Contexto esperado por turno (estimado, se confirmará con API):**

| Turno | Tokens aprox. | Contexto |
|---|---|---|
| T1 | ~1.100 | Docs + system + Q1 |
| T2 | ~1.370 | + respuesta T1 (~256 tok) |
| T3 | ~1.640 | + respuesta T2 |
| T4 | ~1.920 | + respuesta T3 |
| T5 | ~2.200 | + respuesta T4 |
| T6 | ~2.480 | + respuesta T5 |
| T7 | ~2.760 | + respuesta T6 |

*Nota: los valores son menores que los `expected_input_tokens_approx` del JSON porque `combined_content` tiene ~988 tokens, no ~1.500. Los valores reales se confirman en Fase 1.*

---

*Generado por revisión automática — 2026-06-04*
