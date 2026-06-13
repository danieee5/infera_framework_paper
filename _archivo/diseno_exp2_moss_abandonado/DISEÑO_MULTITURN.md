# DISEÑO EXPERIMENTAL — INFERA Experimento 2: Evolución Energética en Conversación Multi-Turno

**Versión:** 1.0 — Junio 2026  
**Estado:** Diseño aprobado — pendiente ejecución  
**Relación con Experimento 1:** Extensión complementaria del factorial estático 3⁴

---

## 1. Motivación y posicionamiento

El Experimento 1 (factorial estático) caracterizó el comportamiento energético bajo configuraciones *controladas y estáticas*: un contexto fijo, un tamaño de output fijo, concurrencia fija. Ese diseño es metodológicamente riguroso para estimar efectos principales e interacciones, pero no representa cómo las aplicaciones empresariales realmente usan los LLMs.

En un entorno empresarial real — asistente de RRHH, soporte interno, análisis documental —, el usuario no envía prompts aislados. **Conversa.** Cada turno hereda el historial de los anteriores. El contexto crece. El KV-cache se llena. Las fases de prefill y decode se rebalancean.

La literatura revisada (Watt Counts 2026, From Prompts to Power 2025, TokenPowerBench 2025) cubre contextos largos en solicitudes individuales, pero ningún trabajo caracteriza de forma sistemática la evolución energética turno a turno bajo distintos esquemas de cuantización. Este experimento cubre ese vacío.

**Formulación del gap (defendible en jurado):**  
> "Hasta donde conocemos, la literatura revisada no caracteriza de manera sistemática la evolución energética de las conversaciones multi-turno bajo distintos esquemas de cuantización en hardware de consumo."

---

## 2. Diseño experimental

### 2.1 Estructura

| Dimensión | Valores |
|---|---|
| VI1 — Cuantización | fp16, int8_w8a16 (bitsandbytes), int4_awq |
| VI2 — Batch size | 1, 4 |
| Turno | 1, 2, 3, 4, 5, 6, 7 |
| Repeticiones | 3 |
| **Total mediciones** | **3 × 2 × 7 × 3 = 126** |

batch=8 se excluye: 8 usuarios simultáneos en la *misma conversación activa* no representa un escenario empresarial creíble para un asistente interno.

### 2.2 Conversación empresarial — escenario MOSS

**Escenario:** Asistente interno RAG de MOSS Operaciones (empresa de seguridad privada). El usuario es un supervisor consultando el manual operativo para resolver una situación real.

**Documentos de contexto:** ~1500 tokens. Incluyen: Manual Operativo de Guardias (procedimientos, obligaciones), Procedimiento de Reporte de Novedades, Protocolo de Emergencias, Política de Escalamiento y RRHH.

**Flujo de turnos:**

| Turn | Pregunta | Tipo de tarea | Contexto aprox. |
|---|---|---|---|
| 1 | Resume el procedimiento para reportar una novedad | Síntesis | ~1600 tok |
| 2 | ¿Qué pasa si el supervisor no responde? | Seguimiento | ~1870 tok |
| 3 | Compara escalamiento de emergencia vs novedades | Comparación | ~2140 tok |
| 4 | Crea tabla de responsabilidades por rol | Síntesis estructurada | ~2420 tok |
| 5 | Escribe capacitación para guardia nuevo | Generación | ~2700 tok |
| 6 | Redacta correo formal a RRHH | Generación con formato | ~2980 tok |
| 7 | Resume conversación en 5 acciones concretas | Meta-síntesis | ~3250 tok |

**Crecimiento de contexto:** ~1600 tokens (T1) → ~3250 tokens (T7).  
**Mapeo a factorial estático:** T1 ≈ Case_B (~1024 tok), T5–T7 ≈ Case_C (~4096 tok).

### 2.3 Protocolo de medición por turno

```
Para cada (quantization, batch_size, repetition):
  Para cada turn in [1..7]:
    1. Idle wait: 5s (igual que factorial estático)
    2. start_monitoring()  ← 500ms pre-buffer MELODI
    3. asyncio.gather(batch_size × [messages_hasta_turno_N])
    4. energy = stop_monitoring()  ← 500ms post-buffer MELODI
    5. Registrar métricas
    6. Cooling: 120s (mismo que factorial estático)
  Cooling adicional entre repeticiones: 120s
Cooling adicional entre batch sizes: 240s
```

**Historia conversacional:** Pre-generada una vez con FP16, temperature=0, batch=1. Fijada para todos los esquemas y repeticiones. Con temperature=0, todos los esquemas producirían respuestas idénticas; pre-generar explícitamente hace el diseño auditable.

**TTFT:** Medido vía streaming (tiempo al primer chunk no vacío). Es la métrica que revela el crecimiento del costo de prefill en turnos tardíos.

---

## 3. Hipótesis formalizadas

**H1 — Escalación energética:** La energía por output_token (J/tok) aumenta de forma monótona con el número de turno para todos los esquemas y tamaños de batch, reflejando el crecimiento del KV-cache y el costo cuadrático del prefill con el contexto acumulado.

*Test:* Correlación de Spearman entre turn_number y J/tok_mean por (quantization, batch). ρ > 0, p < 0.05.

**H2 — Persistencia de la anomalía INT8:** La anomalía energética de bitsandbytes INT8 a batch=4 (observada en el factorial estático como +18% J/tok respecto a batch=1) persiste y se amplifica en turnos tardíos, donde el mayor contexto incrementa el overhead de dequantización dinámica.

*Test:* Ratio = (INT8_b4 J/tok) / (INT8_b1 J/tok) por turno. ¿El ratio en turnos 6–7 > ratio en turnos 1–2?

**H3 — Erosión de la ventaja AWQ:** La ventaja energética de AWQ sobre FP16 (≈42% en contexto corto, hallazgo del factorial estático) se erosiona en turnos tardíos, consistente con la penalidad asimétrica de contexto (+122% AWQ vs +59% FP16) documentada en el factorial.

*Test:* Ratio = (AWQ_b1 J/tok) / (FP16_b1 J/tok) por turno. ¿El ratio aumenta de T1 a T7 (AWQ converge hacia FP16)?

**H4 — KV-cache como proxy VRAM:** El pico de VRAM (VRAM_peak − VRAM_start) crece de forma monótona con el turno para todos los esquemas, validando VRAM_delta como proxy medible del KV-cache acumulado.

*Test:* Análisis descriptivo de VRAM_delta por turno. Spearman si hay suficientes puntos.

---

## 4. Métricas registradas por turno

| Métrica | Fuente | Justificación |
|---|---|---|
| `energy_j` | NVML trapezoidal | Energía total del turno |
| `j_per_output_token` | energy_j / completion_tokens | Métrica principal de eficiencia |
| `ttft_ms` | Streaming (primer chunk) | Costo de prefill — crece con contexto |
| `tpot_ms` | total_time / completion_tokens | Costo de decode — estable si batch fijo |
| `throughput_tok_s` | completion_tokens / total_time_s | Rendimiento |
| `vram_peak_mb` | NVML | Proxy de memoria total (modelo + KV-cache) |
| `vram_start_mb` | NVML | Baseline por esquema (peso del modelo) |
| `vram_delta_mb` | peak − start | Proxy específico del KV-cache del turno |
| `prompt_tokens` | vLLM usage | Input tokens del turno (contexto acumulado) |
| `completion_tokens` | vLLM usage | Output tokens generados |
| `avg_power_w` | NVML | Potencia media del turno |
| `peak_power_w` | NVML | Potencia pico |
| `nvml_samples` | GPUPowerMonitor | Valida calidad del muestreo |

---

## 5. Amenazas a la validez

**Amenaza 1 — Contenido fijo de la conversación (validez externa):** El flujo de 7 turnos es específico del dominio de seguridad privada y de la estructura documental de MOSS. Los patrones energéticos pueden diferir en otras industrias o estilos conversacionales. *Mitigación:* Los documentos son sustituibles por cualquier conjunto equivalente; el framework es reproducible.

**Amenaza 2 — Historia pre-generada con FP16 (validez interna):** INT8 y AWQ procesan respuestas generadas por FP16 como contexto, no sus propias respuestas. Con temperature=0, las diferencias son mínimas para un modelo de instrucciones, pero no son cero. *Mitigación:* Declarado explícitamente como decisión metodológica auditada; se documenta el procedimiento de generación.

**Amenaza 3 — Cooling artificial entre turnos (validez ecológica):** Los 120s de cooling no reflejan conversaciones en tiempo real. *Mitigación:* El objetivo es la medición independiente por turno, no la simulación de tiempo real. Se declara como limitación.

**Amenaza 4 — max_tokens fijo (256) para todos los turnos:** Los turnos 5 y 6 (generativos) producirían respuestas más largas en producción real. La truncación a 256 puede subestimar la energía para tareas generativas. *Mitigación:* La consistencia de max_tokens entre turnos permite atribuir diferencias energéticas al contexto, no a la variación de output length. Consistente con VI3 del factorial estático.

**Amenaza 5 — Única conversación / único flujo narrativo:** No se prueba variabilidad de temas ni estilos de pregunta. *Mitigación:* Propuesto como extensión futura (múltiples flujos conversacionales con documentos variables).

**Amenaza 6 — Hardware único y motor único:** RTX 4090 + vLLM 0.5.3. Los resultados absolutos son específicos de este stack. *Mitigación:* Misma limitación que el Experimento 1; los resultados relativos entre esquemas son la contribución principal.

---

## 6. Protocolo de ejecución en RunPod

```bash
# PASO 1 — Pre-generar conversación (una sola vez, ~5 minutos)
# Asegurarse de que vLLM FP16 esté activo
bash scripts/start_vllm_fp16.sh
# Esperar a que levante (~60s), luego:
python scripts/build_multiturn_conversation.py
# Verificar: data/multiturn/conversation_history.json

# PASO 2 — FP16 (misma instancia)
python scripts/multiturn_runner.py --quantization fp16
# Duración estimada: 7 turnos × 2 batch × 3 reps × (turno_promedio ~30s + 120s cooling)
# ≈ 7 × 2 × 3 × 150s ≈ 105 min

# PASO 3 — INT8 W8A16
# Reiniciar vLLM con INT8
bash scripts/start_vllm_int8.sh
python scripts/multiturn_runner.py --quantization int8_w8a16

# PASO 4 — AWQ INT4
bash scripts/start_vllm_awq.sh
python scripts/multiturn_runner.py --quantization int4_awq

# PASO 5 — Análisis
python scripts/multiturn_analysis.py \
    --results-dir results/multiturn/fp16_* \
                  results/multiturn/int8_* \
                  results/multiturn/awq_*
```

**Duración total estimada:** ~3.5 horas de GPU activo (sin contar cooling overhead).  
**Costo estimado RunPod RTX 4090:** ~$12–15 para el experimento completo.

---

## 7. Piloto recomendado antes de la corrida completa

```bash
# Solo rep=1, batch=1, FP16 — verifica todo el pipeline en ~15 min
python scripts/multiturn_runner.py --quantization fp16 \
    --batch-sizes 1 --pilot
```

Verificar en los resultados:
- `nvml_samples` ≥ 10 por turno
- `j_per_output_token` crece de T1 a T7 (esperado)
- `vram_peak_mb` crece de T1 a T7 (esperado)
- `ttft_ms` crece de T1 a T7 (esperado)
- `status == "success"` en todos los turnos

---

## 8. Conexión narrativa con el paper

**Sección de resultados, estructura sugerida:**

```
5. Resultados
  5.1 Experimento 1: Caracterización factorial estática [ya escrita]
    5.1.1 Efecto de cuantización (VI1)
    5.1.2 Anomalía INT8 batch=4 (VI2)
    5.1.3 Impacto asimétrico del contexto (VI4 × VI1)
    5.1.4 Matrices de decisión IEC
  5.2 Experimento 2: Evolución energética en conversación multi-turno [nueva]
    5.2.1 Escalación energética por turno (H1)
    5.2.2 Persistencia de la anomalía INT8 en contexto creciente (H2)
    5.2.3 Erosión de la ventaja AWQ (H3)
    5.2.4 KV-cache como proxy de VRAM y costo de prefill (H4)
    5.2.5 Implicaciones para despliegue en asistentes empresariales
```

**Frase de transición entre experimentos:**

> "El Experimento 1 estableció que el impacto del contexto sobre la eficiencia energética es asimétrico entre esquemas de cuantización: AWQ sufre una penalidad de +122% frente al +59% de FP16 al escalar de Case_A a Case_C. El Experimento 2 extiende este hallazgo a la dimensión temporal, preguntando: ¿cómo evoluciona esa penalidad cuando el contexto crece de forma orgánica a lo largo de una conversación empresarial realista?"

---

*Fin del documento de diseño. Para preguntas de defensa relacionadas con este experimento, ver DECISION_LOG.md.*
