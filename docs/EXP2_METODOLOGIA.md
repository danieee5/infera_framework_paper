# EXP2 — Metodología, alcance y plan de ejecución (INFERA)

Documento interno (para `DECISION_LOG.md` y secciones de método / amenazas a la validez del paper).
No es texto final del paper; es la justificación defendible de cada decisión.

---

## 1. Qué mide EXP2 (y qué NO mide)

EXP2 mide el **escalamiento del costo energético de la inferencia conforme el contexto
se acumula turno a turno** en una conversación empresarial realista, y verifica si las
anomalías del factorial estático (EXP1) reaparecen o se amplifican en régimen temporal.

- **Variable que en EXP1 era control estático (contexto) aquí es una cantidad emergente
  y creciente.** EXP1 respondía "si el contexto vale X, ¿cuánta energía?". EXP2 responde
  "conforme una conversación real acumula contexto, ¿cómo evoluciona la energía por turno?".
- **No mide degradación de calidad.** La calidad es un fenómeno distinto del consumo
  energético. EXP2 es un experimento de energía/rendimiento.

### Métrica primaria
`j_per_output_token` por turno = `energy_j` / `completion_tokens`. Como la salida está
acotada (256 tokens, control), el crecimiento de esta métrica con el turno refleja el
costo del prefill del contexto acumulado.

### Hipótesis
- **H1** J/output_token crece monótonamente con el turno (presión de prefill / KV-cache).
- **H2** la anomalía INT8 batch=4 (J/tok_b4 > J/tok_b1) se amplifica en turnos tardíos.
- **H3** la ventaja de AWQ sobre FP16 se erosiona con contexto largo (coherente con el
  +122% AWQ vs +59% FP16 del factorial estático).
- **H4** VRAM peak crece monótonamente (confirma acumulación de KV-cache).

---

## 2. Declaración explícita de alcance de la medición

> EXP2 mide **energía total por turno** (prefill + decode) en una sola ventana NVML, con
> el protocolo de buffer MELODI de 500 ms. **No se segmenta** la energía en prefill vs
> decode. Se reportan **TTFT como proxy temporal del prefill** y **TPOT como proxy temporal
> del decode**; el ratio TTFT/TPOT por turno cuantifica la creciente dominancia del prefill.
>
> El régimen medido es de **re-prefill completo**: los servidores vLLM se levantan **sin
> `--enable-prefix-caching`**, por lo que cada turno reprocesa todo el contexto acumulado
> desde cero. Esta es una decisión deliberada: aísla el costo energético del contexto
> creciente. El **prefix caching** —optimización que usaría un despliegue de producción—
> queda **fuera de alcance** y se declara como **amenaza a la validez externa y trabajo
> futuro** (mediría una pregunta distinta: el ahorro de KV-cache reutilizado).
>
> `max_tokens = 256` se fija como **control** para mantener el decode aproximadamente
> constante entre turnos, de modo que el crecimiento de energía por turno sea atribuible
> al prefill creciente y no a una salida más larga.

**Defensa anticipada (jurado):** "En producción usarías prefix caching, entonces tus
números sobreestiman el costo." → Correcto, y por eso se declara explícitamente: EXP2
caracteriza el costo del *procesamiento de contexto*, no el costo de *servicio optimizado*.
Son dos preguntas separadas; medir el régimen sin caché es lo que permite atribuir la
energía al contexto.

---

## 3. Diseño de la conversación (historia fija / pinned)

- **Escenario único, realista:** caso disciplinario-operativo MOSS (guardia nocturno,
  cliente Acciona, contrato por giro de obra, que no reportó una novedad y abandonó el
  puesto 20 min). 7 turnos: clasificar falta → memorando → adaptar a nocturno/Acciona/giro
  de obra → llamado vs suspensión → descargo → tabla F-01 vs F-02 → recomendación final.
- **Corpus base (~1.9k tokens):** 7 documentos internos (reglamento, clasificación de
  faltas, procedimiento disciplinario, política de turnos, contratos giro de obra/Acciona,
  reporte de novedades F-01, protocolo de emergencia F-02). **Ilustrativos y sustituibles:**
  otra organización reemplaza el bloque manteniendo el rango de tokens, preservando la
  comparabilidad. El detalle de dominio (guardia por obra vs operador indefinido, cliente
  Acciona) da realismo pero no define el dominio.
- **Historia fija:** con `temperature=0` (greedy), las respuestas se pre-generan **una sola
  vez con FP16** y se congelan (`build_multiturn_conversation.py`, Fase 1). Los tres esquemas
  (FP16/INT8/AWQ) ven el **mismo contexto idéntico** en cada turno → las diferencias de
  energía son atribuibles al esquema, no a respuestas divergentes. **La historia debe ser
  idéntica en todos los pods.**

**Importante (matiz de validez):** el realismo del corpus mejora la **representatividad
(validez externa/ecológica)**, NO la exactitud de la medición. El medidor mide vatios; lo
que físicamente mueve los números es la trayectoria de tokens (input/output por turno) más
la configuración. No afirmar que "el caso real hace la medición más precisa".

### Envolvente de tokens (puente Case_B → Case_C)
- Objetivo: T1 ∈ [1500, 2000] tok reales, T7 ∈ [3500, 4500] tok reales.
- Estimado de diseño: T1 ≈ 1.9k, T7 ≈ 3.7k, máximo en T7 + 256 salida ≈ 4.06k « 8192.
- **La verdad la da Fase 1** con `usage.prompt_tokens` de vLLM (tokenizer real del modelo),
  que actualiza `measured_input_tokens` y valida la banda. Si algún turno cae fuera, el
  script imprime exactamente cuántos tokens añadir/quitar y dónde.

---

## 4. Plan de ejecución (≤ 4 h)

### Costo por esquema (matriz actual: batch [1,4] × 7 turnos × 3 reps, cooling 120 s)
≈ 84 min de cooling + ~10 min de inferencia/idle ≈ **~1.6 h por esquema/pod**.

### Opción A — RECOMENDADA: 3 pods en paralelo (un esquema por pod)
- Wall-clock ≈ **~1.7 h** (todos terminan a la vez). Holgado bajo 4 h, y **conserva el
  cooling conservador de 120 s** (no se sacrifica rigor térmico).
- **Requisito:** los 3 pods deben tener el **mismo modelo de GPU** (RTX 4090) y usar la
  **misma `conversation_history.json`** (generada una vez en Fase 1 y distribuida).
- **Amenaza:** varianza tarjeta-a-tarjeta. Impacto acotado:
  - H1, H2, H4 son **intra-tarjeta** (cada una compara turnos/batch dentro del mismo pod) →
    **no afectadas**.
  - H3 compara AWQ (pod 3) vs FP16 (pod 1) → **única expuesta**. Un offset constante de
    potencia entre tarjetas desplaza el valor absoluto del ratio pero **preserva la
    tendencia** (la erosión a lo largo de los turnos). 
  - **Mitigación:** `setup_exp2.sh` captura 60 s de potencia en reposo por pod
    (`idle_power_<host>.json`); se reporta el offset entre tarjetas. Si el offset en reposo
    es < ~2%, el confound de H3 es despreciable.

### Opción B — fallback: 1 pod secuencial (3 esquemas)
- 3 × 1.6 h + reinicios ≈ **~5 h → excede 4 h**.
- Para entrar en 4 h habría que **reducir el cooling inter-turno de 120 s a ~60 s**, lo que
  exige justificar la estabilidad térmica (caracterizar el retorno a temperatura de reposo).
  Menos limpio que la Opción A. Conserva una sola tarjeta (sin confound entre tarjetas).

**Decisión:** Opción A (paralelo, 3 pods 4090 idénticos, historia compartida, calibración de
reposo). Cumple ≤ 4 h con margen y solo expone H3 a un confound declarado y mitigado.

---

## 5. Amenazas a la validez (añadir al paper)

1. **Prefix caching fuera de alcance** (ver §2). Régimen de re-prefill completo; un
   despliegue real con caché consumiría menos. Trabajo futuro: medir ambos regímenes.
2. **Escenario único (N=1 conversación).** Los hallazgos podrían depender de la trayectoria
   de tokens de este caso. Mitigación parcial: el caso es representativo de RAG empresarial
   y el corpus es sustituible. Trabajo futuro: replicar con escenarios de distinto perfil de
   carga (generativo-pesado vs recuperación-pesada).
3. **Historia fija generada con FP16.** Los esquemas cuantizados se evalúan sobre un contexto
   generado por FP16, no autogenerado. Es un control deliberado (aísla el efecto del esquema);
   el costo es que no se mide la deriva conversacional propia de cada esquema.
4. **Varianza tarjeta-a-tarjeta** (solo si se usa la Opción A; afecta solo H3). Mitigada con
   calibración de potencia en reposo (ver §4).
5. **TTFT/TPOT como proxies temporales** del prefill/decode, no medición energética
   segmentada. La separación energética prefill/decode queda como trabajo futuro.

---

## 6. Archivos del repositorio (EXP2)

| Archivo | Ubicación sugerida | Rol |
|---|---|---|
| `conversation_flow.json` | `data/multiturn/` | Spec: documentos + 7 turnos de usuario + metadatos. Entrada de Fase 1. |
| `build_multiturn_conversation.py` | `scripts/` | Fase 1: genera historia fija con FP16 + valida tokens reales. |
| `multiturn_runner.py` | `scripts/` | Fase 2: mide energía por turno (parcheado: registra texto + declaración de alcance). |
| `multiturn_analysis.py` | `scripts/` | Análisis: H1–H4, VRAM, TTFT/TPOT, puente al factorial. (Sin cambios.) |
| `setup_exp2.sh` | `scripts/` | Addendum de setup: data dirs + calibración de reposo + secuencia. |
| `EXP2_METODOLOGIA.md` | raíz / `docs/` | Este documento. |

`setup_runpod_v3.sh` **no cambia**: el entorno y los modelos de EXP2 son los mismos de EXP1.
