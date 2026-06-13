# EXP2 — Protocolo C: Sobre Conjunto Energía×Calidad y Rentabilidad Energética de la Compactación en Modelos Pequeños Auto-Hospedados

**Estado:** Diseño sellado, listo para construir.
**Hardware de medición:** RTX 4090 (pod dedicado RunPod, mismo instrumento que EXP1).
**Modelo:** Llama-3.1-8B-Instruct. Espina de cuantización: FP16 + AWQ INT4 (INT8 opcional).
**Motor:** vLLM (PagedAttention, continuous batching).

---

## 1. Declaración de contribución (lo que defiendes)

> Una caracterización empírica y reproducible del **sobre útil conjunto energía×calidad** para modelos LLM pequeños auto-hospedados, bajo una carga realista de **sesión de proyecto con contexto acumulativo** sobre hardware de clase consumo. Se identifica el **codo de acumulación de contexto** —el punto donde el costo energético marginal por petición sigue subiendo mientras la calidad de tarea empieza a degradarse— y se evalúa empíricamente si la **compactación proactiva de contexto es energéticamente y cualitativamente rentable** en este hardware. El resultado es una **regla de decisión para el practicante** y un **monitor de referencia** que la operacionaliza.

### Por qué sobrevive a la literatura (delimitación explícita para Trabajos Relacionados)

| Trabajo cercano | Qué hace | Qué NO hace = tu rendija |
|---|---|---|
| **The Efficiency Frontier** (Shen et al., 2605.23071) | Costo×calidad para estrategias de contexto; costo = **tokens**, calidad = F1, sobre HotpotQA | Joules medidos; hardware de consumo; sesión real mixta; el impuesto de compactación en energía |
| **Context rot** (Chroma 2025, Du et al. 2025, NoLiMa) | Degradación de calidad vs longitud de contexto | En modelos frontera vía API; **sin energía** |
| **Ley 1/W** (2603.17280) | Energía/tok vs contexto (cae ~40× de 2K→128K) | **Sin calidad**; datacenter analítico |
| **SUPO** (2510.06727) | RL para que el agente resuma su historia | Optimiza éxito de tarea, **no joules**; no pregunta si resumir ahorra energía neta |
| **qMeter / Shi & Ding** (2508.16712) | Energía×calidad×cuantización | Datacenter A100/H100; saturado; **no acumulación conversacional ni compactación** |

**El foso:** las dos curvas (energía y calidad) sobre el **mismo eje de acumulación de contexto**, en un **modelo pequeño auto-hospedado**, con la **prueba de recuperación por compactación medida en joules Y en éxito de tarea**, sobre un **caso real (MOSS)**. Token-proxy ≠ Joule-medido por la atención cuadrática: 2000 tokens al final de la ventana no cuestan la misma energía que 2000 al inicio. Ese es el valor central.

---

## 2. Escenario y persona

**Persona:** organización pequeña (empresa de seguridad privada tipo MOSS) que auto-hospeda Llama-3.1-8B localmente para escapar del rate-limiting, el costo impredecible y por soberanía de datos. No hace consultas puntuales: trabaja como en un **proyecto de ChatGPT/Claude** — carga conocimiento del negocio una vez y luego ejecuta tareas reales durante un período de trabajo (p. ej., un mes operativo), acumulando historial.

**Conocimiento de proyecto (KB persistente, inyectado al inicio de la sesión):** reglas de turnos, nómina de guardias, contratos de clientes, formatos de reporte de incidentes, políticas de descanso. Es el "system context" del proyecto.

**Sesión = secuencia de peticiones heterogéneas reales** que acumulan contexto turno a turno:
1. Resumir un reporte de incidente.
2. Redactar un memo para un cliente.
3. Proponer el rol de turnos del mes (con restricciones duras).
4. Clasificar una excepción de asistencia.
5. … (la secuencia completa la defines tú; ver §12).

Cada petición + su respuesta se **anexan** al contexto, como en una sesión real. El contexto crece monótonamente hasta acercarse a la ventana del modelo (8K en Llama-3.1 base; documentar la ventana efectiva usada).

---

## 3. Variables

**IV principal — contexto acumulado.** Operacionalizada como el avance de la sesión: el índice de petición *i* y los tokens acumulados de contexto en el momento de servir esa petición. Se mide en cada paso.

**IV secundaria — cuantización.** FP16 vs AWQ INT4 (INT8 opcional). Permite ver si el codo se mueve con la precisión.

**Variables dependientes:**
- **Energía:** energía por petición (J), J/token, J por respuesta útil, vía NVML (reusar `gpu_power_monitor.py`). Más TPOT, TTFT, potencia media/pico, VRAM pico.
- **Calidad / éxito de tarea (DV NUEVA):** ver §4. Es el eje que nadie midió junto con energía sobre acumulación.

**Controladas:** mismo modelo, misma temperatura/seed de decodificación, mismo KB inicial, mismo orden de tareas (con repeticiones), mismo hardware, mismo motor.

---

## 4. Métrica de calidad (la decisión crítica — diseñada para ser defendible)

Regla de oro: **diseñar las tareas para que la corrección sea verificable programáticamente siempre que se pueda.** Esto mata el ataque "BLEU es malo / la calidad es subjetiva".

**Nivel 1 — Verificable por código (preferido):**
- *Rol de turnos del mes:* satisfacción de restricciones duras → ningún guardia doble-asignado, cobertura completa de turnos, respeto de horas de descanso, cuadre de personal. Es un puntaje objetivo de constraint-satisfaction (0–1 o % de restricciones cumplidas).
- *Clasificación de excepciones:* exactitud contra etiqueta conocida.
- *Campos requeridos en documentos/memos:* presencia y consistencia de campos obligatorios (fecha, cliente, firmante, etc.) verificada por reglas.

**Nivel 2 — Consistencia factual con el KB (semi-verificable):**
- Detección de contradicciones con el KB (¿inventó un guardia que no existe? ¿un cliente que no está en cartera?) — chequeos de exact-match / entidades contra el KB.

**Nivel 3 — Juez con rúbrica (para lo abierto, p. ej. calidad del memo):**
- LLM-as-judge con un modelo fuerte (vía API), rúbrica fija y explícita.
- **Validación obligatoria:** etiquetar a mano una muestra (p. ej. 30 respuestas) y reportar acuerdo juez-humano (Cohen's κ o correlación). Si el acuerdo es alto, el juez es defendible; si no, te quedas con Niveles 1–2.

**Puntaje de calidad por petición** = combinación documentada de los niveles aplicables a esa tarea. Reportar por tipo de tarea, no solo agregado (las tareas complejas se pudren antes — Chroma).

---

## 5. Brazos experimentales

- **A — Acumulación naive (baseline):** dejar crecer el contexto sin compactar. Es lo que hace el usuario ingenuo. Genera las dos curvas crudas → localiza el codo.
- **B — Compactación en el codo:** al cruzar el umbral medido, compactar (§6), iniciar sesión "fresca" con KB + handoff, continuar las tareas restantes.
- **C — Sensibilidad del umbral:** compactar demasiado temprano y demasiado tarde, para demostrar que el punto importa (no es trivial truncar en cualquier *n*).
- **D — Contraste RAG (opcional, para discusión):** en vez de cargar todo el historial, recuperar solo los fragmentos relevantes del KB por petición. Delimita contra RAG y muestra la diferencia (RAG gestiona *qué recuperar*; tú estudias *acumulación de historial*).

Repeticiones: 3 por brazo × condición (protocolo MELODI). Orden de tareas con seed fijo; variantes de orden si el tiempo lo permite (para descartar que el efecto sea del orden y no de la acumulación).

---

## 6. La compactación y el "impuesto de compactación"

**Operación de compactación (el handoff):** resumir/destilar el historial acumulado en un handoff compacto que preserve lo esencial para las tareas restantes (decisiones tomadas, datos clave, estado). Se mide explícitamente:

- **Impuesto de compactación:** los Joules (y tokens, y tiempo) que cuesta *generar* el resumen. Resumir no es gratis — es una llamada de inferencia más.
- **Energía post-compactación:** J/petición después del handoff (debería caer al régimen de contexto corto).
- **Calidad post-compactación:** ¿el éxito de tarea se recupera, o el handoff con pérdidas lo daña?
- **Balance neto:** ¿la compactación se paga sola en Joules a lo largo de las peticiones restantes, manteniendo o restaurando la calidad? Esto responde directamente la advertencia de JetBrains ("los resúmenes a veces alargan la conversación") y el gap "¿es el context-folding energéticamente rentable en un 8B local o es un impuesto demasiado caro?".

---

## 7. Análisis y figura insignia

**Figura central:** doble eje Y sobre el eje X de contexto acumulado (tokens o índice de petición):
- Eje izquierdo: J/petición (o J por respuesta útil) — sube (1/W + cuadrática).
- Eje derecho: puntaje de calidad — baja (context rot).
- **El codo / zona de cruce:** el rango donde pagas *más* joules para obtener *peor* calidad. Ese es el hallazgo memorable.

**Figura de recuperación:** trayectoria de J/petición y calidad en el brazo B vs A, marcando el impuesto de compactación y el punto de break-even energético post-handoff.

**Regla de decisión (entregable práctico):** "para un modelo pequeño X en hardware Y, compacta al alcanzar ~Z tokens acumulados; el handoff cuesta ~W J pero recupera ~Q de calidad y se paga en ~K peticiones."

---

## 8. Hardware: dónde se mide vs dónde despliega el practicante

- **Medición (núcleo):** RTX 4090, pod dedicado RunPod. Mismo instrumento que EXP1 → comparabilidad directa. NVML completo.
- **Robustez (opcional, barato):** una GPU más accesible en RunPod (RTX 3090 o A4000) para mostrar que la *forma* del sobre se sostiene en hardware más commodity.
- **Mac:** NO para el experimento medido (vLLM es CUDA; Mac usaría llama.cpp/MLX + `powermetrics`, otro instrumento → rompe comparabilidad). Trabajo futuro explícito.
- **Serverless (RunPod Serverless/Modal/Vast):** NO es plataforma de medición — sin NVML sobre GPU dedicada no mides energía, solo costo/latencia/cold-start. Es el **comparador económico de la discusión**: el sobre (física) es independiente del modelo de facturación; se mide una vez en el pod dedicado y se *razona* qué implica bajo dedicado (pagas idle) vs serverless (pagas por segundo, sin idle, pero con cold-start) vs API (energía ajena, estimada vía "How Hungry is AI?").

---

## 9. Reproducibilidad (reusar EXP1)

- NVML: BUFFER_MS=500, SAMPLING_MS=100, integración trapezoidal, VRAM pico.
- Warmup: 5 peticiones descartadas antes de medir.
- Cooling: 2 min entre condiciones.
- Seed=42 para orden y decodificación.
- AutoTokenizer de HuggingFace (hard-fail si no disponible).
- Ediciones de archivos en `/workspace` con Python `open().read()/write()`, nunca `sed -i`.
- `snapshot_download` con `local_dir_use_symlinks=False` y cache HF en volumen de red.

---

## 10. Artefacto: monitor de referencia (el demostrador, NO la tesis)

Una librería ligera (`infera_monitor`) que, dados los parámetros del sobre medido, en una sesión viva: rastrea el contexto acumulado, estima J/petición y calidad esperada según las curvas, marca el codo, y al cruzarlo recomienda compactar y produce el handoff. Es la prueba de concepto que vuelve tangible el hallazgo. Se presenta como ingeniería de soporte, no como contribución científica central.

---

## 11. Qué se reusa de EXP1 y qué se rehace

**Se reusa tal cual:** `gpu_power_monitor.py`, metodología MELODI, infraestructura vLLM, KB y conversación MOSS (`conversation_flow.json`) como semilla del workload.

**Se reusa como ancla:** la curva energía-vs-contexto corto de EXP1 (FP16/AWQ) = extremo izquierdo del sobre.

**Se rehace/extiende:** el eje de contexto se extiende mucho más allá de 4096 (acumulación realista); `multiturn_runner.py` y `multiturn_analysis.py` reciben la DV de calidad y el brazo de compactación; el workload pasa de chat a sesión de proyecto con tareas heterogéneas.

---

## 12. Plan de 7 días

- **Día 1:** congelar el set de tareas MOSS (§2) y el esquema de puntaje de calidad por tarea (§4). Construir el KB y la secuencia de peticiones. *(Requiere tu input: qué tareas reales hace la empresa.)*
- **Día 2:** instrumentar el runner — sesión acumulativa, captura por petición de energía + tokens + salida; integrar puntaje de calidad Nivel 1–2 (verificable por código).
- **Día 3:** correr brazo A (acumulación naive) en FP16 + AWQ, 3 reps. Generar las dos curvas crudas. Localizar el codo.
- **Día 4:** implementar y correr brazo B (compactación en el codo) + medir el impuesto de compactación. Brazo C (sensibilidad) si hay tiempo.
- **Día 5:** juez LLM con rúbrica para tareas abiertas + validación de acuerdo en muestra. Robustez en GPU barata (opcional).
- **Día 6:** análisis, figura insignia, figura de recuperación, regla de decisión, capa económica serverless/API (discusión).
- **Día 7:** escritura de resultados/discusión + sección de amenazas y defensa.

---

## 13. Amenazas anticipadas (preparación de defensa)

- *"Esto ya lo hizo The Efficiency Frontier."* → Ellos usan tokens como proxy y QA sintético; yo mido Joules en hardware real donde el costo por token no es constante por la atención cuadrática, sobre una sesión empresarial real, e incluyo el impuesto de compactación en energía. Cita y delimita.
- *"Truncar cada n tokens es trivial y ahorra energía igual."* → Por eso el brazo C demuestra que el punto importa, y la contribución no es truncar sino localizar el codo conjunto y medir si la compactación recupera calidad además de energía.
- *"La calidad es subjetiva."* → Niveles 1–2 son verificables por código (constraint-satisfaction, consistencia con KB); el juez del Nivel 3 está validado contra humano.
- *"N=1 hardware / N=1 caso."* → Ancla 4090 + robustez en GPU barata; caso real documentado; artefacto reproducible para que otros extiendan.
- *"El individuo tiene huella insignificante."* → El argumento ambiental es agregado y de responsabilidad; la decisión del individuo es por rate-limit/costo/soberanía. El hallazgo de que el auto-hosting puede ser MENOS verde por petición es honesto y valioso.
- *Fine-tuning está fuera de alcance — no mencionarlo como solución.*
