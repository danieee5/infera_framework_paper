# INFERA: Un Marco Reproducible para Evaluar el Sobre Conjunto Energía–Calidad y la Rentabilidad Energética de la Compactación de Contexto en Modelos de Lenguaje Pequeños Auto-Hospedados

*(Documento preliminar — hasta Metodología. Resultados, Discusión y Conclusiones quedan como marcadores de posición.)*

---

## Resumen

El abandono progresivo de las APIs comerciales de modelos de lenguaje hacia el auto-hospedaje de modelos abiertos está motivado por límites de tasa, costos impredecibles y soberanía de datos, y no por sostenibilidad. Sin embargo, la literatura que cuantifica el costo de la inferencia se ha concentrado en hardware de centro de datos y en escenarios saturados, dejando sin caracterizar el régimen que realmente vive el practicante individual, el estudiante o la pequeña empresa: un único equipo, peticiones servidas de a una, y un contexto conversacional que crece a lo largo de una sesión de trabajo. En ese régimen, alargar el contexto cobra un doble precio que nadie ha medido conjuntamente sobre hardware accesible: la energía por token crece —por el costo de la atención y de la caché de claves-valores— mientras la calidad de la respuesta se degrada por el fenómeno conocido como *context rot*. Este trabajo propone INFERA, un marco reproducible que mide ambas curvas sobre el mismo eje de contexto acumulado en un modelo Llama 3.1 8B servido localmente con vLLM, localiza el punto de inflexión en el que el usuario paga más energía por peor calidad, y evalúa empíricamente si la compactación proactiva del contexto recupera tanto eficiencia como calidad, contabilizando el costo energético de la propia compactación. El caso de estudio es una empresa de seguridad privada ecuatoriana, con tareas operativas reales (clasificación disciplinaria, redacción de memorandos, asignación de turnos con restricciones, consulta de registros), lo que dota al experimento de validez ecológica. La contribución central no es una herramienta de compactación —existen varias— sino la caracterización medida del sobre energía–calidad y la verificación del balance energético de la compactación en el hardware del usuario final.

---

## 1. Introducción

La adopción de modelos de lenguaje de gran escala se ha vuelto cotidiana para una población que excede ampliamente a las grandes corporaciones: estudiantes, desarrolladores independientes, personas que transitan desde la programación asistida hacia una comprensión más profunda de la ingeniería de sistemas, y pequeñas empresas con presupuestos ajustados. Para esta población, el modelo de suscripción mensual y, sobre todo, los límites de uso por ventana de tiempo impuestos por los proveedores, se han convertido en una fricción concreta: el servicio se interrumpe, encarece o restringe justo cuando se lo necesita. La respuesta natural ha sido el auto-hospedaje de modelos abiertos, que promete independencia del proveedor, costo predecible y control sobre los datos.

Esta migración rara vez se justifica por motivos ambientales, pero el costo energético y de agua de la inferencia es un problema real y creciente que el usuario que se auto-hospeda asume directamente, a diferencia del usuario de API, para quien ese costo es invisible y diferido. Surge entonces una pregunta de responsabilidad: ¿cómo puede un practicante auto-hospedar de forma consciente, sabiendo cuánta energía consume y bajo qué condiciones ese consumo deja de ser razonable?

El cuerpo de trabajo que mide el costo de la inferencia ha respondido en su mayoría desde la perspectiva del operador de centro de datos y bajo condiciones de saturación, donde la eficiencia es máxima. Ese encuadre no describe el régimen del practicante individual, que sirve peticiones de a una sobre un único equipo cuya GPU permanece ociosa la mayor parte del tiempo, y que trabaja en sesiones donde el contexto se acumula —como en un proyecto de ChatGPT o Claude— a medida que encadena tareas. Es precisamente en la acumulación de contexto donde aparece un fenómeno doblemente costoso para este usuario y que la literatura ha estudiado de forma fragmentada: por un lado, la energía por token producido crece de manera no lineal con la longitud del contexto, debido al costo cuadrático de la atención y al crecimiento de la caché de claves-valores; por otro, la calidad de las respuestas se degrada mucho antes de agotar la ventana de contexto nominal, un efecto documentado de forma consistente en modelos frontera. El usuario que se auto-hospeda paga ambos precios a la vez —más joules y peor calidad— sin que ningún trabajo previo haya trazado ese sobre conjunto sobre su hardware.

Este trabajo aborda ese vacío. Propone medir, sobre un mismo eje de contexto acumulado y en un modelo pequeño servido localmente, la curva de energía por tarea y la curva de calidad de tarea, para localizar el punto de inflexión —el "codo"— a partir del cual continuar acumulando contexto significa pagar más energía a cambio de menor calidad. Sobre esa base, evalúa una intervención concreta y disponible para cualquier usuario: la compactación del contexto, es decir, resumir la sesión y reiniciarla con un traspaso ("handoff") compacto. La pregunta no es si la compactación es posible —los marcos de agentes ya la implementan— sino si es **energéticamente rentable** en hardware de consumo para un modelo de 8B, dado que la propia operación de resumir consume energía y que un resumen con pérdidas puede dañar la calidad posterior.

## 2. Planteamiento del problema y pregunta de investigación

**Problema.** No existe una caracterización medida y reproducible, sobre hardware accesible, del costo conjunto energético y de calidad de la acumulación de contexto en modelos pequeños auto-hospedados, ni una evaluación del balance energético de la compactación como mitigación.

**Pregunta de investigación.** Para un modelo de lenguaje pequeño auto-hospedado sobre hardware de clase consumo, bajo una carga realista de sesión de proyecto con contexto acumulativo: ¿en qué punto de acumulación de contexto la energía marginal por tarea crece mientras la calidad de tarea se degrada, y la compactación proactiva del contexto recupera eficiencia y calidad una vez descontado su propio costo energético?

## 3. Objetivos

**General.** Caracterizar de manera reproducible el sobre conjunto energía–calidad de la inferencia con contexto creciente en un modelo pequeño auto-hospedado, y evaluar la rentabilidad energética de la compactación de contexto.

**Específicos.**
1. Medir la energía por tarea (mediante NVML, en joules) y la calidad de tarea (mediante verificación programática) a lo largo de una sesión de contexto creciente, para FP16 y AWQ INT4.
2. Localizar el punto de inflexión conjunto ("codo") donde la energía marginal aumenta y la calidad disminuye.
3. Cuantificar el "impuesto de compactación": el costo energético de generar el resumen de traspaso.
4. Comparar el brazo de acumulación naive contra el brazo de compactación en energía total, calidad media y punto de equilibrio.
5. Derivar una regla de decisión accionable para el practicante y publicar un marco reproducible.

## 4. Trabajos relacionados y delimitación

La degradación de calidad por longitud de contexto está bien documentada: estudios recientes sobre modelos frontera muestran que la precisión cae mucho antes de llenar la ventana nominal, y que la longitud de entrada es una causa de primer orden de la degradación, independiente de la recuperación de información (Chroma, 2025; Du et al., 2025; estudios sobre ventana efectiva). Sin embargo, estos trabajos se realizan sobre modelos accedidos por API y **no miden energía**.

Del lado energético, la dependencia de la eficiencia respecto del contexto es conocida: la métrica de tokens por vatio puede variar en casi un orden y medio de magnitud a lo largo del rango de contexto (estudios de la "ley 1/W", 2026), pero esos análisis **no incorporan calidad** y se sitúan en hardware de centro de datos. El trabajo más cercano al presente, *The Efficiency Frontier* (Shen et al., 2026), une costo y rendimiento para decidir estrategias de gestión de contexto, pero emplea **tokens como aproximación del costo** y F1 sobre un conjunto de preguntas sintéticas; al usar tokens como proxy, no captura que en hardware local la energía por token no es constante a lo largo de la ventana. La caracterización conjunta de energía y calidad para cuantización (Shi & Ding, 2025) y la medición de potencia de servicio (TokenPowerBench, 2025) permanecen en el régimen saturado de centro de datos. Respecto de la compactación, trabajos como SUPO (2025) optimizan el resumen del historial para preservar el éxito de la tarea del agente, pero **no analizan si resumir ahorra energía neta**; los marcos de agentes en producción compactan por límite de ventana, no por un criterio energético medido.

La delimitación de INFERA es, por tanto, la intersección que ningún trabajo único ocupa: medición de **energía real** (no proxy de tokens) sobre **hardware de consumo**, bajo una **sesión empresarial real** con contexto acumulativo, con la **prueba del balance energético de la compactación** sobre un modelo pequeño. El protocolo de medición hereda de MELODI (Husom et al., 2026) y se valida en un experimento piloto previo (EXP1) descrito en la sección de metodología.

## 5. Hipótesis

- **H1.** Existe un punto de acumulación de contexto a partir del cual la energía por tarea crece mientras la calidad de tarea decrece (sobre conjunto con codo identificable).
- **H2.** La compactación proactiva reduce la energía por tarea posterior y recupera calidad, pero su rentabilidad neta depende del costo de la operación de resumen y del punto en que se aplica.
- **H3.** El codo y la rentabilidad de la compactación dependen del esquema de cuantización (FP16 vs AWQ INT4).

## 6. Metodología

### 6.1 Enfoque y diseño
Estudio cuantitativo de enfoque cuasi-experimental y medición instrumental. La variable independiente principal es el **contexto acumulado**, operacionalizada como el avance de una sesión incremental (índice de tarea y tokens de prompt efectivamente procesados). Las variables independientes secundarias son el **esquema de cuantización** (FP16, AWQ INT4; INT8 opcional) y el **brazo experimental** (acumulación naive vs compactación).

### 6.2 Caso de estudio y carga de trabajo
El caso es una empresa de seguridad privada ecuatoriana (anonimizada como VIGÍA Seguridad S.A., datos ficticios por protección de datos). El **conocimiento de proyecto** —contexto general, perfiles de personal, reglamento interno disciplinario, registros de permisos médicos e inventario de uniformes— se inyecta como contexto fijo idéntico en toda sesión, emulando un proyecto de asistente con conocimiento cargado.

La **sesión** es una conversación incremental única: una secuencia ordenada de tareas heterogéneas reales del negocio —consulta de hechos sobre el reglamento y los registros, clasificación de faltas disciplinarias con su base legal, redacción de memorandos formales con campos obligatorios, asignación de turnos sujeta a restricciones duras, y resumen de novedades— donde cada par petición–respuesta se anexa al historial, haciendo crecer el contexto. Se incluyen tareas de **recuperación de información temprana** (RECALL) deliberadamente ubicadas en fases avanzadas de la sesión, que funcionan como detectores de *context rot*: miden si el modelo conserva datos introducidos al inicio bajo carga creciente. Cada sesión es **aislada**: no existe memoria entre sesiones ni consultas cruzadas; la única información compartida es el conocimiento de proyecto.

### 6.3 Unidad experimental
Una sesión por combinación (cuantización × brazo × repetición), con tres repeticiones por combinación. Antes de medir se ejecutan cinco peticiones de calentamiento descartadas, y entre corridas se aplican dos minutos de enfriamiento, replicando el protocolo del piloto EXP1.

### 6.4 Variables dependientes y su medición

**Energía.** Se mide la energía por petición en joules mediante NVML, con un monitor que muestrea la potencia de la GPU cada 100 ms (10 Hz) y aplica un búfer de 500 ms antes y después de la llamada de inferencia, integrando el perfil potencia–tiempo por el método trapezoidal. El búfer de 500 ms es el mínimo que garantiza una tasa de captura completa del 100 %, según la validación experimental de MELODI (Husom et al., 2026). Se registran además potencia media y de pico, duración, VRAM de pico, y los tokens de prompt y de completado reportados por el servidor (medida directa del contexto efectivamente procesado).

**Calidad de tarea.** Variable dependiente nueva respecto del piloto. Se prioriza la verificación **programática y objetiva**: presencia de elementos requeridos (artículos del reglamento, porcentajes de sanción), presencia de al menos una variante por grupo semántico, ausencia de entidades prohibidas (detección de alucinación), presencia de campos obligatorios en los memorandos, y satisfacción de restricciones duras en la asignación de turnos (cobertura completa, no doble asignación en un mismo día, descanso mínimo entre turnos). Para las tareas abiertas (resúmenes) se contempla un juez basado en modelo con rúbrica fija, **opcional y desactivado por defecto**, cuyo uso exige validar el acuerdo con anotación humana sobre una muestra. El puntaje por tarea se normaliza a [0,1] combinando los sub-criterios aplicables, con la detección de alucinación operando como penalización multiplicativa.

### 6.5 Brazos experimentales y compactación
En el brazo **naive**, el contexto crece sin intervención hasta completar la secuencia de tareas. En el brazo **compactación**, al cruzar un umbral de contexto acumulado, se solicita al mismo modelo un traspaso estructurado (decisiones y documentos generados, hechos clave consultados, pendientes), se reinicia el contexto a conocimiento de proyecto más traspaso, y se continúan las tareas restantes. La energía de esa llamada de resumen se mide con el mismo protocolo y constituye el **impuesto de compactación**. El umbral se calibra a partir del codo identificado en el brazo naive; un brazo de sensibilidad (compactar antes y después del codo) permite mostrar que el punto de intervención no es trivial.

### 6.6 Hardware y entorno
La medición se realiza sobre una RTX 4090 (24 GB) en un pod dedicado, mismo instrumento que el piloto EXP1, lo que garantiza comparabilidad. El modelo se sirve con vLLM (PagedAttention, batching continuo) mediante su interfaz compatible con OpenAI; la plantilla de chat de Llama 3.1 se aplica internamente. Opcionalmente se replica el experimento en una GPU de menor costo para verificar que la forma del sobre se conserva. La medición de energía requiere acceso directo a NVML sobre una GPU controlada; en consecuencia, las plataformas serverless —donde la GPU está abstraída— no son entornos de medición, sino objeto del análisis económico de despliegue en la discusión: el sobre es una propiedad física independiente del modelo de facturación y se mide una sola vez en el equipo dedicado.

### 6.7 Análisis
Para cada cuantización (brazo naive) se grafican, sobre el eje de contexto acumulado, la energía por tarea y la calidad de tarea promediadas entre repeticiones, y se detecta el codo mediante un criterio transparente: la primera tarea en que la calidad cae por debajo de una fracción de la calidad basal de la sesión temprana y la energía por token de salida supera su mediana. Para la recuperación se comparan, entre brazos, la energía total por sesión, la calidad media, el impuesto de compactación y el punto de equilibrio energético tras el traspaso. La reproducibilidad se garantiza fijando semilla, temperatura cero, versiones ancladas de software y publicando el código y los datos de configuración.

### 6.8 Validación previa (piloto EXP1)
El instrumento de medición y la línea base energía–contexto fueron validados en un experimento factorial previo (243 corridas, tres cuantizaciones, sobre la misma RTX 4090), que confirmó el funcionamiento del protocolo NVML/MELODI y estableció el comportamiento de la energía por token frente a la longitud de contexto en contextos cortos. EXP1 funciona como piloto metodológico y ancla de contexto corto del sobre caracterizado aquí; el presente estudio extiende el eje de contexto hacia la acumulación realista de una sesión e incorpora la dimensión de calidad y la intervención de compactación.

### 6.9 Amenazas a la validez
La cercanía de *The Efficiency Frontier* exige delimitar explícitamente el uso de joules medidos frente a tokens-proxy y el régimen de hardware de consumo. La objeción de trivialidad (truncar reduce energía simplemente por procesar menos) se atiende mostrando que la contribución es localizar el codo conjunto y verificar la recuperación de calidad, no truncar. La subjetividad de la calidad se mitiga con verificación programática en la mayoría de las tareas y validación humana del juez en el resto. La dependencia de un único hardware y un único caso se atenúa con la réplica en GPU económica y con la publicación de un marco reproducible que permite a terceros extender el estudio.

---

## Referencias (preliminares)

- Husom et al. (2026). *MELODI*: protocolo de medición energética de inferencia LLM.
- Shen, B., Jin, L., Cai, H., Hu, L., Xin, Y. (2026). *The Efficiency Frontier: A Unified Framework for Cost–Performance Optimization in LLM Context Management*. arXiv:2605.23071.
- Shi & Ding (2025). *Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective*. arXiv:2508.16712.
- (2025). *Scaling LLM Multi-turn RL with Summarization-based Context Management* (SUPO). arXiv:2510.06727.
- Du et al. (2025); Chroma (2025). Estudios sobre *context rot* y ventana efectiva de contexto.
- Estudios de la "ley 1/W" (2026) sobre tokens por vatio y longitud de contexto. arXiv:2603.17280.
- TokenPowerBench (2025); ML.ENERGY Benchmark (2025); Jegham et al. (2025), *How Hungry is AI?*. arXiv:2505.09598.
- Lin et al., *AWQ*; Kwon et al., *vLLM / PagedAttention*; Dettmers et al., *bitsandbytes*.

*(Completar formato de citación según los lineamientos del programa.)*
