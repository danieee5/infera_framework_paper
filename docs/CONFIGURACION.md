# Configurar una corrida

## Repetir el diseño publicado

Desde `infera/`, crea el único archivo local de entorno:

```bash
cp config/reference/experiment.env.example config/experiment.env
```

Conserva los parámetros declarados y sustituye únicamente las rutas locales de
`FP16_MODEL` y `AWQ_MODEL` si quieres repetir la configuración:

- 29 tareas de `config/reference/session_tasks.json`;
- base sintética de `kb/`;
- representaciones AWQ y FP16;
- políticas completo, resumen y descarte;
- tres repeticiones instrumentales;
- contexto máximo de 8.192 tokens;
- disparador de resumen en 4.500 tokens;
- `K=4` pares completos para descarte;
- temperatura 0, semilla fija y caché de prefijos apagada.

El preflight del launcher valida estos valores antes de levantar vLLM.

## Variables de política

`completo` mantiene todo el historial. `resumen` lee el historial activo,
genera un relevo y continúa con base fija, relevo y pares nuevos. `descarte`
mantiene la base fija y los cuatro pares completos más recientes.

El disparador y `K` deben definirse antes de medir. INFERA no los optimiza. Un
valor distinto cambia las intervenciones y constituye otra caracterización.

## Presupuesto

`MAX_MODEL_LEN=8192` limita cada petición, no la suma de toda la sesión. Antes
de enviar, el runner cuenta la plantilla de chat real y exige:

```text
prompt_tokens + max_tokens ≤ MAX_MODEL_LEN
```

La longitud de las respuestas también altera el historial posterior. Por eso
no basta con estimar el presupuesto a partir de palabras o número de turnos.

## Cumplimiento programático

Cada tarea tiene un identificador, tipo, prompt, dependencias y reglas
`verify`. Las reglas buscan campos o restricciones predeclaradas. El campo
histórico `judge` no instala un juez LLM; el runner utiliza `score_task`.

El resultado permite comprobar consistencia operacional, pero no equivale a
calidad semántica integral ni evaluación humana.

## Adaptar el caso

Las plantillas `config/experiment.env.example` y
`config/session_tasks.example.json` sirven para una corrida nueva. Si cambias
tareas o base:

1. utiliza información ficticia, anonimizada o autorizada;
2. adapta las reglas programáticas a las respuestas esperadas;
3. valida estructura, dependencias y presupuesto con el tokenizador real;
4. declara de antemano cantidad de tareas, políticas y repeticiones;
5. conserva la misma secuencia para todas las condiciones;
6. publica esa corrida bajo un identificador nuevo.

Una semilla fija y temperatura 0 mejoran repetibilidad local, pero no
garantizan igualdad entre motores, versiones o hardware.
