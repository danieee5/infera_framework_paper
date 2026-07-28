# Campaña de tres políticas de gestión del historial

Energía y puntaje programático se informan por separado. Las tres
pasadas son repeticiones instrumentales de una trayectoria fija;
no son réplicas independientes de tareas o calidad.

## Energía media por sesión

| Precisión | Política | Tareas, J | Mecanismo, J | Total, J | Sobre reposo, J | CV instrumental % |
|---|---|---:|---:|---:|---:|---:|
| AWQ | historial completo (calibración) | 11171.37 | 0.00 | 11171.37 | 10590.78 | 0.180 |
| AWQ | recencia ciega por antigüedad | 9681.30 | 0.00 | 9681.30 | 9148.17 | 0.182 |
| AWQ | resumen del historial | 10722.77 | 4017.26 | 14740.03 | 14001.63 | 0.181 |
| FP16 | historial completo (calibración) | 15342.46 | 0.00 | 15342.46 | 14599.96 | 0.170 |
| FP16 | recencia ciega por antigüedad | 13806.75 | 0.00 | 13806.75 | 13106.18 | 0.130 |
| FP16 | resumen del historial | 15480.11 | 7005.17 | 22485.27 | 21460.41 | 0.004 |

## Contraste primario resumen − descarte

- **AWQ: diferencia end-to-end +5058.73 J.**
  Costo directo medio de llamadas de resumen: 4017.26 J; descarte local: 0 J.
  Tareas con puntaje programático 1: resumen 27.00 (rango 27–27), descarte 24.00 (rango 24–24).
- **FP16: diferencia end-to-end +8678.52 J.**
  Costo directo medio de llamadas de resumen: 7005.17 J; descarte local: 0 J.
  Tareas con puntaje programático 1: resumen 27.00 (rango 27–27), descarte 23.00 (rango 23–23).

La diferencia end-to-end incluye la llamada de resumen, la ocupación
realizada, el contenido retenido, las respuestas y su propagación.
No es solo el precio del mecanismo.

## Diagnóstico por tarea frente a completo

- AWQ, repetición 1, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- AWQ, repetición 1, recencia ciega por antigüedad — preserva: 24, perjudica: 3, repara: 0, ambos fallan: 2
- AWQ, repetición 2, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- AWQ, repetición 2, recencia ciega por antigüedad — preserva: 24, perjudica: 3, repara: 0, ambos fallan: 2
- AWQ, repetición 3, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- AWQ, repetición 3, recencia ciega por antigüedad — preserva: 24, perjudica: 3, repara: 0, ambos fallan: 2
- FP16, repetición 1, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- FP16, repetición 1, recencia ciega por antigüedad — preserva: 23, perjudica: 4, repara: 0, ambos fallan: 2
- FP16, repetición 2, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- FP16, repetición 2, recencia ciega por antigüedad — preserva: 23, perjudica: 4, repara: 0, ambos fallan: 2
- FP16, repetición 3, resumen del historial — preserva: 26, perjudica: 1, repara: 1, ambos fallan: 1
- FP16, repetición 3, recencia ciega por antigüedad — preserva: 23, perjudica: 4, repara: 0, ambos fallan: 2

## Diagnóstico de medición

La traza NVML cruda de cada llamada y baseline fue validada y permite
recalcular energía, duración, potencia y memoria. `por_sesion.csv`
incluye temperatura, clocks, utilización, eventos de clock y salidas
terminadas por `max_tokens`. Cualquier proceso GPU fuera del grupo de
vLLM invalida la sesión.

## Alcance y amenazas

Estas cifras describen una trayectoria sintética, un orden, K=4,
un umbral, dos checkpoints, una GPU y vLLM sin caché de prefijos.
K=4 es un parámetro libre sin análisis de sensibilidad. Un umbral
común no iguala ocupación, y los prompts divergen al acumular outputs.
El descarte por antigüedad es una línea base débil, no el estado del arte.
El puntaje es programático; tareas con reglas semánticas abiertas no
equivalen a una evaluación humana de éxito.
No se estiman valores p, población, umbral óptimo, política universal,
energía por fase ni generalización a otros sistemas.
