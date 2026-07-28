# Limitaciones de reproducción e interpretación

- NVML informa la potencia total de la GPU. Otros procesos que usen la misma
  tarjeta contaminan la medición.
- El margen de 500 ms forma parte de esta configuración. Incluye tiempo
  alrededor de la petición y no separa por sí solo energía de prellenado,
  decodificación y reposo.
- El experimento usa una RTX 4090, Llama 3.1 8B, vLLM 0.5.3, dos
  representaciones y una secuencia fija. Los valores no son universales.
- El caché de prefijos estuvo desactivado. Activarlo cambia el trabajo repetido
  entre peticiones y requiere una nueva medición.
- Tres repeticiones instrumentales describen estabilidad local de una
  trayectoria fija; no equivalen a múltiples GPU, tareas independientes ni
  réplicas independientes de calidad.
- La regla de 4.500 tokens y `K=4` fueron decisiones operativas. El repositorio
  no estima valores óptimos.
- `K=4` cuenta pares completos usuario/asistente conservados; no representa
  tokens ni mensajes individuales.
- Los 18 JSONL guardan respuestas, resúmenes y razones de cierre. Esto permite
  auditoría focal, pero no reemplaza una evaluación humana ciega y
  preespecificada.
- Menos tokens de entrada no implican necesariamente menos energía de sesión:
  también intervienen las salidas, llamadas de resumen, duración y potencia.
- El cumplimiento programático comprueba reglas declaradas y puede producir
  falsos positivos o negativos semánticos; no equivale a calidad integral.
- No se debe extrapolar el punto de equilibrio más allá de las 29 tareas
  observadas.

Estas limitaciones no impiden reproducir la contabilidad publicada; delimitan
qué conclusiones pueden obtenerse de ella.
