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
- Tres réplicas describen repetibilidad local; no equivalen a múltiples GPU ni
  a una muestra amplia de hardware.
- La regla de 4.500 tokens fue operativa. El repositorio no estima un umbral
  óptimo.
- Los JSONL principales no guardaron respuestas ni resúmenes. El puntaje
  programático histórico se describe, pero no puede reconstruirse manualmente
  desde esos doce archivos.
- Menos tokens de entrada no implican necesariamente menos energía de sesión:
  también intervienen las salidas, llamadas de resumen, duración y potencia.
- No se realizó una evaluación humana ciega de las respuestas principales.
- No se debe extrapolar el punto de equilibrio más allá de las 29 tareas
  observadas.

Estas limitaciones no impiden reproducir la contabilidad publicada; delimitan
qué conclusiones pueden obtenerse de ella.
