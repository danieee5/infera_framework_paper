# Archivos de salida

## JSONL de cada sesión

El runner escribe un objeto JSON por línea. Los campos principales son:

- `run_id`: representación, política y repetición.
- `task_id`, `task_index`, `task_type`: evento y posición.
- `accumulated_prompt_tokens`: tokens de entrada informados por vLLM.
- `completion_tokens`: tokens de salida.
- `energy_j`: energía integrada en la ventana NVML.
- `cumulative_energy_j`: energía acumulada de la sesión.
- `is_compaction`: distingue tarea ordinaria y llamada de resumen.
- `quality`: puntaje programático.
- `status`, `error`: resultado de la petición.
- `nvml_available`, `nvml_samples`, `nvml_trace`: controles y muestras de
  instrumentación.
- `prompt_text`, `response_text`, `finish_reason`: textos completos y cierre
  de la generación.

Los 18 JSONL vigentes conservan respuestas, tokens, tiempos y la traza NVML
de cada llamada.

## Análisis del experimento principal

`analiza_tres_brazos.py` produce dentro de `analisis/`:

- `agregado.csv`: seis condiciones y sus totales medios;
- `por_sesion.csv`: las 18 sesiones;
- `efecto_por_tarea.csv`: energía, tokens y score por tarea;
- `ocupacion_post_intervencion.csv`: estado después de resumen o descarte;
- `informe.md`: síntesis legible;
- `manifiesto_analisis.json`: procedencia, hashes y controles.

Para los resultados vigentes, revisa:

- totales y energía sobre baseline: `agregado.csv`;
- estabilidad instrumental: `por_sesion.csv`;
- diferencias a lo largo de la trayectoria: `efecto_por_tarea.csv`;
- cuándo y cómo intervino cada política:
  `ocupacion_post_intervencion.csv`;
- procedencia: `manifiesto_analisis.json`.

Las corridas nuevas se guardan en `infera/results/runs/`. La evidencia
publicada se encuentra únicamente en
`experimentos/experimento_principal/evidencia/`.
