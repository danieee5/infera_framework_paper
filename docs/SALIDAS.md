# Archivos de salida

## JSONL de cada sesión

El runner escribe un objeto JSON por línea. Los campos principales son:

- `run_id`: representación, estrategia y réplica.
- `task_id`, `task_index`, `task_type`: evento y posición.
- `accumulated_prompt_tokens`: tokens de entrada informados por vLLM.
- `completion_tokens`: tokens de salida.
- `energy_j`: energía integrada en la ventana NVML.
- `cumulative_energy_j`: energía acumulada de la sesión.
- `is_compaction`: distingue tarea ordinaria y resumen.
- `quality`: puntaje programático.
- `status`, `error`: resultado de la petición.
- `nvml_available`, `nvml_samples`: controles de instrumentación.
- `prompt_text`, `response_text`: textos preservados por el runner actual.

Los doce JSONL del conjunto de referencia se produjeron antes de incorporar
los dos últimos campos.

## Análisis de una corrida

`analyze_results.py` produce:

- `integrity_provenance.csv`;
- `normalized_rows.csv`;
- `per_run_summary.csv`;
- `aggregate_summary.csv`;
- `compaction_accounting.csv`;
- `cycle_accounting.csv`;
- `cumulative_by_task.csv`;
- `cumulative_curve_mean.csv`;
- `paired_task_effects.csv`;
- `compaction_events.csv`;
- `handoff_summary.csv`;
- `token_summary.csv`;
- `quality_programmatic_summary.csv`;
- `figura_delta_energia_acumulada.png`;
- `figura_contabilidad_energia.png`;
- `figura_tokens_por_politica.png`;
- `analysis_summary.md`;
- `manifest.json`.

`manifest.json` registra archivos leídos, huellas, alcance declarado, errores
y advertencias.

## Qué archivo revisar

- Totales por condición: `aggregate_summary.csv`.
- Costo de los resúmenes: `compaction_accounting.csv`.
- Punto de equilibrio: `cumulative_by_task.csv` y `cycle_accounting.csv`.
- Tokens: `token_summary.csv`.
- Procedencia: `manifest.json` e `integrity_provenance.csv`.
- Explicación breve: `analysis_summary.md`.

Las nuevas corridas se guardan en `infera/results/runs/`. El conjunto publicado
está separado en `infera/results/reference/`.
