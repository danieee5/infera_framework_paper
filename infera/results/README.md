# Resultados

Esta carpeta separa los datos publicados del estudio y las mediciones que
genere cada usuario.

## `runs/`

Es el destino de las corridas nuevas. `overnight.sh` crea una subcarpeta
fechada y guarda allí:

- un JSONL por representación, estrategia y réplica;
- registros del servidor y del launcher;
- una subcarpeta `analysis/` con tablas, figuras e informe.

Git ignora el contenido de `runs/`, excepto `.gitkeep`, para evitar publicar
accidentalmente mediciones, respuestas o bases propias.

La campaña de tres brazos crea una carpeta `tres_brazos_<UTC>` con `raw/`,
`logs/`, `preflight.json`, `manifiesto_campana.json` y `analisis/`. Debe
descargarse completa: las trazas NVML primarias están en cada JSONL y en los
manifiestos de sesión, y los logs de vLLM son parte del diagnóstico. No copies
una corrida nueva dentro de `reference/`.

## `reference/`

Es un ejemplo completo y auditable construido con las mediciones del paper.
No es el destino de una corrida nueva.

- `reference/raw/`: doce JSONL y sus huellas SHA-256.
- `reference/expected/`: resultados que `analyze_results.py` debe regenerar.

Una persona puede usarlo para comprobar el analizador sin disponer de GPU:

```bash
python analyze_results.py
```

Los nombres `v3` conservados dentro de los archivos son identificadores de la
corrida original; no representan una versión que el usuario deba conocer.
