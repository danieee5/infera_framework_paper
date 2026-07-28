# Resultados

Esta carpeta separa los datos publicados del estudio y las mediciones que
genere cada usuario.

## `runs/`

Es el destino de las corridas nuevas.
`run_campana_tres_brazos.sh` crea una subcarpeta fechada y guarda allí:

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

## Evidencia publicada

Los resultados del paper no se copian en `runs/`. La única ruta pública
vigente es:

```text
../../experimentos/experimento_principal/evidencia/
```

El conjunto anterior bajo `results/reference/` fue archivado para impedir que
sus 12 sesiones se mezclen con las 18 sesiones finales.
