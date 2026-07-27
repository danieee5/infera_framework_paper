# Mapa del repositorio para el anexo

## Propósito

El repositorio permite:

1. configurar y ejecutar una medición propia en una GPU NVIDIA;
2. transformar los JSONL obtenidos en tablas, figuras e informe;
3. auditar, sin GPU, las mediciones utilizadas en el paper.

## Flujo de una medición

```text
configuración de tareas + base de conocimiento
                    ↓
       infera_session_runner.py
                    ↓
          JSONL por cada sesión
                    ↓
          analyze_results.py
                    ↓
       tablas + figuras + manifiesto
```

La medición física ocurre en el runner. El analizador trabaja sobre los JSONL;
no inventa ni simula energía.

## Componentes públicos

- `infera/config/`: plantillas, validación y configuración de referencia.
- `infera/kb/`: base ficticia de ejemplo.
- `infera/*.py`: medición, compactación, calidad y análisis.
- `infera/results/runs/`: destino local de nuevas corridas.
- `infera/results/reference/raw/`: mediciones publicadas.
- `infera/results/reference/expected/`: productos que deben regenerarse.
- `docs/`: instalación, configuración, salidas y límites.

## Conjunto de referencia

Incluye doce sesiones:

```text
2 representaciones × 2 estrategias × 3 réplicas
```

Los brazos comparten 29 tareas. Las sesiones compactadas incluyen tres
llamadas adicionales de resumen.

## Comando de auditoría

```bash
cd infera
shasum -a 256 -c results/reference/raw/SHA256SUMS
python analyze_results.py
```

El resultado esperado es 12 sesiones, 366 filas y validación correcta.

## Controles de una corrida nueva

- archivo de configuración explícito;
- validación de tareas y presupuesto;
- salida protegida contra sobrescritura;
- cancelación si NVML no está disponible;
- respuestas conservadas en las nuevas mediciones;
- huellas y manifiesto;
- análisis automático de tablas y figuras;
- documentación de parámetros y límites.

La numeración y el nombre del anexo deben corresponder con la versión final del
manuscrito.
