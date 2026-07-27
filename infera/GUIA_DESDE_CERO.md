# Guía desde cero

Esta guía conduce desde una configuración propia hasta los JSONL, tablas y
figuras finales. Para entenderla no necesitas haber leído el paper.

## 1. Comprender qué se medirá

INFERA ejecuta una conversación incremental bajo dos estrategias:

- **Historial completo:** conserva todas las interacciones.
- **Compactación:** genera un resumen cuando el prompt supera `THRESH`.

La energía de la estrategia compactada incluye tanto las tareas ordinarias
como las llamadas que producen los resúmenes. Cada representación numérica y
cada estrategia se repiten según `REPS`.

## 2. Preparar el equipo

Necesitas:

- Linux;
- GPU NVIDIA dedicada;
- CUDA y NVML funcionales;
- Python 3.10;
- modelos compatibles servibles por vLLM;
- espacio suficiente para pesos, caché y entorno.

Comprueba primero:

```bash
nvidia-smi
```

No utilices una GPU compartida con otras cargas: NVML mide la potencia total
de la tarjeta.

## 3. Crear la configuración

Desde `infera/`:

```bash
cp config/experiment.env.example config/experiment.env
cp config/session_tasks.example.json config/mi_sesion.json
```

Edita `config/mi_sesion.json` y reemplaza prompts, dependencias y reglas. El
ejemplo de tres tareas es un humo, no un experimento suficiente para evaluar
compactación.

Edita `config/experiment.env` y declara:

- las rutas de los modelos;
- `SESSION=config/mi_sesion.json`;
- una etiqueta nueva en `SESSION_TAG`;
- la regla `THRESH`;
- el presupuesto `MAX_MODEL_LEN`;
- tareas y compactaciones esperadas;
- réplicas y tiempo de enfriamiento.

No incluyas tokens ni credenciales.

## 4. Preparar la base de conocimiento

La base ficticia está en `kb/` y contiene un Markdown y dos CSV. Reemplaza el
contenido con información ficticia, anonimizada o autorizada.

El runner actual espera esos tres nombres. Si necesitas otro número o formato
de archivos, adapta `infera_kb.py` y vuelve a validar el procedimiento.

## 5. Validar estructura y tokens

```bash
python config/validate_configuration.py \
  --tasks config/mi_sesion.json \
  --kb-dir kb \
  --max-model-len 8192 \
  --threshold 4500 \
  --tokenizer /ruta/al/modelo-o-tokenizador
```

La validación comprueba identificadores, dependencias, reglas, archivos de
base y margen de entrada/salida. Corrige todos los errores.

## 6. Instalar el entorno

```bash
bash setup_infera.sh
source .runtime/venv/bin/activate
```

Si trabajas en un proveedor con volumen persistente:

```bash
export INFERA_RUNTIME=/ruta/persistente/infera_runtime
bash setup_infera.sh
source /ruta/persistente/infera_runtime/venv/bin/activate
```

El instalador no descarga automáticamente los modelos. Configura sus rutas en
`config/experiment.env`.

## 7. Ejecutar un humo

Sirve FP16 en una terminal:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /ruta/al/modelo-fp16 \
  --dtype float16 \
  --max-model-len 8192 \
  --port 8000
```

En otra terminal:

```bash
RUN_TAG=smoke_fp16 \
TASKS=config/mi_sesion.json \
./run_all.sh FP16 /ruta/al/modelo-fp16 1 4500
```

Revisa el JSONL. Debe contener energía positiva, NVML disponible, estados
correctos y el número de tareas previsto. No continúes con la corrida completa
si el humo falla.

## 8. Ejecutar ambos modelos y ambas estrategias

Cuando el humo sea correcto:

```bash
tmux new -s infera
bash overnight.sh
```

El launcher no se inicia sin `config/experiment.env`. Sirve FP16 y AWQ de
forma secuencial, ejecuta ambos brazos y escribe en una carpeta nueva:

```text
results/runs/<SESSION_TAG_fecha>/
```

Para desprenderte de `tmux` sin detener la ejecución, pulsa `Ctrl+b` y luego
`d`.

## 9. Revisar el análisis

Al terminar, la subcarpeta `analysis/` debe contener tablas, figuras,
`analysis_summary.md` y `manifest.json`.

La diferencia acumulada se interpreta así:

- positiva: la compactación todavía debe energía;
- cero: alcanzó el punto de equilibrio;
- negativa: recuperó el costo dentro del horizonte.

Menos tokens no demuestran por sí solos menos energía. Revisa también las
salidas, la cantidad de resúmenes, la duración y el puntaje programático.

## 10. Repetir únicamente el análisis

```bash
python analyze_results.py \
  --source results/runs/mi_corrida \
  --out results/runs/mi_corrida/analysis \
  --session-tag mi_experimento \
  --quants AWQ,FP16 \
  --reps 1,2,3 \
  --expected-tasks 29 \
  --expected-compactions 3
```

Los parámetros deben coincidir con tu configuración declarada.

## Auditar el conjunto publicado, sin GPU

Esta ruta es opcional y está destinada a revisores:

```bash
python analyze_results.py
```

El comando lee `results/reference/raw/` y regenera
`results/reference/expected/`. Debe informar 12 sesiones y 366 filas.

## Problemas frecuentes

- **NVML no disponible:** detén la corrida y corrige el acceso a la GPU.
- **Energía igual a cero:** considera inválida la medición.
- **Memoria insuficiente:** ajusta y documenta el diseño antes de medir; no
  cambies parámetros entre brazos.
- **Salida existente:** usa otra etiqueta. No sobrescribas una corrida que
  quieras conservar.
- **Conteo inesperado:** revisa la configuración y los errores; no cambies los
  valores esperados para ocultar archivos faltantes.
- **No hubo compactaciones:** el ejemplo es demasiado corto o la regla no se
  activó. Rediseña la sesión antes de la medición definitiva.

Más información:

- [Configuración](../docs/CONFIGURACION.md)
- [Instalación](../docs/INSTALACION.md)
- [Archivos de salida](../docs/SALIDAS.md)
- [Limitaciones](../docs/LIMITACIONES.md)
