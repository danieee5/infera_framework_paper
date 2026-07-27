# Ejecutar INFERA

Esta carpeta contiene el runner principal para medir y analizar una
conversación incremental. El conjunto de referencia publicado conserva el
diseño original de dos brazos. La campaña de tres brazos añade una tercera
política de descarte por recencia.

INFERA envía la misma secuencia de tareas a:

- un brazo que conserva el historial completo;
- un brazo que genera un resumen cuando el prompt supera la regla declarada;
- en la campaña nueva, un brazo que conserva solo los cuatro pares completos
  más recientes mediante un recorte local.

La unidad de análisis es la sesión completa. El costo de cada resumen se mide
y se añade a la energía de la estrategia compactada.

La campaña nueva compara `completo | resumen | descarte`. El descarte conserva
los cuatro pares usuario/asistente completos más recientes y no emite una
petición propia. Esto no iguala por sí solo la ocupación del prompt frente al
resumen; el analizador informa la ocupación observada.

## Antes de comenzar

Para una medición completa necesitas:

- Linux;
- una GPU NVIDIA dedicada;
- acceso funcional a NVML;
- CUDA compatible;
- Python 3.10;
- una representación FP16 y otra AWQ del modelo;
- espacio para modelos, entorno y resultados.

Una GPU o versión de software diferente puede utilizar el procedimiento, pero
producirá una nueva caracterización. No se espera que repita exactamente los
joules de la RTX 4090.

## Archivos que vas a utilizar

- `config/experiment.env.example`: plantilla de modelos y parámetros.
- `config/session_tasks.example.json`: ejemplo mínimo de tareas.
- `config/validate_configuration.py`: comprueba estructura y presupuesto.
- `kb/`: base ficticia que se envía como mensaje de sistema.
- `setup_infera.sh`: crea el entorno de ejecución con GPU.
- `overnight.sh`: sirve FP16 y AWQ, ejecuta ambos brazos y analiza la corrida.
- `run_all.sh`: prueba una sola representación que ya esté servida.
- `infera_session_runner.py`: ejecuta una sesión y escribe el JSONL.
- `gpu_power_monitor.py`: toma muestras NVML e integra la energía.
- `infera_compaction.py`: genera y aplica los resúmenes.
- `infera_quality.py`: calcula el puntaje programático.
- `analyze_results.py`: valida los JSONL y genera tablas, figuras e informe.
- `run_campana_tres_brazos.sh`: orquesta las 18 sesiones del diseño nuevo.
- `preflight_campana_tres_brazos.py`: congela tokenizers, versiones, hashes y
  presupuesto sin ejecutar inferencia.
- `analiza_tres_brazos.py`: valida la campaña completa y separa energía,
  mecanismo y puntaje programático.
- `reanaliza_campana_tres_brazos.sh`: recupera el análisis de 18 sesiones ya
  recolectadas sin levantar vLLM ni volver a usar GPU.
- `audita_paquete_tres_brazos.py`: comprueba sin GPU el paquete descargado,
  todos sus hashes, conteos y una reanálisis opcional sobre copias temporales.
- `figuras_tres_brazos.py`: genera cuatro figuras PNG/PDF desde una campaña
  ya validada, sin modificar raws ni emitir inferencias.

## Preparar tu experimento

### 1. Crear archivos de trabajo

```bash
cp config/experiment.env.example config/experiment.env
cp config/session_tasks.example.json config/mi_sesion.json
```

`config/experiment.env` está ignorado por Git. No guardes allí tokens ni
credenciales.

### 2. Definir la sesión

Edita `config/mi_sesion.json`. Cada tarea debe tener:

- un identificador único;
- un tipo;
- un prompt;
- sus dependencias;
- reglas de verificación que correspondan con la respuesta esperada.

El ejemplo contiene tres tareas únicamente para comprobar el flujo. Una
medición de compactación necesita una sesión suficientemente larga para que
la regla se active y debe declarar de antemano cuántos eventos espera.

### 3. Preparar la base de conocimiento

Los archivos ficticios están en `kb/`. Puedes reemplazar su contenido
manteniendo los nombres, o adaptar `infera_kb.py` si necesitas otra estructura.
Utiliza únicamente información ficticia, anonimizada o autorizada.

### 4. Configurar modelos y política

Edita `config/experiment.env` y reemplaza:

- `FP16_MODEL` y `AWQ_MODEL`;
- `SESSION` por `config/mi_sesion.json`;
- `SESSION_TAG` por una etiqueta breve sin espacios;
- `THRESH` por la regla predefinida de compactación;
- `MAX_MODEL_LEN`;
- `EXPECTED_TASKS`;
- `EXPECTED_COMPACTIONS`;
- `REPS`, si no utilizarás tres.

El umbral se compara con los tokens del prompt informados por vLLM. Es una
decisión de la política, no un valor óptimo calculado por INFERA.

## Validar sin usar GPU

```bash
python config/validate_configuration.py \
  --tasks config/mi_sesion.json \
  --kb-dir kb \
  --max-model-len 8192 \
  --threshold 4500 \
  --tokenizer /ruta/al/modelo-o-tokenizador
```

No continúes si aparecen errores. La advertencia que indica ausencia de
tokenizador significa que todavía no se comprobó el presupuesto real.

## Campaña nueva de tres brazos

El launcher nuevo es independiente de `overnight.sh` y no modifica
`results/reference/`:

```bash
cp config/reference/experiment.env.example config/experiment.env
# Edita FP16_MODEL y AWQ_MODEL con las rutas reales.
bash run_campana_tres_brazos.sh
```

Antes de levantar vLLM exige el stack histórico exacto, tokenizers equivalentes,
29 tareas, una RTX 4090 exclusiva, telemetría NVML completa y un directorio de
salida nuevo.
Cada petición se vuelve a contar con la plantilla de chat real y aborta antes
de enviarse si `prompt_tokens + max_tokens > 8192`. Los JSONL se escriben como
`.partial` y solo se publican al completar la sesión. La caché de prefijos queda
apagada; encenderla define otro sistema experimental.

Cada muestra NVML queda persistida dentro del registro de su llamada: potencia,
VRAM, temperatura, clocks, utilización, estado de rendimiento, razones de
clock/throttling y PIDs/PGIDs de cómputo. El baseline conserva la misma traza en
el manifiesto de sesión. El analizador vuelve a integrar cada traza y comprueba
cobertura temporal, buffers y ausencia de procesos GPU ajenos. También registra
`finish_reason`, respuesta completa, tokens, tiempos y metadatos del servidor.

El análisis solo se publica si existen las 18 sesiones, sus manifiestos y
hashes, el orden contrabalanceado predeclarado y exactamente 29 tareas medidas
por sesión. Sus tres pasadas son repeticiones instrumentales de una trayectoria
fija, no réplicas independientes de calidad.

### Única corrida final en RunPod

No ejecutes una prueba GPU separada si solo puedes pagar una campaña. Las
pruebas unitarias siguientes no usan GPU:

```bash
cd infera
python -m unittest tests.test_tres_brazos
cp config/reference/experiment.env.example config/experiment.env
# Edita únicamente FP16_MODEL y AWQ_MODEL con directorios locales reales.
```

Instala y activa el entorno una sola vez:

```bash
cd ..
bash infera/setup_infera.sh
source infera/.runtime/venv/bin/activate
cd infera
```

Inicia la campaña dentro de `tmux`, y despréndete con `Ctrl-b d`:

```bash
tmux new -s infera-final
bash run_campana_tres_brazos.sh
```

No lances una segunda instancia. El preflight inicial no emite inferencias,
pero puede tardar porque calcula una huella completa de ambos modelos. Cada
repetición carga un servidor nuevo; sus tres brazos comparten los pesos dentro
del bloque, con orden latino, 120 s de cooldown, cinco warmups, 30 s de
estabilización y 30 s de baseline.

Al terminar, conserva y descarga la carpeta completa
`results/runs/tres_brazos_<UTC>/`, no solo los CSV. El estado correcto es
`complete`:

```bash
python -c 'import json,glob; p=sorted(glob.glob("results/runs/tres_brazos_*/manifiesto_campana.json"))[-1]; d=json.load(open(p)); print(p, d["status"], len(d["artifacts"]["raws"]))'
```

Si el estado es `failed` pero existen los 18 JSONL finales y sus 18 manifiestos,
recupera solo el análisis con:

```bash
bash reanaliza_campana_tres_brazos.sh \
  results/runs/tres_brazos_<UTC>
```

Esto no levanta vLLM. Si faltan sesiones o quedó algún `.partial`, la campaña
es incompleta y no debe incorporarse al manuscrito. Una interrupción del Pod,
un modelo incompatible o un fallo físico siguen siendo incertidumbres que
ningún launcher puede eliminar.

### Auditar la descarga y generar figuras sin GPU

Desde la raíz del repositorio, conserva la carpeta completa, el archivo
comprimido y el `.sha256`. La auditoría es de solo lectura; la reanálisis usa
copias temporales:

```bash
python3 infera/audita_paquete_tres_brazos.py \
  --campaign tres_brazos_<UTC> \
  --archive tres_brazos_<UTC>.tar.gz.bin \
  --checksum tres_brazos_<UTC>.tar.gz.sha256 \
  --reanalysis /tmp/reanalysis_tres_brazos
```

Solo acepta el paquete si informa `ok: true`,
`reanalysis_matches_download: true`, 18 raws, 18 manifiestos de sesión,
29 tareas y 0 parciales. La extensión `.bin` permite conservar el contenedor
exacto cuando el navegador descomprime automáticamente un `.tar.gz`.

Las figuras se crean en un directorio nuevo y fallan si la salida ya existe:

```bash
python3 infera/figuras_tres_brazos.py \
  --campaign tres_brazos_<UTC> \
  --output /ruta/nueva/figuras
```

El puntaje mostrado es programático, no una evaluación humana de tarea
resuelta. La cuarta figura incluye una sensibilidad mínima que excluye
respuestas terminadas por `max_tokens`. Los textos completos permanecen en
los JSONL para una revisión humana posterior.

## Instalar el entorno GPU

```bash
bash setup_infera.sh
source .runtime/venv/bin/activate
```

Para guardar entorno y caché en un volumen persistente:

```bash
export INFERA_RUNTIME=/ruta/persistente/infera_runtime
bash setup_infera.sh
source /ruta/persistente/infera_runtime/venv/bin/activate
```

## Ejecutar una prueba de humo

Sirve una representación en una terminal. En otra terminal:

```bash
RUN_TAG=smoke_fp16 \
TASKS=config/mi_sesion.json \
./run_all.sh FP16 /ruta/al/modelo-fp16 1 4500
```

Comprueba en `results/runs/smoke_fp16/` que:

- todas las filas tengan `status=ok`;
- `nvml_available` sea verdadero;
- `energy_j` sea positivo;
- no exista truncamiento inesperado;
- la cantidad de tareas sea la prevista.

Una réplica comprueba el funcionamiento, pero no permite estimar variación.

## Ejecutar la medición completa

Cuando la configuración y el humo sean correctos:

```bash
tmux new -s infera
bash overnight.sh
```

`overnight.sh` exige `config/experiment.env`; no utiliza silenciosamente la
configuración del paper. El launcher:

1. sirve FP16;
2. ejecuta historial completo y compactación;
3. repite según `REPS`;
4. detiene únicamente el servidor que inició;
5. repite el proceso con AWQ;
6. ejecuta `analyze_results.py`.

La corrida queda en:

```text
results/runs/<etiqueta_fecha>/
├── run_<etiqueta>_FP16_naive_rep1.jsonl
├── run_<etiqueta>_FP16_compaction_rep1.jsonl
├── run_<etiqueta>_AWQ_naive_rep1.jsonl
├── run_<etiqueta>_AWQ_compaction_rep1.jsonl
├── analysis/
├── execution.log
└── vllm_*.log
```

## Analizar manualmente una corrida

Si necesitas repetir solo el análisis:

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

Ajusta los últimos cuatro parámetros al diseño que declaraste antes de medir.
El analizador comprueba los nombres esperados y no mezcla archivos
adicionales.

## Conjunto de referencia del paper

`results/reference/` no es el destino de una corrida nueva. Contiene:

- `raw/`: los doce JSONL publicados y sus huellas SHA-256;
- `expected/`: tablas y figuras que deben regenerarse desde esos JSONL.

Para auditarlos sin GPU:

```bash
python analyze_results.py
```

La salida debe informar 12 sesiones y 366 filas.

## Cómo interpretar la diferencia de energía

El análisis calcula:

```text
energía acumulada de compactación − energía acumulada del historial completo
```

- Un valor positivo significa que la compactación conserva una deuda.
- Cero significa que alcanzó el punto de equilibrio.
- Un valor negativo significa que recuperó el costo dentro del horizonte.

Consulta también:

- [Guía detallada desde cero](./GUIA_DESDE_CERO.md)
- [Configuración](../docs/CONFIGURACION.md)
- [Salidas](../docs/SALIDAS.md)
- [Limitaciones](../docs/LIMITACIONES.md)
