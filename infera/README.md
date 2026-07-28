# Ejecutar INFERA

Esta carpeta contiene la instrumentación utilizada en el experimento final de
tres políticas de historial:

- `completo`: conserva todos los pares usuario/asistente;
- `resumen`: genera un relevo cuando el prompt supera la regla declarada;
- `descarte`: conserva localmente los últimos `K` pares completos.

La unidad energética es la sesión end-to-end. La energía de los relevos forma
parte del total de `resumen`; el recorte de `descarte` no emite una petición
adicional.

## Componentes del experimento principal

### Medición y controles

- `infera_session_runner.py`: recorre una sesión y publica el JSONL solo al
  completarla.
- `gpu_power_monitor.py`: muestrea NVML e integra potencia por la regla
  trapezoidal.
- `infera_compaction.py`: genera y aplica los relevos.
- `infera_quality.py`: calcula cumplimiento programático.
- `infera_kb.py`: construye el contexto fijo desde `kb/`.
- `preflight_campana_tres_brazos.py`: congela escenario, tokenizadores,
  software, hashes, presupuesto y GPU.
- `run_campana_tres_brazos.sh`: orquesta las 18 sesiones.
- `escribe_manifiesto_campana.py`: mantiene el manifiesto y el inventario.

### Auditoría y derivados

- `analiza_tres_brazos.py`: valida e integra los registros y genera seis
  artefactos de análisis.
- `reanaliza_campana_tres_brazos.sh`: recupera análisis sin levantar vLLM.
- `audita_paquete_tres_brazos.py`: comprueba hashes, paquete y reanálisis sin
  GPU.
- `figuras_tres_brazos.py`: genera cuatro figuras PNG/PDF y su manifiesto.
- `tests/test_tres_brazos.py`: pruebas CPU del diseño, runner, analizador y
  paquete.

Los nombres históricos de estos scripts se conservan porque aparecen en los
hashes y metadatos de la ejecución original. La interfaz pública estable para
los datos es
[`../experimentos/experimento_principal/`](../experimentos/experimento_principal/).

## Verificar primero sin GPU

Desde la raíz:

```bash
python3 -m unittest infera.tests.test_tres_brazos
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

Las pruebas no ejecutan inferencia ni requieren CUDA.

## Requisitos para una medición física

- Linux y Python 3.10;
- una GPU NVIDIA RTX 4090 de 24 GB dedicada para repetir la configuración
  publicada;
- CUDA, NVML y el stack de `requirements-gpu.txt`;
- Llama 3.1 8B Instruct bajo FP16 y AWQ;
- espacio persistente para entorno, pesos y resultados.

Otra GPU o versión puede utilizar el procedimiento, pero produce una nueva
caracterización. No se espera igualdad exacta de joules.

## Preparar la corrida

Desde `infera/`:

```bash
cp config/reference/experiment.env.example config/experiment.env
# Edita FP16_MODEL y AWQ_MODEL con directorios locales reales.
python -m unittest tests.test_tres_brazos
```

`config/experiment.env` está ignorado por Git. No guardes credenciales allí.
El escenario de referencia es `config/reference/session_tasks.json` y la base
sintética está en `kb/`.

Instala el entorno una sola vez:

```bash
cd ..
bash infera/setup_infera.sh
source infera/.runtime/venv/bin/activate
cd infera
```

El instalador no descarga ni cuantiza los pesos.

## Ejecutar

Utiliza una sesión persistente y no lances una segunda instancia:

```bash
tmux new -s infera
bash run_campana_tres_brazos.sh
```

El preflight no emite inferencias. Antes de levantar vLLM exige:

- versiones y tokenizadores equivalentes;
- 29 tareas y hashes de la base;
- RTX 4090 exclusiva y telemetría NVML completa;
- presupuesto `prompt_tokens + max_tokens ≤ 8192`;
- caché de prefijos desactivada;
- directorio de salida nuevo.

Cada repetición carga un servidor nuevo. Sus tres políticas comparten los
pesos dentro del bloque, con orden contrabalanceado, 120 s de enfriamiento,
cinco warmups, 30 s de estabilización y 30 s de baseline.

Los JSONL se escriben como `.partial` y solo se renombran al completar la
sesión. Cada registro conserva potencia, tiempos, tokens, respuesta,
`finish_reason`, utilización, clocks, temperatura, procesos y traza NVML.

## Aceptar o recuperar

La salida queda en:

```text
results/runs/tres_brazos_<UTC>/
├── raw/
├── analisis/
├── logs/
├── preflight.json
└── manifiesto_campana.json
```

Solo debe incorporarse a un estudio si el manifiesto declara `complete`,
`exit_code=0`, 18 sesiones, 18 manifiestos, 29 tareas por sesión y cero
parciales.

Si el launcher terminó como `failed` después de completar los 18 JSONL, puede
recuperarse únicamente el cálculo:

```bash
bash reanaliza_campana_tres_brazos.sh \
  results/runs/tres_brazos_<UTC>
```

Este comando no levanta vLLM. Si falta una sesión o existe un `.partial`, no
recupera la medición física.

## Decisiones operativas

- El disparador de 4.500 tokens decide cuándo resumir; no es un óptimo
  estimado.
- `K=4` conserva cuatro pares completos, es decir, hasta ocho mensajes del
  historial; no representa cuatro tokens.
- La caché de prefijos apagada define el sistema medido. Activarla requiere
  otra caracterización.
- Las tres pasadas son repeticiones instrumentales de una trayectoria fija,
  no réplicas independientes de desempeño o calidad.
- El puntaje es cumplimiento programático. No equivale a evaluación humana
  integral.

Consulta:

- [Reproducción sin GPU](../docs/REPRODUCCION.md)
- [Instalación](../docs/INSTALACION.md)
- [Configuración](../docs/CONFIGURACION.md)
- [Salidas](../docs/SALIDAS.md)
- [Limitaciones](../docs/LIMITACIONES.md)
