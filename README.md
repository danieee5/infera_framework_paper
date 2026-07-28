# INFERA

INFERA es un procedimiento reproducible para caracterizar la energía de una
GPU durante conversaciones incrementales con un modelo de lenguaje
autoalojado. Registra potencia, duración, tokens, respuestas, intervenciones
sobre el historial y cumplimiento programático de las tareas.

El repositorio permite dos recorridos:

1. auditar sin GPU el experimento publicado;
2. repetir o adaptar la medición física en una GPU NVIDIA.

Proyecto académico de Daniela Mora, Universidad de Especialidades Espíritu
Santo, Ecuador, 2026.

## Experimento principal

El estudio empleó un diseño factorial completo `3 × 2`:

- política de historial: `completo`, `resumen` y `descarte`;
- representación numérica: `AWQ` y `FP16`;
- tres repeticiones instrumentales por condición.

Cada sesión recorrió las mismas 29 tareas. Las tres pasadas comprobaron
estabilidad local del instrumento sobre una trayectoria fija; no son réplicas
independientes de calidad.

La evidencia está en
[`experimentos/experimento_principal/`](./experimentos/experimento_principal/).
Los nombres de los crudos siguen esta forma:

```text
run_<representación>_<política>_rep<número>.jsonl
```

Por ejemplo, `run_AWQ_resumen_rep2.jsonl` es la segunda repetición
instrumental de la condición AWQ con resumen.

## Auditar los resultados sin GPU

La auditoría verifica hashes, tamaños, conteos, finalización, ausencia de
parciales, equivalencia con el paquete preservado y reanálisis de los
resultados:

```bash
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

La salida válida informa `ok: true`, 18 sesiones, 29 tareas por sesión, cero
parciales y `reanalysis_matches_download: true`. Este recorrido utiliza CPU y
la biblioteca estándar de Python.

Los detalles y controles están en la
[guía de reproducción](./docs/REPRODUCCION.md).

## Regenerar las figuras

Instala el entorno de análisis y genera los derivados en una ruta nueva:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python3 infera/figuras_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --output /tmp/infera_figuras
```

Las figuras ya generadas y su manifiesto están en
[`experimentos/experimento_principal/figuras/`](./experimentos/experimento_principal/figuras/).

## Repetir la medición física

Una repetición física requiere Linux, Python 3.10, CUDA, NVML, una GPU NVIDIA
dedicada y los dos checkpoints declarados. Desde `infera/`:

```bash
python -m unittest tests.test_tres_brazos
cp config/reference/experiment.env.example config/experiment.env
# Edita únicamente las rutas de FP16_MODEL y AWQ_MODEL.
bash run_campana_tres_brazos.sh
```

Antes de inferir, el launcher valida escenario, tokenizadores, presupuesto,
versiones, GPU exclusiva y telemetría. Una GPU, checkpoint o versión diferente
produce una nueva caracterización y no tiene por qué repetir los joules de la
RTX 4090.

Consulta [la guía de ejecución](./infera/README.md) antes de utilizar GPU.

## Organización

- [`experimentos/`](./experimentos/): experimento vigente y mapa de evidencia.
- [`infera/`](./infera/): instrumentación, runner, políticas, controles,
  análisis y pruebas.
- [`docs/`](./docs/): reproducción, configuración, salidas y limitaciones.
- [`requirements.txt`](./requirements.txt): análisis y figuras sin GPU.
- [`requirements-gpu.txt`](./requirements-gpu.txt): stack físico validado.

El conjunto anterior de dos políticas no forma parte de la evidencia vigente.
Se conserva en la historia Git, separado del experimento principal, para
evitar que sus 12 sesiones se mezclen con las 18 sesiones finales.

## Alcance

Los valores caracterizan RTX 4090, Llama 3.1 8B Instruct, vLLM 0.5.3, FP16,
AWQ, caché de prefijos desactivada y la secuencia fija de 29 tareas. El
disparador de 4.500 tokens y `K=4` fueron decisiones operativas
preespecificadas, no valores óptimos.

El repositorio todavía no declara una licencia de reutilización ni una forma
de citación definitiva. Esos archivos solo deben añadirse cuando se confirmen
las condiciones institucionales y los metadatos finales.
