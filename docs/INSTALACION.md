# Instalación

## Ejecutar una medición con GPU

La configuración del estudio se validó con Linux, Python 3.10, CUDA 12.1,
PyTorch 2.3.1, vLLM 0.5.3 y una RTX 4090 de 24 GB dedicada.

Una GPU diferente puede ejecutar el procedimiento si tiene capacidad
suficiente, pero sus joules describirán otra configuración.

Desde la raíz:

```bash
cd infera
bash setup_infera.sh
source .runtime/venv/bin/activate
```

Para usar un volumen persistente:

```bash
export INFERA_RUNTIME=/ruta/del/volumen/infera_runtime
bash setup_infera.sh
source /ruta/del/volumen/infera_runtime/venv/bin/activate
```

El instalador prepara el entorno, pero no descarga los pesos. Declara las
rutas o identificadores de los modelos en `config/experiment.env`.

El modelo AWQ debe estar cuantizado previamente. `autoawq` no es necesario
para servir un checkpoint compatible ya construido.

## Comprobaciones

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import pynvml; pynvml.nvmlInit(); print('NVML OK')"
python -m unittest tests.test_tres_brazos
```

El preflight final se ejecuta dentro del launcher y cancela antes de inferir
si la configuración, el presupuesto o la telemetría no cumplen. No publiques
como válida una corrida con energía igual a cero.

## Auditar el experimento principal sin GPU

Si solo quieres comprobar las cifras y figuras publicadas:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 infera/audita_paquete_tres_brazos.py \
  --campaign experimentos/experimento_principal/evidencia \
  --archive experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.bin \
  --checksum experimentos/experimento_principal/paquete_preservado/experimento_principal_evidencia.tar.gz.sha256 \
  --reanalysis /tmp/infera_reanalysis
```

La auditoría no descarga modelos ni necesita CUDA. Instala
`requirements.txt` únicamente si también regenerarás las figuras.

## Credenciales

No guardes tokens de Hugging Face, claves o datos sensibles dentro de
`config/experiment.env`. Utiliza variables de entorno o el sistema de secretos
del proveedor.
