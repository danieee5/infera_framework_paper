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
python config/validate_configuration.py
```

El runner cancela si NVML no puede inicializarse. No publiques como válida una
corrida con energía igual a cero.

## Analizar el conjunto de referencia sin GPU

Si solo quieres comprobar las cifras y figuras publicadas:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd infera
python analyze_results.py
```

Esta instalación no descarga modelos ni necesita CUDA.

## Credenciales

No guardes tokens de Hugging Face, claves o datos sensibles dentro de
`config/experiment.env`. Utiliza variables de entorno o el sistema de secretos
del proveedor.
