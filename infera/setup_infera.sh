#!/usr/bin/env bash
# setup_infera.sh
# Entorno de ejecución completa para INFERA en Linux con GPU NVIDIA.
# El stack del conjunto de referencia se validó en RTX 4090, CUDA 12.1 y
# Python 3.10.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INFERA_RUNTIME="${INFERA_RUNTIME:-${SCRIPT_DIR}/.runtime}"

# ---------------------------------------------------------------------------
# 0. Caché y entorno. Para un volumen persistente, define INFERA_RUNTIME antes
#    de ejecutar este script.
# ---------------------------------------------------------------------------
export HF_HOME="${HF_HOME:-${INFERA_RUNTIME}/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# 1. Entorno aislado. No instala sobre el Python global del contenedor.
# ---------------------------------------------------------------------------
VENV="${INFERA_VENV:-${INFERA_RUNTIME}/venv}"
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. Dependencias fijadas para el entorno de referencia.
#    torch debe instalarse con el indice cu121 ANTES que vLLM.
# ---------------------------------------------------------------------------
python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "${REPO_ROOT}/requirements-gpu.txt"

# NOTA: NO instalamos el paquete `autoawq`. Solo se necesita para CUANTIZAR
# modelos uno mismo; para SERVIR un modelo AWQ ya cuantizado, vLLM trae soporte
# nativo (--quantization awq). Ademas autoawq==0.2.5 arrastra un resolver que
# intenta subir a torch 2.12 + CUDA 13 (varios GB extra), lo cual revienta la
# cuota del volumen de red sin aportar nada que necesitemos. Se omite a proposito.

# ---------------------------------------------------------------------------
# Compatibilidad con pyairports:
# outlines 0.0.46 (dependencia de vLLM 0.5.3) importa AIRPORT_LIST de pyairports.
# El repo original fue eliminado de GitHub; el paquete de PyPI viene vacío
# (solo dist-info, sin .py). vLLM no usa gramaticas de aeropuertos -> stub vacio.
# ---------------------------------------------------------------------------
echo "Aplicando fix de pyairports..."
python -m pip install "pyairports==0.0.1" -q 2>/dev/null || true
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/pyairports"

cat > "$SITE/pyairports/__init__.py" << 'PYEOF'
"""pyairports stub para outlines 0.0.46. Repo original eliminado de GitHub."""
from pyairports.airports import Airports, AirportNotFoundException
PYEOF

cat > "$SITE/pyairports/airports.py" << 'PYEOF'
AIRPORT_LIST = []

class AirportNotFoundException(Exception):
    pass

class Airport:
    def __init__(self, **kwargs):
        for k, v in kwargs.items(): setattr(self, k, v)

class Airports:
    def __init__(self): self.airports = {}
    def airport_iata(self, iata): raise AirportNotFoundException(iata)
    def other_iata(self, iata): raise AirportNotFoundException(iata)
    def lookup(self, iata): raise AirportNotFoundException(iata)
    def airport_city(self, city, country=None): return []
PYEOF
echo "[OK] stub de pyairports en $SITE/pyairports"

echo
echo "[OK] Entorno listo en $VENV"
echo "[OK] HF cache en $HF_HOME"
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "| CUDA:", torch.version.cuda, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
print("vllm :", vllm.__version__)
try:
    import pynvml; pynvml.nvmlInit(); print("NVML : OK")
except Exception as e:
    print("NVML : FALLO ->", e)
PY

cat <<'NOTE'

------------------------------------------------------------------
COMO SERVIR EL MODELO (en una terminal aparte, una config a la vez):

# FP16
python -m vllm.entrypoints.openai.api_server \
  --model /models/llama3.1-8b-instruct \
  --dtype float16 --max-model-len 8192 --port 8000

# AWQ INT4
python -m vllm.entrypoints.openai.api_server \
  --model /models/llama3.1-8b-instruct-awq \
  --quantization awq --dtype float16 --max-model-len 8192 --port 8000

# INT8 (bitsandbytes, opcional)
python -m vllm.entrypoints.openai.api_server \
  --model /models/llama3.1-8b-instruct \
  --quantization bitsandbytes --load-format bitsandbytes \
  --max-model-len 8192 --port 8000

Configura las rutas reales en config/experiment.env antes de ejecutar
overnight.sh. Los valores /models/... de arriba son solo ejemplos.
------------------------------------------------------------------
NOTE
