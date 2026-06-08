#!/usr/bin/env bash
# setup_infera.sh
# Entorno reproducible para INFERA Protocolo C en RunPod (RTX 4090, CUDA 12.1).
# Aplica las lecciones de EXP1 para evitar cascadas de dependencias.
set -euo pipefail

# ---------------------------------------------------------------------------
# 0. Variables de entorno: redirigir cache HF al volumen de red persistente
#    (evita duplicar disco y agotar el contenedor).
# ---------------------------------------------------------------------------
export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache/hub
mkdir -p "$HF_HOME"

# ---------------------------------------------------------------------------
# 1. venv LIMPIO en el volumen persistente (NUNCA instalar sobre el base del
#    contenedor: en EXP1 eso causo cascadas de dependencias).
# ---------------------------------------------------------------------------
VENV=/workspace/venv
if [ ! -d "$VENV" ]; then
  python3 -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip

# ---------------------------------------------------------------------------
# 2. Dependencias PINNED (stack validado en EXP1).
#    torch debe instalarse con el indice cu121 ANTES que vLLM.
# ---------------------------------------------------------------------------
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.5.3
pip install "transformers>=4.43,<4.46"   # compatible con plantilla chat LLaMA 3.1
pip install nvidia-ml-py==12.560.30       # provee 'pynvml' (NVML)
pip install requests pandas matplotlib

# autoawq solo si se va a servir el modelo AWQ
pip install autoawq==0.2.5 || echo "WARNING: autoawq no instalado (omitir si no usas AWQ)"

# ---------------------------------------------------------------------------
# Fix pyairports (lección documentada en EXP1, scripts/setup_runpod.sh):
# outlines 0.0.46 (dependencia de vLLM 0.5.3) importa AIRPORT_LIST de pyairports.
# El repo original fue eliminado de GitHub; el paquete de PyPI viene vacío
# (solo dist-info, sin .py). vLLM no usa gramaticas de aeropuertos -> stub vacio.
# ---------------------------------------------------------------------------
echo "Aplicando fix de pyairports..."
pip install "pyairports==0.0.1" -q 2>/dev/null || true
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

NOTA: edita archivos en /workspace con Python open().read()/write(), NUNCA con
'sed -i' (falla en silencio si la cuota del filesystem esta excedida).
------------------------------------------------------------------
NOTE
