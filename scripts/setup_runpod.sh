#!/usr/bin/env bash
# =============================================================================
# INFERA — setup_runpod.sh v3
# Configura el entorno de ejecución para el benchmark desde cero.
#
# LECCIONES SESIÓN 2 (01-Jun-2026):
#   1. pyairports: dep transitiva rota de outlines 0.0.46. PyPI 0.0.1 instala
#      solo dist-info sin archivos Python. Fix: crear módulo manualmente.
#   2. /workspace (red RunPod): tiene quota estricta ~20-25 GB.
#      venv (12 GB) + intentos de modelo la llenan rápido.
#   3. HuggingFace snapshot_download descarga a ~/.cache Y a local_dir
#      simultáneamente → doble uso de disco. Fix: local_dir_use_symlinks=False
#      + HF_HUB_CACHE=/workspace/.hf_cache (redirige cache al volumen de red).
#   4. Modelos van al container disk (/models/) que tiene 77+ GB libres.
#      venv va al volumen de red (/workspace/venv) que persiste entre reinicios.
#
# DISTRIBUCIÓN DE DISCO:
#   /workspace (red, persiste): venv (~12 GB), HF cache (~22 GB), results
#   / (container, 80 GB):      OS (3.5 GB), modelos (~22 GB) → total ~26 GB
#
# USO:
#   export HF_TOKEN=hf_tu_token_aqui
#   bash scripts/setup_runpod.sh
# =============================================================================

set -euo pipefail

VLLM_VERSION="0.5.3"
PYNVML_VERSION="11.5.0"
HTTPX_VERSION="0.27.0"
TRANSFORMERS_VERSION="4.43.3"
HUGGINGFACE_HUB_VERSION="0.24.0"

# MODELOS EN CONTAINER DISK — no en /workspace para evitar quota
MODEL_FP16="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MODEL_DIR_FP16="/models/llama3.1-8b-instruct"
MODEL_DIR_AWQ="/models/llama3.1-8b-instruct-awq"

# VENV EN VOLUMEN DE RED — persiste entre reinicios del pod
VENV_PATH="/workspace/venv"
WORKSPACE="/workspace/infera"

# HF CACHE EN VOLUMEN DE RED — evita duplicar modelos en container disk
export HF_HUB_CACHE="/workspace/.hf_cache"
export TMPDIR="/workspace/tmp"

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
fail() { echo "[$(date '+%H:%M:%S')] ✗ ERROR: $*" >&2; exit 1; }

# ─── PASO 0: Verificaciones ──────────────────────────────────────────────────
log "=== PASO 0: Verificaciones previas ==="
[[ -z "${HF_TOKEN:-}" ]] && fail "HF_TOKEN no definido."

nvidia-smi &>/dev/null || fail "nvidia-smi no encontrado."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU: $GPU_NAME | VRAM: $GPU_VRAM"

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $NF}')
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
[[ "$CUDA_MAJOR" -lt 12 ]] && fail "CUDA $CUDA_VERSION < 12.0"
ok "CUDA: $CUDA_VERSION"

# Verificar espacio en container disk
DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
[[ "$DISK_FREE" -lt 40 ]] && fail "Container disk: solo ${DISK_FREE}GB libres. Se necesitan ≥40 GB."
ok "Container disk: ${DISK_FREE}GB libres"

# Crear directorios de trabajo en volumen de red
mkdir -p "$HF_HUB_CACHE" "$TMPDIR"

# ─── PASO 1: venv aislado ────────────────────────────────────────────────────
log "=== PASO 1: Creando venv en $VENV_PATH ==="
if [[ -f "$VENV_PATH/bin/activate" ]]; then
    ok "venv ya existe — reutilizando"
else
    python3 -m venv "$VENV_PATH"
    ok "venv creado"
fi
source "$VENV_PATH/bin/activate"
ok "venv activo: $(which python3)"

# ─── PASO 2: Dependencias ────────────────────────────────────────────────────
log "=== PASO 2: Instalando dependencias (~15 min) ==="

pip install --upgrade pip setuptools wheel -q
ok "pip/setuptools/wheel actualizados"

pip install "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" -q
ok "huggingface-hub==${HUGGINGFACE_HUB_VERSION}"

log "Instalando vLLM ${VLLM_VERSION} (~3 GB de wheels)..."
pip install "vllm==${VLLM_VERSION}" -q
ok "vLLM==${VLLM_VERSION}"

# Limpiar cache de pip para liberar espacio en /workspace
pip cache purge 2>/dev/null || true
ok "pip cache limpiado"

# Verificar outlines <0.1.0
OUTLINES_VER=$(pip show outlines 2>/dev/null | grep Version | awk '{print $2}')
[[ -z "$OUTLINES_VER" ]] && fail "outlines no instalado"
ok "outlines==$OUTLINES_VER"

# ── Fix pyairports ─────────────────────────────────────────────────────────
# PROBLEMA DOCUMENTADO: outlines 0.0.46 importa AIRPORT_LIST de pyairports.
# Repo original eliminado de GitHub. PyPI 0.0.1 tiene dist-info sin archivos Python.
# vLLM no usa gramáticas de aeropuertos. AIRPORT_LIST=[] es correcto.
log "Aplicando fix de pyairports..."
pip install "pyairports==0.0.1" -q 2>/dev/null || true

SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE/pyairports"

cat > "$SITE/pyairports/__init__.py" << 'PYEOF'
"""pyairports stub para outlines 0.0.46. Repo original eliminado de GitHub."""
from pyairports.airports import Airports, AirportNotFoundException
PYEOF

cat > "$SITE/pyairports/airports.py" << 'PYEOF'
# pyairports stub — interfaz exacta que outlines/types/airports.py necesita.
# AIRPORT_LIST es lo único que outlines usa. Lista vacía es correcto para vLLM.
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

python3 -c "from pyairports.airports import AIRPORT_LIST; import outlines.fsm" \
    && ok "pyairports + outlines.fsm OK" \
    || fail "pyairports fix falló"

# Paquetes del benchmark
pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "pynvml==${PYNVML_VERSION}" \
    "httpx==${HTTPX_VERSION}" \
    "tqdm>=4.66.0" \
    "numpy>=1.24.0" \
    -q
ok "Dependencias del benchmark instaladas"

# ─── PASO 3: Verificar imports ───────────────────────────────────────────────
log "=== PASO 3: Verificando imports ==="
python3 -c "
import sys; errors = []

tests = [
    ('vllm',            lambda: __import__('vllm')),
    ('outlines.fsm',    lambda: __import__('outlines.fsm', fromlist=['fsm'])),
    ('pyairports',      lambda: __import__('pyairports.airports', fromlist=['AIRPORT_LIST'])),
    ('pynvml+GPU',      lambda: [__import__('pynvml').nvmlInit(),
                                  __import__('pynvml').nvmlDeviceGetPowerUsage(
                                  __import__('pynvml').nvmlDeviceGetHandleByIndex(0))]),
    ('httpx async',     lambda: __import__('asyncio').run(
                                  __import__('httpx').AsyncClient().__aenter__())),
    ('transformers',    lambda: __import__('transformers').AutoTokenizer),
    ('torch+CUDA',      lambda: __import__('torch').cuda.is_available() or True),
]

for name, fn in tests:
    try:
        fn()
        print(f'  {name} — OK')
    except Exception as e:
        errors.append(f'{name}: {e}')
        print(f'  {name} — FAIL: {e}', file=sys.stderr)

if errors:
    sys.exit(1)
"
ok "Todos los imports verificados"

# ─── PASO 4: Descargar modelos ───────────────────────────────────────────────
log "=== PASO 4: Descargando modelos ==="
mkdir -p "$MODEL_DIR_FP16" "$MODEL_DIR_AWQ"

if [[ -f "$MODEL_DIR_FP16/config.json" ]] && \
   [[ $(ls "$MODEL_DIR_FP16"/*.safetensors 2>/dev/null | wc -l) -ge 4 ]]; then
    ok "FP16 ya descargado ($(du -sh $MODEL_DIR_FP16 | cut -f1))"
else
    log "Descargando FP16 (~16 GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_FP16}',
    local_dir='${MODEL_DIR_FP16}',
    token=os.environ['HF_TOKEN'],
    ignore_patterns=['*.pt', 'original/*'],
    local_dir_use_symlinks=False,
    max_workers=1,
)
print('FP16 OK')
"
    ok "FP16 descargado: $(du -sh $MODEL_DIR_FP16 | cut -f1)"
fi

if [[ -f "$MODEL_DIR_AWQ/config.json" ]] && \
   [[ $(ls "$MODEL_DIR_AWQ"/*.safetensors 2>/dev/null | wc -l) -ge 1 ]]; then
    ok "AWQ ya descargado ($(du -sh $MODEL_DIR_AWQ | cut -f1))"
else
    log "Descargando AWQ (~5.7 GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_AWQ}',
    local_dir='${MODEL_DIR_AWQ}',
    token=os.environ['HF_TOKEN'],
    ignore_patterns=['*.pt'],
    local_dir_use_symlinks=False,
    max_workers=1,
)
print('AWQ OK')
"
    ok "AWQ descargado: $(du -sh $MODEL_DIR_AWQ | cut -f1)"
fi

# Verificar uso de disco al final de descargas
DISK_USED=$(df -BG / | tail -1 | awk '{print $3}' | tr -d 'G')
DISK_FREE=$(df -BG / | tail -1 | awk '{print $4}' | tr -d 'G')
ok "Container disk: ${DISK_USED}GB usados, ${DISK_FREE}GB libres"

# ─── PASO 5: Activar venv en scripts de lanzamiento ─────────────────────────
log "=== PASO 5: Configurando scripts de lanzamiento ==="
for script in start_vllm_fp16.sh start_vllm_int8.sh start_vllm_awq.sh; do
    SPATH="$WORKSPACE/scripts/$script"
    [[ ! -f "$SPATH" ]] && { log "  WARN: $script no encontrado"; continue; }

    python3 -c "
content = open('$SPATH').read()
content = content.replace('/workspace/models/', '/models/')
if 'workspace/venv/bin/activate' not in content:
    lines = content.split('\n')
    lines.insert(1, 'source /workspace/venv/bin/activate')
    content = '\n'.join(lines)
open('$SPATH', 'w').write(content)
"
    ok "  $script actualizado"
done

# ─── PASO 6: Snapshot de reproducibilidad ───────────────────────────────────
log "=== PASO 6: Generando reproducibility.json ==="
mkdir -p "$WORKSPACE/results"
python3 -c "
import json, datetime, sys
try:
    import importlib.metadata as im
    import subprocess
    def v(p):
        try: return im.version(p)
        except: return 'N/A'
    def run(c):
        try: return subprocess.check_output(c, shell=True, text=True).strip()
        except: return 'N/A'
    snap = {
        'timestamp': datetime.datetime.utcnow().isoformat()+'Z',
        'hardware': {
            'gpu': run('nvidia-smi --query-gpu=name --format=csv,noheader'),
            'vram': run('nvidia-smi --query-gpu=memory.total --format=csv,noheader'),
            'cuda': run(\"nvidia-smi | grep 'CUDA Version' | awk '{print \\\$NF}'\"),
        },
        'software': {
            'python': sys.version.split()[0],
            'vllm': v('vllm'), 'torch': v('torch'),
            'outlines': v('outlines'), 'transformers': v('transformers'),
            'pynvml': v('pynvml'), 'httpx': v('httpx'),
        },
        'model_paths': {
            'fp16': '${MODEL_DIR_FP16}',
            'awq': '${MODEL_DIR_AWQ}',
        },
        'git': run('cd ${WORKSPACE} && git rev-parse HEAD 2>/dev/null || echo N/A'),
    }
    p = '${WORKSPACE}/results/reproducibility.json'
    open(p,'w').write(json.dumps(snap, indent=2))
    print(f'Guardado: {p}')
except Exception as e:
    print(f'WARN: reproducibility.json no generado: {e}')
"

echo ""
echo "════════════════════════════════════════════════════════════"
echo " SETUP COMPLETADO"
echo "════════════════════════════════════════════════════════════"
echo ""
echo " Activar venv en cada terminal:"
echo "   source /workspace/venv/bin/activate"
echo ""
echo " Modelos en:"
echo "   FP16: $MODEL_DIR_FP16"
echo "   AWQ:  $MODEL_DIR_AWQ"
echo ""
echo " Próximos pasos:"
echo "   python scripts/build_prompt_dataset.py --model-dir $MODEL_DIR_FP16 --verify-only"
echo "   bash scripts/start_vllm_fp16.sh"
echo "   python scripts/benchmark_runner.py --quantization fp16 --pilot"
echo "════════════════════════════════════════════════════════════"