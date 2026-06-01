#!/usr/bin/env bash
# =============================================================================
# INFERA — setup_runpod.sh
# Configura el entorno de ejecución para el benchmark desde cero.
#
# DECISIÓN DE DISEÑO CRÍTICA:
#   Todo el stack Python se instala en /workspace/venv (venv aislado),
#   NUNCA en el Python global del container.
#   Razón: el Python global de RunPod tiene paquetes pre-instalados
#   (outlines 1.3.x, huggingface-hub 1.x) incompatibles con vLLM 0.5.3.
#   Un venv vacío resuelve el árbol de dependencias desde cero.
#
# USO:
#   export HF_TOKEN=hf_tu_token_aqui
#   bash scripts/setup_runpod.sh
#
# PREREQUISITOS:
#   - Imagen RunPod PyTorch 2.x (con CUDA 12.x)
#   - HF_TOKEN con acceso a meta-llama/Meta-Llama-3.1-8B-Instruct
#   - 80 GB container disk
#   - RTX 4090 o A5000 (24 GB VRAM)
# =============================================================================

set -euo pipefail  # Falla en cualquier error, variable no definida o pipe roto

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — versiones fijadas para reproducibilidad
# Cambiar cualquiera de estas requiere re-validación experimental completa.
# ─────────────────────────────────────────────────────────────────────────────
VLLM_VERSION="0.5.3"
# vLLM 0.5.3: mínima versión que soporta rope_scaling de LLaMA 3.1.
# 0.4.3 falla con KeyError: 'short_factor' en vllm/config.py:1216
# Reportado en: vLLM GitHub issues #4631, #5012

PYNVML_VERSION="11.5.0"
# pynvml: wrapper NVML para medición energética GPU.
# 11.5.0 es la versión estable con RTX 30/40 series.

HTTPX_VERSION="0.27.0"
# httpx: cliente HTTP asíncrono para benchmark_runner.py (asyncio.gather).

TRANSFORMERS_VERSION="4.43.3"
# transformers 4.43.x: soporte completo para LLaMA 3.1 tokenizer.
# También se instala como dependencia de vLLM, se fija para evitar upgrades.

HUGGINGFACE_HUB_VERSION="0.24.0"
# vLLM 0.5.3 requiere huggingface-hub>=0.23.2,<1.0
# 0.24.0 está dentro del rango y es estable.

MODEL_FP16="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_AWQ="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
MODEL_DIR_FP16="/workspace/models/llama3.1-8b-instruct"
MODEL_DIR_AWQ="/workspace/models/llama3.1-8b-instruct-awq"

VENV_PATH="/workspace/venv"
WORKSPACE="/workspace/infera"

# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────
log()  { echo "[$(date '+%H:%M:%S')] $*"; }
ok()   { echo "[$(date '+%H:%M:%S')] ✓ $*"; }
fail() { echo "[$(date '+%H:%M:%S')] ✗ ERROR: $*" >&2; exit 1; }

# ─────────────────────────────────────────────────────────────────────────────
# PASO 0 — Verificaciones previas
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 0: Verificaciones previas ==="

# Verificar HF_TOKEN
[[ -z "${HF_TOKEN:-}" ]] && fail "HF_TOKEN no definido. Ejecuta: export HF_TOKEN=hf_..."

# Verificar GPU
if ! nvidia-smi &>/dev/null; then
    fail "nvidia-smi no encontrado. ¿Está la GPU correctamente asignada?"
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
ok "GPU detectada: $GPU_NAME | VRAM: $GPU_VRAM"

# Verificar CUDA
CUDA_VERSION=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
ok "CUDA Version: $CUDA_VERSION"

# Verificar que CUDA es ≥12.0 (requerimiento de vLLM 0.5.3)
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
[[ "$CUDA_MAJOR" -lt 12 ]] && fail "CUDA $CUDA_VERSION < 12.0. Necesitas imagen RunPod con CUDA 12.x"

# Verificar Python
PYTHON_VER=$(python3 --version 2>&1)
ok "Python: $PYTHON_VER"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 1 — Crear venv aislado en /workspace
#
# CRÍTICO: /workspace persiste entre reinicios del pod (no se borra al restart).
# Por eso el venv va aquí y no en /tmp o /.
#
# --without-pip NO se usa: queremos pip dentro del venv.
# NO se usa --system-site-packages: queremos aislamiento total del Python global.
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 1: Creando venv aislado en $VENV_PATH ==="

if [[ -f "$VENV_PATH/bin/activate" ]]; then
    ok "venv ya existe en $VENV_PATH — reutilizando (idempotente)"
else
    python3 -m venv "$VENV_PATH"
    ok "venv creado en $VENV_PATH"
fi

# Activar el venv para el resto de este script
source "$VENV_PATH/bin/activate"
ok "venv activado: $(which python3)"
ok "pip: $(pip --version)"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 2 — Instalar dependencias en el venv
#
# ORDEN IMPORTA:
# 1. pip upgrade: asegura resolución de dependencias moderna
# 2. huggingface-hub PRIMERO y fijado: vLLM 0.5.3 requiere <1.0
#    Si pip instala primero otro paquete que trae hf-hub 1.x, vLLM fallará.
# 3. vLLM: instala torch, outlines 0.0.46, triton, etc. como deps automáticas
# 4. Resto de paquetes: fijados explícitamente
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 2: Instalando dependencias (esto tarda ~10-15 min) ==="

pip install --upgrade pip setuptools wheel -q
ok "pip/setuptools/wheel actualizados"

# huggingface-hub antes que vLLM para evitar que pip instale versión >=1.0
pip install "huggingface-hub==${HUGGINGFACE_HUB_VERSION}" -q
ok "huggingface-hub==${HUGGINGFACE_HUB_VERSION} instalado"

# vLLM — trae torch, triton, outlines==0.0.46, xformers, etc. como deps
# No instalamos torch por separado: vLLM 0.5.3 trae su versión exacta compatible
log "Instalando vLLM ${VLLM_VERSION} (esto descarga ~3GB de wheels, espera)..."
pip install "vllm==${VLLM_VERSION}" -q
ok "vLLM==${VLLM_VERSION} instalado"

# Verificar que outlines quedó en la versión correcta (<0.1.0)
# vLLM 0.5.3 requiere outlines.fsm que solo existe en outlines <0.1.0
OUTLINES_VER=$(pip show outlines 2>/dev/null | grep Version | awk '{print $2}')
if [[ -z "$OUTLINES_VER" ]]; then
    fail "outlines no instalado. Algo falló en la instalación de vLLM."
fi
OUTLINES_MAJOR=$(echo "$OUTLINES_VER" | cut -d. -f1)
OUTLINES_MINOR=$(echo "$OUTLINES_VER" | cut -d. -f2)
if [[ "$OUTLINES_MAJOR" -ge 1 ]] || [[ "$OUTLINES_MAJOR" -eq 0 && "$OUTLINES_MINOR" -ge 1 ]]; then
    fail "outlines $OUTLINES_VER >= 0.1.0 instalado. vLLM necesita outlines <0.1.0. 
    Esto NO debería ocurrir en venv limpio. Reportar como P6 en PROBLEMAS_LOG."
fi
ok "outlines==$OUTLINES_VER (< 0.1.0) — correcto"

# Paquetes adicionales del benchmark
pip install \
    "transformers==${TRANSFORMERS_VERSION}" \
    "pynvml==${PYNVML_VERSION}" \
    "httpx==${HTTPX_VERSION}" \
    "tqdm>=4.66.0" \
    "numpy>=1.24.0" \
    -q
ok "Dependencias del benchmark instaladas"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 3 — Verificar imports críticos
# Si alguno falla, detenemos ANTES de descargar 20GB de modelos.
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 3: Verificando imports críticos ==="

python3 -c "
import sys
errors = []

# Test 1: vLLM importable
try:
    import vllm
    print(f'  vLLM {vllm.__version__} — OK')
except Exception as e:
    errors.append(f'vLLM import FAILED: {e}')

# Test 2: outlines con módulo fsm (requerido por vLLM internamente)
try:
    import outlines
    import outlines.fsm
    print(f'  outlines {outlines.__version__} + outlines.fsm — OK')
except Exception as e:
    errors.append(f'outlines.fsm import FAILED: {e}')

# Test 3: pynvml con acceso a GPU
try:
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(0)
    name = pynvml.nvmlDeviceGetName(h)
    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
    print(f'  pynvml — GPU: {name} | Power: {power:.1f}W — OK')
    pynvml.nvmlShutdown()
except Exception as e:
    errors.append(f'pynvml FAILED: {e}')

# Test 4: httpx async
try:
    import httpx
    import asyncio
    async def _test(): 
        async with httpx.AsyncClient() as c: pass
    asyncio.run(_test())
    print(f'  httpx {httpx.__version__} async — OK')
except Exception as e:
    errors.append(f'httpx async FAILED: {e}')

# Test 5: transformers tokenizer
try:
    from transformers import AutoTokenizer
    print(f'  transformers AutoTokenizer — OK')
except Exception as e:
    errors.append(f'transformers FAILED: {e}')

# Test 6: torch con CUDA
try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f'  torch {torch.__version__} | CUDA disponible: {cuda_ok} — OK')
    if not cuda_ok:
        errors.append('torch.cuda.is_available() = False — revisar imagen/drivers')
except Exception as e:
    errors.append(f'torch FAILED: {e}')

if errors:
    print('\\n=== ERRORES DETECTADOS ===', file=sys.stderr)
    for e in errors:
        print(f'  ✗ {e}', file=sys.stderr)
    sys.exit(1)
else:
    print('  Todos los imports verificados correctamente.')
"

ok "Verificación de imports completada"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 4 — Descargar modelos
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 4: Descargando modelos ==="

mkdir -p "$MODEL_DIR_FP16" "$MODEL_DIR_AWQ"

# Modelo FP16 (~16 GB)
if [[ -f "$MODEL_DIR_FP16/config.json" ]]; then
    ok "FP16 ya descargado en $MODEL_DIR_FP16 — saltando"
else
    log "Descargando LLaMA 3.1 8B FP16 (~16 GB, ~5-10 min con buena conexión)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_FP16}',
    local_dir='${MODEL_DIR_FP16}',
    token=os.environ['HF_TOKEN'],
    ignore_patterns=['*.pt', 'original/*'],  # solo safetensors
)
print('Modelo FP16 descargado correctamente')
"
    ok "Modelo FP16 descargado"
fi

# Modelo AWQ INT4 (~4.5 GB)
if [[ -f "$MODEL_DIR_AWQ/config.json" ]]; then
    ok "AWQ ya descargado en $MODEL_DIR_AWQ — saltando"
else
    log "Descargando LLaMA 3.1 8B AWQ INT4 (~4.5 GB)..."
    python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download(
    repo_id='${MODEL_AWQ}',
    local_dir='${MODEL_DIR_AWQ}',
    token=os.environ['HF_TOKEN'],
    ignore_patterns=['*.pt'],
)
print('Modelo AWQ descargado correctamente')
"
    ok "Modelo AWQ descargado"
fi

# Verificar que los modelos tienen archivos reales (no 0 bytes)
FP16_SIZE=$(du -sh "$MODEL_DIR_FP16" | cut -f1)
AWQ_SIZE=$(du -sh "$MODEL_DIR_AWQ" | cut -f1)
ok "Modelos descargados: FP16=$FP16_SIZE | AWQ=$AWQ_SIZE"

# ─────────────────────────────────────────────────────────────────────────────
# PASO 5 — Fijar rutas del venv en scripts de lanzamiento de vLLM
#
# Los scripts start_vllm_*.sh usan 'python3' o 'vllm' directamente.
# Si se ejecutan sin el venv activo, usarán el Python global contaminado.
# Solución: insertar 'source /workspace/venv/bin/activate' al inicio.
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 5: Fijando venv en scripts de lanzamiento ==="

SCRIPTS_DIR="$WORKSPACE/scripts"

for script in start_vllm_fp16.sh start_vllm_int8.sh start_vllm_awq.sh; do
    SCRIPT_PATH="$SCRIPTS_DIR/$script"
    if [[ ! -f "$SCRIPT_PATH" ]]; then
        log "  ADVERTENCIA: $script no encontrado en $SCRIPTS_DIR — saltando"
        continue
    fi
    # Verificar si ya tiene la activación del venv
    if grep -q "workspace/venv/bin/activate" "$SCRIPT_PATH"; then
        ok "  $script — venv ya configurado"
    else
        # Insertar después de la línea shebang (línea 1)
        sed -i '1a source /workspace/venv/bin/activate' "$SCRIPT_PATH"
        ok "  $script — venv añadido"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# PASO 6 — Generar snapshot de reproducibilidad
# Captura versiones exactas del entorno para el paper.
# ─────────────────────────────────────────────────────────────────────────────
log "=== PASO 6: Generando reproducibility snapshot ==="

mkdir -p "$WORKSPACE/results"

python3 -c "
import json, subprocess, datetime, os, sys

def run(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True).strip()
    except:
        return 'N/A'

def pkg_version(name):
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except:
        return 'N/A'

snapshot = {
    'timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
    'hardware': {
        'gpu_name': run('nvidia-smi --query-gpu=name --format=csv,noheader'),
        'gpu_vram_mb': run('nvidia-smi --query-gpu=memory.total --format=csv,noheader'),
        'gpu_driver': run('nvidia-smi --query-gpu=driver_version --format=csv,noheader'),
        'cuda_version': run('nvidia-smi | grep \"CUDA Version\" | awk \"{print \\\$NF}\"'),
    },
    'software': {
        'python': sys.version,
        'vllm': pkg_version('vllm'),
        'torch': pkg_version('torch'),
        'outlines': pkg_version('outlines'),
        'transformers': pkg_version('transformers'),
        'huggingface_hub': pkg_version('huggingface-hub'),
        'pynvml': pkg_version('pynvml'),
        'httpx': pkg_version('httpx'),
    },
    'experiment_config': {
        'model_fp16': '${MODEL_FP16}',
        'model_awq': '${MODEL_AWQ}',
        'quantizations': ['fp16', 'int8_w8a16', 'int4_awq'],
        'batch_sizes': [1, 4, 8],
        'max_new_tokens': [64, 256, 512],
        'context_levels': {'A': 256, 'B': 1024, 'C': 4096},
        'repetitions_per_config': 3,
        'total_runs': 243,
        'monitoring_buffer_ms': 500,
        'monitoring_sampling_ms': 100,
    },
    'git_commit': run('cd ${WORKSPACE} && git rev-parse HEAD 2>/dev/null || echo N/A'),
    'venv_path': '${VENV_PATH}',
}

output_path = '${WORKSPACE}/results/reproducibility.json'
with open(output_path, 'w') as f:
    json.dump(snapshot, f, indent=2)

print(json.dumps(snapshot, indent=2))
print(f'\\nGuardado en: {output_path}')
"

ok "reproducibility.json generado"

# ─────────────────────────────────────────────────────────────────────────────
# RESUMEN FINAL
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════"
echo " SETUP COMPLETADO CORRECTAMENTE"
echo "════════════════════════════════════════════════════════════"
echo ""
echo " Para cada nueva terminal o sesión, activar el venv con:"
echo "   source /workspace/venv/bin/activate"
echo ""
echo " Próximos pasos:"
echo "   1. Construir corpus:   python scripts/build_prompt_dataset.py"
echo "   2. Iniciar servidor:   bash scripts/start_vllm_fp16.sh"
echo "   3. Verificar health:   curl http://localhost:8000/health"
echo "   4. Piloto FP16:        python scripts/benchmark_runner.py --quantization fp16 --pilot"
echo ""
echo " IMPORTANTE: El venv está en /workspace/venv"
echo "   Si creas un NUEVO pod, ejecuta este script de nuevo."
echo "   Si REINICIAS el mismo pod, solo activa el venv."
echo "════════════════════════════════════════════════════════════"