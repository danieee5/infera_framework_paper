#!/usr/bin/env bash
# =============================================================================
# INFERA — setup_exp2.sh   (ADDENDUM de setup_runpod_v3.sh, NO lo reemplaza)
#
# Prepara un pod ya configurado (entorno + modelos) para el Experimento 2
# multi-turno. Corre DESPUES de setup_runpod_v3.sh en cada pod.
#
# Hace 3 cosas:
#   1. Crea data/multiturn/ y verifica que conversation_flow.json este presente.
#   2. Captura la POTENCIA EN REPOSO de la GPU (60 s) -> calibracion entre pods.
#      Necesario solo si corres en PARALELO (un esquema por pod): permite reportar
#      el offset tarjeta-a-tarjeta que afecta unicamente la comparacion AWQ vs FP16
#      (H3). H1, H2 y H4 son intra-tarjeta y no se ven afectadas.
#   3. Imprime la secuencia exacta de ejecucion.
#
# USO:
#   cd /workspace/infera
#   bash scripts/setup_exp2.sh
# =============================================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace/infera}"
VENV_PATH="${VENV_PATH:-/workspace/venv}"
DATA_DIR="$WORKSPACE/data/multiturn"
RES_DIR="$WORKSPACE/results/multiturn"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
ok()  { echo "[$(date '+%H:%M:%S')] ✓ $*"; }

source "$VENV_PATH/bin/activate"

# ─── 1. Estructura de datos ─────────────────────────────────────────────────
log "=== Preparando data/multiturn ==="
mkdir -p "$DATA_DIR" "$RES_DIR"

if [[ ! -f "$DATA_DIR/conversation_flow.json" ]]; then
    echo "  ✗ Falta $DATA_DIR/conversation_flow.json"
    echo "    Copialo desde el repo (scripts/.. o data/multiturn/) a esa ruta y reintenta."
    exit 1
fi
ok "conversation_flow.json presente"

# ─── 2. Calibracion: potencia en reposo (60 s) ──────────────────────────────
log "=== Capturando potencia en reposo (60 s) para calibracion entre pods ==="
HOST="$(hostname)"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python3 - "$RES_DIR" "$HOST" "$GPU_NAME" << 'PYEOF'
import sys, json, time, subprocess, statistics
res_dir, host, gpu = sys.argv[1], sys.argv[2], sys.argv[3]
samples = []
t_end = time.time() + 60
while time.time() < t_end:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
        text=True).strip().splitlines()[0]
    samples.append(float(out))
    time.sleep(0.5)
rec = {
    "host": host, "gpu": gpu, "n": len(samples),
    "idle_power_w_mean": round(statistics.mean(samples), 2),
    "idle_power_w_std":  round(statistics.pstdev(samples), 2),
    "idle_power_w_min":  round(min(samples), 2),
    "idle_power_w_max":  round(max(samples), 2),
}
p = f"{res_dir}/idle_power_{host}.json"
open(p, "w").write(json.dumps(rec, indent=2))
print(f"  idle power: {rec['idle_power_w_mean']} ± {rec['idle_power_w_std']} W  ({gpu})")
print(f"  guardado: {p}")
PYEOF
ok "Calibracion de reposo capturada"

# ─── 3. Secuencia de ejecucion ──────────────────────────────────────────────
cat << 'TXT'

════════════════════════════════════════════════════════════════════
  SETUP EXP2 LISTO. Secuencia de ejecucion:
════════════════════════════════════════════════════════════════════

  FASE 1 — generar la historia fija (SOLO en UN pod, con FP16):
    Terminal A:  bash scripts/start_vllm_fp16.sh
    Terminal B:  python scripts/build_multiturn_conversation.py
       -> produce data/multiturn/conversation_history.json
       -> valida la envolvente de tokens (T1 1.5-2k, T7 3.5-4.5k)

    >>> Copia conversation_history.json IDENTICO a los demas pods <<<
        (commit al repo, o scp). Los 3 esquemas DEBEN usar la misma historia.

  FASE 2 — medir energia (un esquema por pod, en paralelo):
    Pod FP16:  bash scripts/start_vllm_fp16.sh   &&  python scripts/multiturn_runner.py --quantization fp16
    Pod INT8:  bash scripts/start_vllm_int8.sh   &&  python scripts/multiturn_runner.py --quantization int8_w8a16
    Pod AWQ :  bash scripts/start_vllm_awq.sh    &&  python scripts/multiturn_runner.py --quantization int4_awq
       (servidor en una terminal, runner en otra; cada pod ~1.6 h)

  ANALISIS (tras juntar results/multiturn/* de los 3 pods):
    python scripts/multiturn_analysis.py --results-dir results/multiturn/

  NOTA: NO agregues --enable-prefix-caching a ningun start_vllm_*.sh.
        EXP2 mide el regimen de re-prefill completo a proposito.
════════════════════════════════════════════════════════════════════
TXT
ok "Hecho."
