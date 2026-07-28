#!/usr/bin/env bash
# Campaña científica fija: 3 brazos × 2 precisiones × 3 repeticiones.
set -Eeuo pipefail
cd "$(dirname "$0")"

ENV_FILE="${ENV_FILE:-config/experiment.env}"
[[ -f "$ENV_FILE" ]] || {
  echo "Falta $ENV_FILE; créalo desde config/reference/experiment.env.example" >&2
  exit 2
}
# shellcheck disable=SC1090
source "$ENV_FILE"

: "${FP16_MODEL:?define FP16_MODEL en $ENV_FILE}"
: "${AWQ_MODEL:?define AWQ_MODEL en $ENV_FILE}"
SESSION="${SESSION:-config/reference/session_tasks.json}"
KB_DIR="${KB_DIR:-kb}"
THRESH="${THRESH:-4500}"
PARES="${PARES_CONSERVADOS:-4}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
EXPECTED_TASKS="${EXPECTED_TASKS:-29}"
REPS="${REPS:-3}"
REPOSO="${SEGUNDOS_REPOSO:-30}"
PORT="${PORT:-8000}"
COOL="${COOL:-120}"
WARMUP="${WARMUP:-5}"
POST_WARMUP_SETTLE_S="${POST_WARMUP_SETTLE_S:-30}"
REQUEST_TIMEOUT_S="${REQUEST_TIMEOUT_S:-600}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-600}"
SHUTDOWN_TIMEOUT_S="${SHUTDOWN_TIMEOUT_S:-60}"
SERVER_START_ATTEMPTS="${SERVER_START_ATTEMPTS:-2}"
URL="http://127.0.0.1:${PORT}/v1/chat/completions"

# No permitir que un experiment.env olvidado cambie silenciosamente el
# protocolo que se pretende reportar hoy.
[[ "$THRESH" == "4500" ]] || { echo "THRESH debe ser 4500" >&2; exit 2; }
[[ "$PARES" == "4" ]] || { echo "PARES_CONSERVADOS debe ser 4" >&2; exit 2; }
[[ "$MAX_MODEL_LEN" == "8192" ]] || { echo "MAX_MODEL_LEN debe ser 8192" >&2; exit 2; }
[[ "$EXPECTED_TASKS" == "29" ]] || { echo "EXPECTED_TASKS debe ser 29" >&2; exit 2; }
[[ "$REPS" == "3" ]] || { echo "REPS debe ser 3" >&2; exit 2; }
[[ "$REPOSO" == "30" ]] || { echo "SEGUNDOS_REPOSO debe ser 30" >&2; exit 2; }
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || {
  echo "PORT debe ser un entero entre 1 y 65535" >&2
  exit 2
}
[[ "$WARMUP" == "5" ]] || {
  echo "WARMUP debe ser 5" >&2
  exit 2
}
[[ "$POST_WARMUP_SETTLE_S" == "30" ]] || {
  echo "POST_WARMUP_SETTLE_S debe ser 30" >&2
  exit 2
}
[[ "$STARTUP_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "STARTUP_TIMEOUT_S debe ser un entero positivo" >&2
  exit 2
}
[[ "$SHUTDOWN_TIMEOUT_S" =~ ^[1-9][0-9]*$ ]] || {
  echo "SHUTDOWN_TIMEOUT_S debe ser un entero positivo" >&2
  exit 2
}
[[ "$SERVER_START_ATTEMPTS" == "2" ]] || {
  echo "SERVER_START_ATTEMPTS debe ser 2" >&2
  exit 2
}
[[ "$REQUEST_TIMEOUT_S" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "REQUEST_TIMEOUT_S debe ser un número positivo" >&2
  exit 2
}
python -c 'import sys; raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)' \
  "$REQUEST_TIMEOUT_S" || {
  echo "REQUEST_TIMEOUT_S debe ser mayor que cero" >&2
  exit 2
}
[[ "$REQUEST_TIMEOUT_S" == "600" ]] || {
  echo "REQUEST_TIMEOUT_S debe ser 600" >&2
  exit 2
}
[[ "$COOL" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
  echo "COOL debe ser un número no negativo" >&2
  exit 2
}
[[ "$COOL" == "120" ]] || {
  echo "COOL debe ser 120" >&2
  exit 2
}

for dependency in python curl setsid nvidia-smi ps tee sed; do
  command -v "$dependency" >/dev/null || {
    echo "Falta la dependencia ejecutable: $dependency" >&2
    exit 2
  }
done
python -c 'import requests, transformers, vllm, pynvml' || {
  echo "Faltan dependencias Python de requirements-gpu.txt" >&2
  exit 2
}
python -m unittest tests.test_tres_brazos >/dev/null || {
  echo "Fallaron las pruebas CPU de la campaña; no se levantará vLLM" >&2
  exit 2
}
[[ -f "$SESSION" ]] || { echo "No existe SESSION=$SESSION" >&2; exit 2; }
[[ -d "$KB_DIR" ]] || { echo "No existe KB_DIR=$KB_DIR" >&2; exit 2; }

GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
[[ "$GPU_COUNT" == "1" ]] || {
  echo "Se exige exactamente una GPU NVIDIA visible; se detectaron $GPU_COUNT" >&2
  exit 2
}
GPU_NAME="$(
  nvidia-smi \
    --query-gpu=name \
    --format=csv,noheader |
    sed -n '1p'
)"
[[ "$GPU_NAME" == *"RTX 4090"* ]] || {
  echo "Esta campaña exige una RTX 4090; se detectó: $GPU_NAME" >&2
  exit 2
}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "0" ]]; then
  echo "CUDA_VISIBLE_DEVICES debe estar vacío o ser 0 para alinear CUDA y NVML" >&2
  exit 2
fi
COMPUTE_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | tr -d ' ')"
[[ -z "$COMPUTE_PIDS" ]] || {
  echo "La GPU ya tiene procesos de cómputo ($COMPUTE_PIDS); no es exclusiva" >&2
  exit 2
}
python -c 'import pynvml; pynvml.nvmlInit(); assert pynvml.nvmlDeviceGetCount() == 1; pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(0)); pynvml.nvmlShutdown()' || {
  echo "NVML no puede medir potencia en la GPU 0" >&2
  exit 2
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTPUT_ROOT:-results/runs/tres_brazos_${STAMP}}"
[[ ! -e "$OUT" ]] || {
  echo "La salida ya existe y no se sobrescribirá: $OUT" >&2
  exit 2
}
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT"
RAW="$OUT/raw"
LOGS="$OUT/logs"
ANALYSIS="$OUT/analisis"
PREFLIGHT="$OUT/preflight.json"
CAMPAIGN_MANIFEST="$OUT/manifiesto_campana.json"
mkdir "$RAW" "$LOGS"
exec > >(tee -a "$LOGS/campana.log") 2>&1
echo "Campaña inicializada: $OUT"

if python preflight_campana_tres_brazos.py \
    --fp16-tokenizer "$FP16_MODEL" \
    --awq-tokenizer "$AWQ_MODEL" \
    --tasks "$SESSION" \
    --kb-dir "$KB_DIR" \
    --out "$PREFLIGHT" \
    --threshold "$THRESH" \
    --pairs "$PARES" \
    --max-model-len "$MAX_MODEL_LEN" \
    --expected-tasks "$EXPECTED_TASKS" \
    --baseline-seconds "$REPOSO" \
    --settle-seconds "$POST_WARMUP_SETTLE_S" \
    --warmup-count "$WARMUP" \
    --cooldown-seconds "$COOL" \
    --request-timeout-seconds "$REQUEST_TIMEOUT_S" \
    --server-start-attempts "$SERVER_START_ATTEMPTS" \
    >"$LOGS/preflight.log" 2>&1; then
  :
else
  preflight_rc=$?
  python escribe_manifiesto_campana.py \
    --manifest "$CAMPAIGN_MANIFEST" \
    --preflight "$PREFLIGHT" \
    --raw-dir "$RAW" \
    --analysis-dir "$ANALYSIS" \
    --status preflight_failed \
    --exit-code "$preflight_rc" || {
    echo "FALLO BLOQUEANTE: no se pudo registrar el preflight fallido" >&2
  }
  echo "El preflight falló; consulta $LOGS/preflight.log" >&2
  exit "$preflight_rc"
fi

if python escribe_manifiesto_campana.py \
    --manifest "$CAMPAIGN_MANIFEST" \
    --preflight "$PREFLIGHT" \
    --raw-dir "$RAW" \
    --analysis-dir "$ANALYSIS" \
    --status running; then
  :
else
  running_rc=$?
  python escribe_manifiesto_campana.py \
    --manifest "$CAMPAIGN_MANIFEST" \
    --preflight "$PREFLIGHT" \
    --raw-dir "$RAW" \
    --analysis-dir "$ANALYSIS" \
    --status preflight_failed \
    --exit-code "$running_rc" || true
  echo "La telemetría GPU no superó el preflight; no se levantó vLLM" >&2
  exit "$running_rc"
fi
MANIFEST_STARTED=1

server_pid=""
actual_quant=""
actual_served_name=""
completed_sessions=0
server_launch_index=0

health_ok() {
  local expected_name="$1"
  local health_file="$LOGS/health_${expected_name}.json"
  curl --fail --silent --show-error \
    "http://127.0.0.1:${PORT}/v1/models" \
    --output "$health_file" || return 1
  python -c 'import json,sys; data=json.load(open(sys.argv[1], encoding="utf-8")); ids=[item["id"] for item in data["data"]]; raise SystemExit(0 if sys.argv[2] in ids else 1)' \
    "$health_file" "$expected_name"
}

wait_for_port_down() {
  local attempts="$SHUTDOWN_TIMEOUT_S"
  for ((i=0; i<attempts; i++)); do
    if ! curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

wait_for_gpu_idle() {
  local attempts="$SHUTDOWN_TIMEOUT_S"
  local pids=""
  for ((i=0; i<attempts; i++)); do
    pids="$(
      nvidia-smi \
        --query-compute-apps=pid \
        --format=csv,noheader,nounits |
        tr -d '[:space:]'
    )"
    [[ -z "$pids" ]] && return 0
    sleep 1
  done
  return 1
}

verificar_exclusividad_vllm() {
  [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null || {
    echo "No existe el grupo de procesos vLLM esperado" >&2
    return 1
  }
  local found=0
  local pid=""
  local pgid=""
  while IFS= read -r pid; do
    pid="${pid//[[:space:]]/}"
    [[ -n "$pid" ]] || continue
    found=1
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d '[:space:]')"
    if [[ "$pgid" != "$server_pid" ]]; then
      echo "PID GPU ajeno o no verificable: pid=$pid pgid=${pgid:-ausente}; esperado=$server_pid" >&2
      return 1
    fi
  done < <(
    nvidia-smi \
      --query-compute-apps=pid \
      --format=csv,noheader,nounits
  )
  (( found == 1 )) || {
    echo "nvidia-smi no ve ningún proceso vLLM en la GPU" >&2
    return 1
  }
}

detener() {
  local stop_rc=0
  if [[ -n "$server_pid" ]]; then
    kill -TERM -- "-$server_pid" 2>/dev/null || true
    for ((i=0; i<SHUTDOWN_TIMEOUT_S; i++)); do
      kill -0 -- "-$server_pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 -- "-$server_pid" 2>/dev/null; then
      echo "vLLM no terminó con SIGTERM; se fuerza su grupo" >&2
      kill -KILL -- "-$server_pid" 2>/dev/null || true
    fi
    wait "$server_pid" 2>/dev/null || true
  fi
  server_pid=""
  actual_quant=""
  actual_served_name=""
  if ! wait_for_port_down; then
    echo "El puerto $PORT sigue ocupado; no se matará un proceso desconocido" >&2
    stop_rc=1
  fi
  if ! wait_for_gpu_idle; then
    echo "La GPU conserva procesos después de detener vLLM; no se cambiará de precisión" >&2
    stop_rc=1
  fi
  return "$stop_rc"
}

finalizar() {
  local rc=$?
  trap - EXIT INT TERM
  detener || rc=1
  if [[ "${MANIFEST_STARTED:-0}" == "1" ]]; then
    local final_status="failed"
    [[ "$rc" == "0" ]] && final_status="complete"
    if ! python escribe_manifiesto_campana.py \
      --manifest "$CAMPAIGN_MANIFEST" \
      --preflight "$PREFLIGHT" \
      --raw-dir "$RAW" \
      --analysis-dir "$ANALYSIS" \
      --status "$final_status" \
      --exit-code "$rc"; then
      echo "FALLO BLOQUEANTE: no se pudo finalizar el manifiesto" >&2
      rc=1
      python escribe_manifiesto_campana.py \
        --manifest "$CAMPAIGN_MANIFEST" \
        --preflight "$PREFLIGHT" \
        --raw-dir "$RAW" \
        --analysis-dir "$ANALYSIS" \
        --status failed \
        --exit-code "$rc" || {
        echo "FALLO BLOQUEANTE: tampoco se pudo marcar failed" >&2
      }
    fi
  fi
  exit "$rc"
}
trap finalizar EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

levantar() {
  local quant="$1"
  local model="$2"
  local served_name="infera-${quant,,}"
  # Cada repetición instrumental empieza desde un servidor recién cargado.
  # Dentro del bloque se reutilizan pesos para los tres brazos; entre bloques
  # se reinicia incluso si la precisión siguiente es la misma.
  detener || return 1
  if curl --fail --silent "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    echo "El puerto $PORT ya está ocupado; no se usará un servidor ajeno" >&2
    return 1
  fi

  server_launch_index=$((server_launch_index + 1))
  local log="$LOGS/vllm_${quant}_launch${server_launch_index}.log"
  local flags=(--dtype float16)
  [[ "$quant" == "AWQ" ]] && flags+=(--quantization awq)
  setsid python -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --served-model-name "$served_name" \
    "${flags[@]}" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    >"$log" 2>&1 &
  server_pid="$!"

  local attempts=$((STARTUP_TIMEOUT_S / 5))
  (( attempts > 0 )) || attempts=1
  for ((i=0; i<attempts; i++)); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "vLLM murió durante el arranque ($log)" >&2
      return 1
    fi
    if health_ok "$served_name"; then
      actual_quant="$quant"
      actual_served_name="$served_name"
      verificar_exclusividad_vllm || return 1
      echo "Estabilización posterior a la carga de $quant: ${COOL}s"
      sleep "$COOL"
      health_ok "$served_name" || return 1
      verificar_exclusividad_vllm || return 1
      return 0
    fi
    sleep 5
  done
  echo "vLLM no respondió dentro de ${STARTUP_TIMEOUT_S}s ($log)" >&2
  return 1
}

sesion() {
  local quant="$1"
  local model="$2"
  local brazo="$3"
  local rep="$4"
  health_ok "$actual_served_name" || {
    echo "vLLM perdió salud antes de $quant/$brazo/rep$rep" >&2
    return 1
  }
  verificar_exclusividad_vllm || return 1
  local intervention_flag=()
  local session_rc=0
  [[ "$brazo" == "completo" ]] || intervention_flag=(--require-intervention)
  echo ">>> $quant / $brazo / repetición instrumental $rep"
  if python infera_session_runner.py \
    --vllm-url "$URL" \
    --model "$actual_served_name" \
    --model-source "$model" \
    --tokenizer "$model" \
    --quant "$quant" \
    --brazo "$brazo" \
    --rep "$rep" \
    --kb-dir "$KB_DIR" \
    --tasks "$SESSION" \
    --compaction-threshold "$THRESH" \
    --pares-conservados "$PARES" \
    --segundos-reposo "$REPOSO" \
    --max-model-len "$MAX_MODEL_LEN" \
    --expected-tasks "$EXPECTED_TASKS" \
    --request-timeout-s "$REQUEST_TIMEOUT_S" \
    --warmup "$WARMUP" \
    --post-warmup-settle-s "$POST_WARMUP_SETTLE_S" \
    --expected-server-pgid "$server_pid" \
    "${intervention_flag[@]}" \
    --out "$RAW/run_${quant}_${brazo}_rep${rep}.jsonl" \
    2>&1 | tee -a "$LOGS/sesiones.log"; then
    :
  else
    session_rc=$?
    echo "La sesión $quant/$brazo/rep$rep falló; no se reintenta" >&2
    return "$session_rc"
  fi
  verificar_exclusividad_vllm || return 1
  completed_sessions=$((completed_sessions + 1))
  if (( completed_sessions < 18 )); then
    sleep "$COOL"
  fi
}

bloque() {
  local quant="$1"
  local model="$2"
  local rep="$3"
  shift 3
  local started=0
  local attempt=0
  for ((attempt=1; attempt<=SERVER_START_ATTEMPTS; attempt++)); do
    if levantar "$quant" "$model"; then
      started=1
      break
    fi
    echo "Falló arranque $attempt/$SERVER_START_ATTEMPTS de $quant" >&2
    detener || return 1
    if (( attempt < SERVER_START_ATTEMPTS )); then
      sleep 30
    fi
  done
  (( started == 1 )) || {
    echo "vLLM no pudo iniciar para $quant/rep$rep" >&2
    return 1
  }
  for brazo in "$@"; do
    sesion "$quant" "$model" "$brazo" "$rep" || return 1
  done
}

echo "Campaña: $OUT"
echo "APC/prefix caching: apagada (no se pasa --enable-prefix-caching)"

# Cuadrado latino por precisión: cada brazo ocupa una vez cada posición.
bloque AWQ  "$AWQ_MODEL"  1 completo resumen descarte
bloque FP16 "$FP16_MODEL" 1 descarte resumen completo
bloque FP16 "$FP16_MODEL" 2 completo descarte resumen
bloque AWQ  "$AWQ_MODEL"  2 resumen descarte completo
bloque AWQ  "$AWQ_MODEL"  3 descarte completo resumen
bloque FP16 "$FP16_MODEL" 3 resumen completo descarte

detener
python analiza_tres_brazos.py \
  --crudos "$RAW" \
  --salida "$ANALYSIS" \
  --expected-tasks "$EXPECTED_TASKS" \
  --max-model-len "$MAX_MODEL_LEN" \
  --pairs-kept "$PARES" \
  2>&1 | tee "$LOGS/analisis.log"
