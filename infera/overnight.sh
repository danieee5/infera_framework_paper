#!/usr/bin/env bash
# overnight.sh — corrida desatendida del diseño historial/compactación.
# Para cada cuantizacion: levanta vLLM, corre naive+compaction x REPS, baja vLLM,
# pasa a la siguiente. Al final corre el analisis. Resiliente: una corrida que
# falle no aborta el resto.
#
# Variables de entorno controlables:
#   CONFIG_FILE  archivo de variables (default: config/experiment.env)
#   SESSION      fichero de tareas declarado en config/experiment.env
#   SESSION_TAG  etiqueta usada en los nombres de salida
#   REPS         replicas por condicion (default: 3)
#   THRESH       tokens para disparar compactacion (default: 4500)
#   PORT         puerto vLLM (default: 8000)
#   COOL         enfriamiento entre corridas en segundos (default: 120)
#   FP16_MODEL   ruta o id del modelo FP16
#   AWQ_MODEL    ruta o id del modelo AWQ
#   OUTPUT_ROOT  destino; por defecto se crea dentro de results/runs/
#
# Uso basico (dentro de tmux):
#   bash overnight.sh
#
# Uso con sesion personalizada:
#   SESSION=config/mi_sesion.json SESSION_TAG=mi_prueba REPS=1 bash overnight.sh

# Siempre correr desde el directorio del propio script,
# sin importar desde donde se invoque (fix: 'python infera_session_runner.py not found').
cd "$(dirname "$0")"

set -uo pipefail   # SIN -e a proposito: queremos continuar pese a fallos puntuales

CONFIG_FILE="${CONFIG_FILE:-config/experiment.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: no existe $CONFIG_FILE"
  echo "Copia config/experiment.env.example, edítalo y vuelve a ejecutar."
  exit 2
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${SESSION:?Define SESSION en $CONFIG_FILE}"
: "${SESSION_TAG:?Define SESSION_TAG en $CONFIG_FILE}"
REPS="${REPS:-3}"
THRESH="${THRESH:-4500}"
PORT="${PORT:-8000}"
COOL="${COOL:-120}"
FP16_MODEL="${FP16_MODEL:-/models/llama3.1-8b-instruct}"
AWQ_MODEL="${AWQ_MODEL:-/models/llama3.1-8b-instruct-awq}"
EXPECTED_TASKS="${EXPECTED_TASKS:-29}"
EXPECTED_COMPACTIONS="${EXPECTED_COMPACTIONS:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
KB_DIR="${KB_DIR:-kb}"
URL="http://localhost:${PORT}/v1/chat/completions"

# Configuraciones: "QUANT|MODEL|FLAGS_VLLM"
CONFIGS=(
  "FP16|${FP16_MODEL}|--dtype float16"
  "AWQ|${AWQ_MODEL}|--quantization awq --dtype float16"
)

OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/${SESSION_TAG}_$(date +%Y%m%d_%H%M%S)}"
LOG="${OUTPUT_ROOT}/execution.log"
VLLM_PID=""
mkdir -p "$OUTPUT_ROOT"

wait_for_vllm() {
  echo "   esperando a que vLLM responda..."
  for _ in $(seq 1 120); do                       # hasta 10 min
    if curl -s "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "   vLLM listo."; return 0
    fi
    sleep 5
  done
  echo "   ERROR: vLLM no respondio en 10 min."; return 1
}

stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    sleep 15
  fi
  VLLM_PID=""
}
trap stop_vllm EXIT

run_session() {
  local sess_file="$1"
  local sess_tag="${3:-$(basename "$sess_file" .json | sed 's/session_tasks_//')}"
  local arms=("$2")        # "naive compaction" o "naive" (para filler)
  IFS=' ' read -ra arms <<< "$2"

  echo ""
  echo ">>>>>> SESION: $sess_file  (tag=$sess_tag)"

  for cfg in "${CONFIGS[@]}"; do
    IFS='|' read -r QUANT MODEL FLAGS <<< "$cfg"
    echo ""
    echo "  >> CUANTIZACION: $QUANT  ($MODEL)"
    stop_vllm
    echo "   levantando vLLM..."
    # shellcheck disable=SC2086
    nohup python -m vllm.entrypoints.openai.api_server \
          --model "$MODEL" $FLAGS --max-model-len "$MAX_MODEL_LEN" --port "$PORT" \
          > "${OUTPUT_ROOT}/vllm_${QUANT}.log" 2>&1 &
    VLLM_PID=$!

    if wait_for_vllm; then
      for arm in "${arms[@]}"; do
        for rep in $(seq 1 "$REPS"); do
          OUT="${OUTPUT_ROOT}/run_${sess_tag}_${QUANT}_${arm}_rep${rep}.jsonl"
          echo "   --- $sess_tag | $QUANT | $arm | rep $rep ---"
          python infera_session_runner.py \
            --vllm-url "$URL" --model "$MODEL" \
            --quant "$QUANT" --arm "$arm" --rep "$rep" \
            --kb-dir "$KB_DIR" --tasks "$sess_file" \
            --compaction-threshold "$THRESH" \
            --out "$OUT" \
            || echo "   WARN: fallo $sess_tag $QUANT $arm rep$rep (continuo)"
          sleep "$COOL"
        done
      done
    else
      echo "   WARN: se omite $QUANT. Revisa ${OUTPUT_ROOT}/vllm_${QUANT}.log"
    fi
    stop_vllm
  done
}

{
echo "===================================================="
echo "  OVERNIGHT INFERA Protocolo C — inicio $(date)"
echo "  SESSION=$SESSION  TAG=$SESSION_TAG"
echo "  REPS=$REPS  THRESH=$THRESH  PORT=$PORT"
echo "===================================================="

# --- Corrida principal: naive + compaction, todos los quants ---
run_session "$SESSION" "naive compaction" "$SESSION_TAG"

echo ""
echo ">>>>>> Analisis final"
python analyze_results.py \
  --source "$OUTPUT_ROOT" \
  --out "${OUTPUT_ROOT}/analysis" \
  --session-tag "$SESSION_TAG" \
  --quants "AWQ,FP16" \
  --reps "$(seq -s, 1 "$REPS")" \
  --expected-tasks "$EXPECTED_TASKS" \
  --expected-compactions "$EXPECTED_COMPACTIONS" \
  || echo "WARN: el analisis fallo; correlo manualmente luego."

echo ""
echo "===================================================="
echo "  FIN $(date)"
echo "===================================================="
} 2>&1 | tee "$LOG"
