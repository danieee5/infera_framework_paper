#!/usr/bin/env bash
# overnight.sh — corrida DESATENDIDA del Protocolo C.
# Para cada cuantizacion: levanta vLLM, corre naive+compaction x REPS, baja vLLM,
# pasa a la siguiente. Al final corre el analisis. Resiliente: una corrida que
# falle no aborta el resto.
#
# Variables de entorno controlables:
#   SESSION      fichero de tareas (default: session_tasks_v3.json)
#   REPS         replicas por condicion (default: 3)
#   THRESH       tokens para disparar compactacion (default: 4500)
#   FILLER       si =1, corre tambien el control causal (filler, solo naive)
#   PORT         puerto vLLM (default: 8000)
#   COOL         enfriamiento entre corridas en segundos (default: 120)
#
# Uso basico (dentro de tmux):
#   bash overnight.sh
#
# Uso con filler control:
#   FILLER=1 bash overnight.sh
#
# Uso con sesion personalizada:
#   SESSION=session_tasks_v3_filler.json REPS=1 bash overnight.sh

set -uo pipefail   # SIN -e a proposito: queremos continuar pese a fallos puntuales

SESSION="${SESSION:-session_tasks_v3.json}"
REPS="${REPS:-3}"
THRESH="${THRESH:-4500}"
PORT="${PORT:-8000}"
COOL="${COOL:-120}"
FILLER="${FILLER:-0}"
URL="http://localhost:${PORT}/v1/chat/completions"
LOG="overnight_$(date +%Y%m%d_%H%M%S).log"

# Configuraciones: "QUANT|MODEL|FLAGS_VLLM"
CONFIGS=(
  "FP16|/workspace/models/llama3.1-8b-instruct|--dtype float16"
  "AWQ|/workspace/models/llama3.1-8b-instruct-awq|--quantization awq --dtype float16"
)

# Derivar prefijo de nombre de salida desde el fichero de sesion
# session_tasks_v3.json -> v3 ; session_tasks_v3_filler.json -> v3_filler
SESSION_TAG=$(basename "$SESSION" .json | sed 's/session_tasks_//')

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
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 15                                          # dar tiempo a liberar VRAM
}

run_session() {
  local sess_file="$1"
  local sess_tag
  sess_tag=$(basename "$sess_file" .json | sed 's/session_tasks_//')
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
          --model "$MODEL" $FLAGS --max-model-len 8192 --port "$PORT" \
          > "vllm_${QUANT}.log" 2>&1 &
    echo $! > vllm.pid

    if wait_for_vllm; then
      for arm in "${arms[@]}"; do
        for rep in $(seq 1 "$REPS"); do
          OUT="results/run_${sess_tag}_${QUANT}_${arm}_rep${rep}.jsonl"
          echo "   --- $sess_tag | $QUANT | $arm | rep $rep ---"
          python infera_session_runner.py \
            --vllm-url "$URL" --model "$MODEL" \
            --quant "$QUANT" --arm "$arm" --rep "$rep" \
            --kb-dir kb --tasks "$sess_file" \
            --compaction-threshold "$THRESH" \
            --out "$OUT" \
            || echo "   WARN: fallo $sess_tag $QUANT $arm rep$rep (continuo)"
          sleep "$COOL"
        done
      done
    else
      echo "   WARN: se omite $QUANT. Revisa vllm_${QUANT}.log"
    fi
    stop_vllm
  done
}

{
echo "===================================================="
echo "  OVERNIGHT INFERA Protocolo C — inicio $(date)"
echo "  SESSION=$SESSION  TAG=$SESSION_TAG"
echo "  REPS=$REPS  THRESH=$THRESH  PORT=$PORT  FILLER=$FILLER"
echo "===================================================="

mkdir -p results

# --- Corrida principal: naive + compaction, todos los quants ---
run_session "$SESSION" "naive compaction"

# --- Corrida filler (control causal): naive SOLO, todos los quants ---
# Se activa con FILLER=1 bash overnight.sh
# Proposito: verificar si el context-rot persiste con historia de relleno neutral
# (si si -> mecanismo es longitud KV cache; si no -> interferencia semantica)
if [[ "$FILLER" == "1" ]]; then
  echo ""
  echo ">>>>>> FILLER CONTROL (naive only)"
  run_session "session_tasks_v3_filler.json" "naive"
fi

echo ""
echo ">>>>>> Analisis final"
python infera_analysis.py --results results --out results/analysis \
  || echo "WARN: el analisis fallo; correlo manualmente luego."

echo ""
echo "===================================================="
echo "  FIN $(date)"
echo "===================================================="
} 2>&1 | tee "$LOG"
