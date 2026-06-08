#!/usr/bin/env bash
# overnight.sh — corrida DESATENDIDA del Protocolo C.
# Para cada cuantizacion: levanta vLLM, corre naive+compaction x REPS, baja vLLM,
# pasa a la siguiente. Al final corre el analisis. Resiliente: una corrida que
# falle no aborta el resto.
#
# Uso (dentro de tmux, ver guia):  bash overnight.sh
set -uo pipefail   # SIN -e a proposito: queremos continuar pese a fallos puntuales

REPS="${REPS:-3}"
THRESH="${THRESH:-4500}"     # Adelantado del piloto (era 5000). Dispara ~T07-T08 del nuevo corpus v2.
PORT="${PORT:-8000}"
COOL="${COOL:-120}"          # enfriamiento entre corridas (s)
URL="http://localhost:${PORT}/v1/chat/completions"
LOG="overnight_$(date +%Y%m%d_%H%M%S).log"

# Configuraciones: "QUANT|MODEL|FLAGS_VLLM". Descomenta INT8 si lo quieres.
CONFIGS=(
  "FP16|/workspace/models/llama3.1-8b-instruct|--dtype float16"
  "AWQ|/workspace/models/llama3.1-8b-instruct-awq|--quantization awq --dtype float16"
  # "INT8|/workspace/models/llama3.1-8b-instruct|--quantization bitsandbytes --load-format bitsandbytes"
)

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

{
echo "===================================================="
echo "  OVERNIGHT INFERA Protocolo C — inicio $(date)"
echo "  REPS=$REPS  THRESH=$THRESH (provisional)  PORT=$PORT"
echo "===================================================="

mkdir -p results

for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r QUANT MODEL FLAGS <<< "$cfg"
  echo ""
  echo ">>>>>> CUANTIZACION: $QUANT  ($MODEL)"
  stop_vllm
  echo "   levantando vLLM..."
  # shellcheck disable=SC2086
  nohup python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" $FLAGS --max-model-len 8192 --port "$PORT" \
        > "vllm_${QUANT}.log" 2>&1 &
  echo $! > vllm.pid

  if wait_for_vllm; then
    for arm in naive compaction; do
      for rep in $(seq 1 "$REPS"); do
        OUT="results/run_${QUANT}_${arm}_rep${rep}.jsonl"
        echo "   --- $QUANT | $arm | rep $rep ---"
        python infera_session_runner.py \
          --vllm-url "$URL" --model "$MODEL" \
          --quant "$QUANT" --arm "$arm" --rep "$rep" \
          --kb-dir kb --tasks session_tasks.json \
          --compaction-threshold "$THRESH" \
          --out "$OUT" \
          || echo "   WARN: fallo $QUANT $arm rep$rep (continuo)"
        sleep "$COOL"
      done
    done
  else
    echo "   WARN: se omite $QUANT. Revisa vllm_${QUANT}.log"
  fi
  stop_vllm
done

echo ""
echo ">>>>>> Analisis final"
python infera_analysis.py --results results --out results/analysis \
  || echo "WARN: el analisis fallo; correlo manualmente luego."

echo ""
echo "===================================================="
echo "  FIN $(date)"
echo "===================================================="
} 2>&1 | tee "$LOG"
