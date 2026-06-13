#!/usr/bin/env bash
# run_all.sh QUANT MODEL_PATH [REPS] [THRESHOLD]
# Corre naive + compaction x REPS para UNA cuantizacion ya servida por vLLM.
# Aplica cooling de 2 min entre corridas (protocolo EXP1).
#
# Ej:  ./run_all.sh FP16 /models/llama3.1-8b-instruct 3 4000
set -euo pipefail

QUANT="${1:?Falta QUANT (FP16/INT8/AWQ)}"
MODEL="${2:?Falta MODEL_PATH}"
REPS="${3:-3}"
THRESH="${4:-4500}"
URL="${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
COOL="${COOL_SECONDS:-120}"

mkdir -p results

for arm in naive compaction; do
  for rep in $(seq 1 "$REPS"); do
    OUT="results/run_${QUANT}_${arm}_rep${rep}.jsonl"
    echo "=============================================================="
    echo ">> ${QUANT} | ${arm} | rep ${rep}  ->  ${OUT}"
    echo "=============================================================="
    python infera_session_runner.py \
      --vllm-url "$URL" --model "$MODEL" \
      --quant "$QUANT" --arm "$arm" --rep "$rep" \
      --kb-dir kb --tasks session_tasks_v3.json \
      --compaction-threshold "$THRESH" \
      --out "$OUT"
    echo ">> cooling ${COOL}s ..."
    sleep "$COOL"
  done
done

echo "[OK] ${QUANT} completo. Cuando termines todas las cuantizaciones: python infera_analysis.py"
