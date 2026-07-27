#!/usr/bin/env bash
# run_all.sh QUANT MODEL_PATH [REPS] [THRESHOLD]
# Corre historial completo + compactación para UNA representación ya servida.
# Escribe en results/runs/<RUN_TAG>/ y nunca toca el conjunto de referencia.
#
# Ej: RUN_TAG=mi_prueba ./run_all.sh FP16 /models/llama3.1-8b-instruct 3 4500
set -euo pipefail

cd "$(dirname "$0")"

QUANT="${1:?Falta QUANT (FP16/INT8/AWQ)}"
MODEL="${2:?Falta MODEL_PATH}"
REPS="${3:-3}"
THRESH="${4:-4500}"
URL="${VLLM_URL:-http://localhost:8000/v1/chat/completions}"
COOL="${COOL_SECONDS:-120}"
RUN_TAG="${RUN_TAG:-custom}"
TASKS="${TASKS:-config/session_tasks.example.json}"
KB_DIR="${KB_DIR:-kb}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/runs/${RUN_TAG}}"

mkdir -p "$OUTPUT_ROOT"

for arm in naive compaction; do
  for rep in $(seq 1 "$REPS"); do
    OUT="${OUTPUT_ROOT}/run_${RUN_TAG}_${QUANT}_${arm}_rep${rep}.jsonl"
    echo "=============================================================="
    echo ">> ${QUANT} | ${arm} | rep ${rep}  ->  ${OUT}"
    echo "=============================================================="
    python infera_session_runner.py \
      --vllm-url "$URL" --model "$MODEL" \
      --quant "$QUANT" --arm "$arm" --rep "$rep" \
      --kb-dir "$KB_DIR" --tasks "$TASKS" \
      --compaction-threshold "$THRESH" \
      --out "$OUT"
    echo ">> cooling ${COOL}s ..."
    sleep "$COOL"
  done
done

echo "[OK] ${QUANT} completo en ${OUTPUT_ROOT}."
echo "[INFO] Analiza cuando estén presentes todas las representaciones declaradas."
