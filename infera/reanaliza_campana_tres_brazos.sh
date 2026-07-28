#!/usr/bin/env bash
# Recupera/finaliza únicamente el análisis de una campaña ya recolectada.
# No levanta vLLM ni emite peticiones GPU.
set -Eeuo pipefail
cd "$(dirname "$0")"

[[ $# -ge 1 && $# -le 2 ]] || {
  echo "Uso: $0 RUTA_CAMPANA [RUTA_SALIDA_ANALISIS]" >&2
  exit 2
}

CAMPAIGN_DIR="$1"
[[ -d "$CAMPAIGN_DIR/raw" ]] || {
  echo "No existe $CAMPAIGN_DIR/raw" >&2
  exit 2
}
[[ -f "$CAMPAIGN_DIR/manifiesto_campana.json" ]] || {
  echo "Falta manifiesto_campana.json en $CAMPAIGN_DIR" >&2
  exit 2
}

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
ANALYSIS_DIR="${2:-$CAMPAIGN_DIR/analisis_recuperado_${STAMP}}"
[[ ! -e "$ANALYSIS_DIR" ]] || {
  echo "No se sobrescribirá $ANALYSIS_DIR" >&2
  exit 2
}

python analiza_tres_brazos.py \
  --crudos "$CAMPAIGN_DIR/raw" \
  --salida "$ANALYSIS_DIR" \
  --expected-tasks 29 \
  --max-model-len 8192 \
  --pairs-kept 4

campaign_status="$(
  python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$CAMPAIGN_DIR/manifiesto_campana.json"
)"
if [[ "$campaign_status" != "complete" ]]; then
  preflight="$(
    python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["preflight"])' \
      "$CAMPAIGN_DIR/manifiesto_campana.json"
  )"
  python escribe_manifiesto_campana.py \
    --manifest "$CAMPAIGN_DIR/manifiesto_campana.json" \
    --preflight "$preflight" \
    --raw-dir "$CAMPAIGN_DIR/raw" \
    --analysis-dir "$ANALYSIS_DIR" \
    --status complete \
    --exit-code 0
fi

echo "Análisis recuperado sin volver a usar GPU: $ANALYSIS_DIR"
