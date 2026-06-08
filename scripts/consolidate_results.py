"""
INFERA — Consolidador de Resultados
====================================
Corre este script en la carpeta raíz del repo (donde está /results/).
Genera dos archivos:
  1. infera_results_raw.csv   → todos los registros individuales
  2. infera_summary.csv       → agregado por configuración (media ± std de 3 reps)

Uso:
  python consolidate_results.py
  python consolidate_results.py --results-dir /ruta/a/results/
"""

import json
import glob
import sys
import argparse
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("ERROR: Necesitas pandas y numpy.")
    print("  pip install pandas numpy")
    sys.exit(1)

# ── Argumentos ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", default="results", help="Carpeta de resultados")
parser.add_argument("--out-raw",     default="infera_results_raw.csv")
parser.add_argument("--out-summary", default="infera_summary.csv")
args = parser.parse_args()

results_dir = Path(args.results_dir)
if not results_dir.exists():
    print(f"ERROR: No existe la carpeta '{results_dir}'")
    print("  Asegúrate de correr el script desde la raíz del repo.")
    sys.exit(1)

# ── Directorios a incluir (SOLO los completos, no pilotos) ────────────────────
INCLUDE_DIRS = [
    "fp16_20260601_175220",         # FP16 completo
    "int8_w8a16_20260602_013445",   # INT8 completo
    "int4_awq_20260602_124420",     # INT4 AWQ completo
]

print("=" * 60)
print("INFERA — Consolidador de Resultados")
print("=" * 60)

# ── Carga ─────────────────────────────────────────────────────────────────────
records = []
for dir_name in INCLUDE_DIRS:
    dir_path = results_dir / dir_name
    if not dir_path.exists():
        print(f"  AVISO: No se encontró '{dir_path}' — omitiendo")
        continue
    
    json_files = [f for f in glob.glob(str(dir_path / "*.json"))
                  if "summary" not in f]
    
    count_ok = 0
    count_err = 0
    for fpath in json_files:
        try:
            data = json.load(open(fpath, encoding="utf-8"))
            records.append(data)
            count_ok += 1
        except Exception as e:
            count_err += 1
    
    print(f"  {dir_name}: {count_ok} JSONs leídos"
          + (f" | {count_err} con error" if count_err else ""))

if not records:
    print("\nERROR: No se cargó ningún registro.")
    sys.exit(1)

df = pd.DataFrame(records)
print(f"\nTotal registros cargados: {len(df)}")

# ── Diagnóstico rápido ────────────────────────────────────────────────────────
print("\n--- Status ---")
print(df["status"].value_counts().to_string())

print("\n--- Cuantizaciones ---")
print(df["vi1_quantization"].value_counts().to_string())

print("\n--- Batch sizes ---")
print(df["vi2_batch_size"].value_counts().sort_index().to_string())

print("\n--- Output lengths ---")
print(df["vi3_output_length"].value_counts().sort_index().to_string())

print("\n--- Context cases ---")
print(df["vi4_context_case"].value_counts().sort_index().to_string())

# ── Filtro para análisis ──────────────────────────────────────────────────────
df_ok = df[
    (df["status"] == "success") &
    (~df["batch_padded"].astype(bool)) &
    (df["energy_j"] > 0) &
    (df["completion_tokens"] > 0)
].copy()

n_oom     = (df["status"] == "oom").sum()
n_error   = (df["status"] == "error").sum()
n_padded  = df["batch_padded"].astype(bool).sum()
n_removed = len(df) - len(df_ok)

print(f"\n--- Filtrado ---")
print(f"  OOM: {n_oom}")
print(f"  Error: {n_error}")
print(f"  Padded (excluidos): {n_padded}")
print(f"  Total excluidos: {n_removed}")
print(f"  Registros válidos para análisis: {len(df_ok)}")

# ── Métricas derivadas ────────────────────────────────────────────────────────
# j_per_token ya viene en el JSON, pero recalculamos para validar
df_ok["j_per_token_calc"] = df_ok["energy_j"] / df_ok["completion_tokens"]

# Emisiones CO2 (factor EU-RO-1 Romania: 0.294 kgCO2/kWh — fuente: ember-climate.org)
df_ok["kgco2eq"] = (df_ok["energy_j"] / 3_600_000) * 0.294
df_ok["gco2eq"]  = df_ok["kgco2eq"] * 1000  # gramos, más legible

# Eficiencia inversa (tokens por julio — mayor = mejor)
df_ok["tok_per_j"] = df_ok["completion_tokens"] / df_ok["energy_j"]

# ── Guardar RAW ───────────────────────────────────────────────────────────────
# Columnas relevantes para el análisis (ordenadas)
cols_export = [
    "run_id", "config_key",
    "vi1_quantization", "vi2_batch_size", "vi3_output_length",
    "vi4_context_case", "vi4_mean_input_tokens", "repetition",
    "batch_index", "batch_total", "batch_actual_size",
    "status", "energy_j", "duration_s",
    "avg_power_w", "peak_power_w",
    "vram_peak_mb", "vram_used_start_mb",
    "prompt_tokens", "completion_tokens",
    "throughput_tok_s", "j_per_token", "tpot_ms",
    "j_per_token_calc", "tok_per_j", "gco2eq",
    "nvml_sample_count", "nvml_available",
]
# Incluir solo columnas que existen en el DataFrame
cols_export = [c for c in cols_export if c in df_ok.columns]

df_ok[cols_export].to_csv(args.out_raw, index=False)
print(f"\n✓ RAW guardado: {args.out_raw} ({len(df_ok)} filas)")

# ── Summary agregado por configuración ───────────────────────────────────────
config_cols = [
    "vi1_quantization", "vi2_batch_size",
    "vi3_output_length", "vi4_context_case"
]

# Primero agrupar por (config + repetición) para obtener 1 valor por rep
per_rep = df_ok.groupby(config_cols + ["repetition"]).agg(
    energy_j       = ("energy_j",        "mean"),
    throughput     = ("throughput_tok_s", "mean"),
    j_per_token    = ("j_per_token",      "mean"),
    tpot_ms        = ("tpot_ms",          "mean"),
    avg_power_w    = ("avg_power_w",      "mean"),
    peak_power_w   = ("peak_power_w",     "mean"),
    vram_peak_mb   = ("vram_peak_mb",     "mean"),
    completion_tok = ("completion_tokens","mean"),
    tok_per_j      = ("tok_per_j",        "mean"),
    gco2eq         = ("gco2eq",           "mean"),
    n_batches      = ("energy_j",         "count"),
).reset_index()

# Luego agrupar por config para obtener media/std entre las 3 reps
summary = per_rep.groupby(config_cols).agg(
    energy_j_mean    = ("energy_j",       "mean"),
    energy_j_std     = ("energy_j",       "std"),
    throughput_mean  = ("throughput",      "mean"),
    throughput_std   = ("throughput",      "std"),
    j_per_token_mean = ("j_per_token",     "mean"),
    j_per_token_std  = ("j_per_token",     "std"),
    tpot_ms_mean     = ("tpot_ms",         "mean"),
    tpot_ms_std      = ("tpot_ms",         "std"),
    avg_power_mean   = ("avg_power_w",     "mean"),
    peak_power_mean  = ("peak_power_w",    "mean"),
    vram_peak_mean   = ("vram_peak_mb",    "mean"),
    tok_per_j_mean   = ("tok_per_j",       "mean"),
    gco2eq_mean      = ("gco2eq",          "mean"),
    n_reps           = ("energy_j",        "count"),
).reset_index()

# Coeficiente de variación (CV) — mide estabilidad entre repeticiones
summary["cv_throughput"] = (summary["throughput_std"] / summary["throughput_mean"] * 100).round(2)
summary["cv_j_per_token"] = (summary["j_per_token_std"] / summary["j_per_token_mean"] * 100).round(2)

# Redondear para legibilidad
for col in summary.select_dtypes(include="float").columns:
    summary[col] = summary[col].round(4)

summary.to_csv(args.out_summary, index=False)
print(f"✓ SUMMARY guardado: {args.out_summary} ({len(summary)} configuraciones)\n")

# ── Verificación de completitud ───────────────────────────────────────────────
print("--- Verificación de completitud ---")
expected_configs = 81  # 3^4 factorial
expected_reps    = 3

for quant in ["fp16", "int8_w8a16", "int4_awq"]:
    sub = summary[summary["vi1_quantization"] == quant]
    ok_configs  = len(sub)
    ok_reps_3   = (sub["n_reps"] == expected_reps).sum()
    print(f"  {quant:15s}: {ok_configs:3d}/{expected_configs} configs | "
          f"{ok_reps_3:3d} con exactamente 3 reps")

print("\n--- Vista previa del summary (primeras 5 filas) ---")
print(summary.head().to_string(index=False))

print("\n✓ Listo. Sube 'infera_results_raw.csv' y 'infera_summary.csv' a Claude.")