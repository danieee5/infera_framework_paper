"""
INFERA — Script de análisis principal (Sesión 4)
Genera todas las figuras del paper y la tabla de decisiones.

Uso:
    cd /workspace/infera   (o la raíz del repo)
    python analysis/analyze.py

Outputs en: analysis/figures/
"""

import json
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── CONFIG ──────────────────────────────────────────────────────────────────
RESULT_DIRS = [
    'results/fp16_20260601_175220/',
    'results/int8_w8a16_20260602_013445/',
    'results/int4_awq_20260602_124420/',
]
OUT_DIR = 'analysis/figures'
CARBON_FACTOR = 0.294  # kgCO2eq/kWh — EU-RO-1 Romania (2024)

QUANT_LABELS = {
    'fp16':        'FP16',
    'int8_w8a16':  'INT8 W8A16',
    'int4_awq':    'INT4 AWQ',
}
QUANT_COLORS = {
    'fp16':        '#2196F3',  # blue
    'int8_w8a16':  '#FF9800',  # orange
    'int4_awq':    '#4CAF50',  # green
}
BATCH_MARKERS = {1: 'o', 4: 's', 8: '^'}

os.makedirs(OUT_DIR, exist_ok=True)

# ─── PASO 1: CARGA ───────────────────────────────────────────────────────────
print("=" * 60)
print("PASO 1 — Carga de datos")
print("=" * 60)

records = []
for d in RESULT_DIRS:
    files = [f for f in glob.glob(f'{d}*.json') if 'summary' not in f]
    print(f"  {d}: {len(files)} JSONs")
    for f in files:
        records.append(json.load(open(f)))

df = pd.DataFrame(records)
print(f"\nTotal cargado: {len(df)} registros")
print("Status distribution:")
print(df['status'].value_counts().to_string())
print("\nCuantizaciones:")
print(df['vi1_quantization'].value_counts().to_string())

# ─── PASO 2: LIMPIEZA ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 2 — Limpieza y métricas derivadas")
print("=" * 60)

df_ok = df[
    (df['status'] == 'success') &
    (~df['batch_padded']) &
    (df['energy_j'] > 0) &
    (df['completion_tokens'] > 0)
].copy()

# Métricas derivadas
df_ok['j_per_token_real'] = df_ok['energy_j'] / df_ok['completion_tokens']
df_ok['kgco2eq'] = (df_ok['energy_j'] / 3_600_000) * CARBON_FACTOR

print(f"Tras filtrado: {len(df_ok)} registros válidos")
print(f"OOM: {(df['status'] == 'oom').sum()} | Errors: {(df['status'] == 'error').sum()}")

# ─── PASO 3: AGREGACIÓN POR CONFIG ───────────────────────────────────────────
print("\n" + "=" * 60)
print("PASO 3 — Agregación por configuración")
print("=" * 60)

config_cols = ['vi1_quantization', 'vi2_batch_size',
               'vi3_output_length', 'vi4_context_case']

# Primero: promediar dentro de cada repetición (múltiples batch calls → 1 valor por rep)
per_rep = df_ok.groupby(config_cols + ['repetition']).agg(
    energy_j        = ('energy_j',        'mean'),
    throughput      = ('throughput_tok_s', 'mean'),
    j_per_token     = ('j_per_token_real', 'mean'),
    tpot_ms         = ('tpot_ms',          'mean'),
    completion_tok  = ('completion_tokens','mean'),
    kgco2eq         = ('kgco2eq',          'mean'),
).reset_index()

# Segundo: estadísticos entre las 3 repeticiones
per_config = per_rep.groupby(config_cols).agg(
    energy_mean     = ('energy_j',    'mean'),
    energy_std      = ('energy_j',    'std'),
    throughput_mean = ('throughput',  'mean'),
    throughput_std  = ('throughput',  'std'),
    j_per_token_mean= ('j_per_token', 'mean'),
    j_per_token_std = ('j_per_token', 'std'),
    tpot_mean       = ('tpot_ms',     'mean'),
    tpot_std        = ('tpot_ms',     'std'),
    cv_throughput   = ('throughput',  lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan),
    n_reps          = ('throughput',  'count'),
).reset_index()

# CEI: Composite Efficiency Index = throughput/j_per_token (normalizado 0–1)
per_config['cei'] = (
    (per_config['throughput_mean'] / per_config['throughput_mean'].max()) /
    (per_config['j_per_token_mean'] / per_config['j_per_token_mean'].max())
)
per_config['cei'] = per_config['cei'] / per_config['cei'].max()

print(f"Configuraciones: {len(per_config)} (esperado: 81)")
print(f"Repeticiones min/max: {per_config['n_reps'].min()}/{per_config['n_reps'].max()}")

# ─── FIGURA 1: Throughput vs Batch Size ──────────────────────────────────────
print("\n[Fig 1] Throughput vs Batch Size")

fig, ax = plt.subplots(figsize=(8, 5))
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    d = per_config[per_config['vi1_quantization'] == quant]
    d_agg = d.groupby('vi2_batch_size').agg(
        tput_mean = ('throughput_mean', 'mean'),
        tput_std  = ('throughput_std',  'mean'),
    ).reset_index()
    ax.errorbar(
        d_agg['vi2_batch_size'], d_agg['tput_mean'],
        yerr=d_agg['tput_std'],
        marker='o', linewidth=2, capsize=4,
        color=QUANT_COLORS[quant], label=QUANT_LABELS[quant]
    )

ax.set_xlabel('Batch Size (VI2)', fontsize=12)
ax.set_ylabel('Throughput (tok/s)', fontsize=12)
ax.set_title('Fig. 1 — Throughput vs Batch Size por Cuantización', fontsize=13, fontweight='bold')
ax.set_xticks([1, 4, 8])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig1_throughput_vs_batch.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig1_throughput_vs_batch.png")

# ─── FIGURA 2: J/token vs Batch Size ─────────────────────────────────────────
print("[Fig 2] J/token vs Batch Size")

fig, ax = plt.subplots(figsize=(8, 5))
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    d = per_config[per_config['vi1_quantization'] == quant]
    d_agg = d.groupby('vi2_batch_size').agg(
        jpt_mean = ('j_per_token_mean', 'mean'),
        jpt_std  = ('j_per_token_std',  'mean'),
    ).reset_index()
    ax.errorbar(
        d_agg['vi2_batch_size'], d_agg['jpt_mean'],
        yerr=d_agg['jpt_std'],
        marker='o', linewidth=2, capsize=4,
        color=QUANT_COLORS[quant], label=QUANT_LABELS[quant]
    )

ax.set_xlabel('Batch Size (VI2)', fontsize=12)
ax.set_ylabel('J/token (↓ mejor)', fontsize=12)
ax.set_title('Fig. 2 — Eficiencia Energética (J/token) vs Batch Size', fontsize=13, fontweight='bold')
ax.set_xticks([1, 4, 8])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig2_jpertoken_vs_batch.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig2_jpertoken_vs_batch.png")

# ─── FIGURA 3: Pareto Frontier ────────────────────────────────────────────────
print("[Fig 3] Pareto Frontier")

def pareto_front(df_in):
    """Retorna filas no dominadas (mayor throughput Y menor j/token)."""
    idx_dominated = []
    for i, ri in df_in.iterrows():
        for j, rj in df_in.iterrows():
            if i == j:
                continue
            if (rj['throughput_mean'] >= ri['throughput_mean'] and
                    rj['j_per_token_mean'] <= ri['j_per_token_mean']):
                idx_dominated.append(i)
                break
    return df_in[~df_in.index.isin(idx_dominated)]

fig, ax = plt.subplots(figsize=(10, 7))

for _, row in per_config.iterrows():
    ax.scatter(
        row['throughput_mean'], row['j_per_token_mean'],
        c=QUANT_COLORS[row['vi1_quantization']],
        marker=BATCH_MARKERS[row['vi2_batch_size']],
        s=70, alpha=0.55, zorder=2
    )

pareto = pareto_front(per_config.copy())
ax.scatter(
    pareto['throughput_mean'], pareto['j_per_token_mean'],
    c='red', s=220, marker='*', zorder=5, label='Pareto-óptimo'
)

# Leyenda de cuantización
legend_quant = [
    mpatches.Patch(color=QUANT_COLORS[q], label=QUANT_LABELS[q])
    for q in ['fp16', 'int8_w8a16', 'int4_awq']
]
# Leyenda de batch markers
legend_batch = [
    plt.Line2D([0], [0], marker=BATCH_MARKERS[b], color='gray',
               linestyle='None', markersize=8, label=f'Batch={b}')
    for b in [1, 4, 8]
]
legend_pareto = [plt.Line2D([0], [0], marker='*', color='red',
                             linestyle='None', markersize=12, label='Pareto-óptimo')]
ax.legend(handles=legend_quant + legend_batch + legend_pareto, fontsize=9)

ax.set_xlabel('Throughput (tok/s)  →  Mayor es mejor', fontsize=12)
ax.set_ylabel('J/token  →  Menor es mejor  ↓', fontsize=12)
ax.set_title('Fig. 3 — Pareto Frontier: Trade-off Throughput vs Eficiencia Energética',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig3_pareto_frontier.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig3_pareto_frontier.png")
print(f"  Configs Pareto-óptimas: {len(pareto)}")

# ─── FIGURA 4: CEI Heatmap ────────────────────────────────────────────────────
print("[Fig 4] CEI Heatmap")

pivot = per_config.groupby(
    ['vi1_quantization', 'vi2_batch_size']
)['cei'].mean().unstack()

# Reordenar filas para que FP16 sea primera
row_order = [q for q in ['fp16', 'int8_w8a16', 'int4_awq'] if q in pivot.index]
pivot = pivot.loc[row_order]
pivot.index = [QUANT_LABELS[q] for q in row_order]

fig, ax = plt.subplots(figsize=(7, 4))
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'CEI (mayor = mejor)'})
ax.set_title('Fig. 4 — Composite Efficiency Index (CEI)\nCuantización × Batch Size (promedio sobre VI3, VI4)',
             fontsize=12, fontweight='bold')
ax.set_xlabel('Batch Size (VI2)', fontsize=11)
ax.set_ylabel('Cuantización (VI1)', fontsize=11)
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig4_cei_heatmap.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig4_cei_heatmap.png")

# ─── FIGURA 5: Varianza inter-repetición (boxplot) ───────────────────────────
print("[Fig 5] Boxplot varianza inter-repetición")

# Usar throughput por rep para cada config
data_box = []
labels_box = []
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    vals = per_rep[per_rep['vi1_quantization'] == quant]['throughput'].values
    data_box.append(vals)
    labels_box.append(QUANT_LABELS[quant])

fig, ax = plt.subplots(figsize=(8, 5))
bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True, notch=False)
for patch, quant in zip(bp['boxes'], ['fp16', 'int8_w8a16', 'int4_awq']):
    patch.set_facecolor(QUANT_COLORS[quant])
    patch.set_alpha(0.7)

ax.set_ylabel('Throughput (tok/s)', fontsize=12)
ax.set_title('Fig. 5 — Varianza Inter-Repetición por Cuantización\n(validación metodológica)',
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig5_varianza_boxplot.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig5_varianza_boxplot.png")

# ─── FIGURA 6: EOS Truncation — INT8 vs FP16 vs AWQ ─────────────────────────
print("[Fig 6] EOS Truncation — fracción de max_tokens generada por request")

df_ok['tok_per_req_tmp'] = df_ok['completion_tokens'] / df_ok['batch_actual_size']
df_ok['eos_ratio'] = df_ok['tok_per_req_tmp'] / df_ok['vi3_output_length']

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
out_labels = {64: 'output=64', 256: 'output=256', 512: 'output=512'}

for ax_idx, out_len in enumerate([64, 256, 512]):
    ax = axes[ax_idx]
    data_eos = []
    tick_labels = []
    for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
        vals = df_ok[(df_ok['vi1_quantization'] == quant) &
                     (df_ok['vi3_output_length'] == out_len)]['eos_ratio'].values
        data_eos.append(vals)
        tick_labels.append(QUANT_LABELS[quant])
    bp = ax.boxplot(data_eos, tick_labels=tick_labels, patch_artist=True)
    for patch, quant in zip(bp['boxes'], ['fp16', 'int8_w8a16', 'int4_awq']):
        patch.set_facecolor(QUANT_COLORS[quant])
        patch.set_alpha(0.7)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.2, label='max_new_tokens')
    ax.set_title(out_labels[out_len], fontsize=11)
    ax.set_ylabel('Fracción de max_tokens generada' if ax_idx == 0 else '')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Fig. 6 — EOS Truncation: Fracción de max_tokens Generada por Request\n'
             'INT8 siempre genera 1.000 (EOS suprimido); FP16/AWQ cortan antes',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{OUT_DIR}/fig6_eos_truncation.png', dpi=150)
plt.close()
print(f"  → {OUT_DIR}/fig6_eos_truncation.png")

# ─── TABLA DE DECISIONES ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TABLA DE DECISIONES")
print("=" * 60)

escenarios = {
    'Latencia mínima (chatbot single-user)':
        {'filter': {'vi2_batch_size': 1, 'vi3_output_length': 64}, 'metric': 'tpot_mean', 'best': 'min'},
    'Throughput máximo (batch processing)':
        {'filter': {'vi2_batch_size': 8, 'vi3_output_length': 512}, 'metric': 'throughput_mean', 'best': 'max'},
    'Eficiencia energética (edge/green)':
        {'filter': {}, 'metric': 'j_per_token_mean', 'best': 'min'},
    'Balance throughput/energía (CEI)':
        {'filter': {}, 'metric': 'cei', 'best': 'max'},
}

decision_rows = []
for escenario, params in escenarios.items():
    d = per_config.copy()
    for col, val in params['filter'].items():
        d = d[d[col] == val]
    if params['best'] == 'min':
        best = d.loc[d[params['metric']].idxmin()]
    else:
        best = d.loc[d[params['metric']].idxmax()]
    row = {
        'Escenario': escenario,
        'Cuantización': QUANT_LABELS[best['vi1_quantization']],
        'Batch': int(best['vi2_batch_size']),
        'Output': int(best['vi3_output_length']),
        'Contexto': best['vi4_context_case'],
        params['metric']: f"{best[params['metric']]:.3f}",
    }
    decision_rows.append(row)
    print(f"\n  {escenario}")
    print(f"    → {QUANT_LABELS[best['vi1_quantization']]} | batch={int(best['vi2_batch_size'])} "
          f"| output={int(best['vi3_output_length'])} | {params['metric']}={best[params['metric']]:.3f}")

# ─── VERIFICACIONES METODOLÓGICAS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFICACIONES METODOLÓGICAS")
print("=" * 60)

# CV > 15%
cv_high = per_config[per_config['cv_throughput'] > 0.15]
print(f"\nConfigs con CV de throughput > 15%: {len(cv_high)}")
if len(cv_high) > 0:
    print(cv_high[config_cols + ['cv_throughput']].to_string())

# EOS truncation — usar tok_per_req = completion_tokens / batch_actual_size
print("\nEOS Truncation (por request, corregido por batch_actual_size):")
df_ok['tok_per_req'] = df_ok['completion_tokens'] / df_ok['batch_actual_size']
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    for out_len in [64, 256, 512]:
        d2 = df_ok[(df_ok['vi1_quantization'] == quant) & (df_ok['vi3_output_length'] == out_len)]
        ratio = d2['tok_per_req'] / out_len
        eos_early = (ratio < 0.99).sum()
        print(f"  {QUANT_LABELS[quant]:12s} output={out_len:3d}: "
              f"generado al {ratio.mean()*100:5.1f}% de max | "
              f"EOS early: {eos_early}/{len(ratio)} ({eos_early/len(ratio)*100:.1f}%)")

# HALLAZGO CRITICO: INT8 no genera EOS
print("""
HALLAZGO CRITICO - INT8 EOS SUPRIMIDO:
  INT8 W8A16 (bitsandbytes) NUNCA genera token EOS early.
  Siempre produce exactamente max_new_tokens por request (ratio=1.000, 0% EOS early).
  FP16 y AWQ si cortan con EOS (output=256: ~67-73% de calls generan menos tokens).

  CAUSA PROBABLE: bitsandbytes W8A16 runtime quantization altera la distribucion
  de logits del token EOS (id=128009 en LLaMA 3.1). Conocido en vLLM 0.5.x + bb <= 0.43.

  IMPACTO EN METRICAS:
  - throughput_tok_s INFLADO para INT8 en output=256 y 512
    (INT8 genera ~2x mas tokens que FP16/AWQ en la misma config)
  - j_per_token_real CORRECTO (normaliza por tokens reales generados)
  - energy_j por call MAYOR para INT8 (run mas largo, no por eficiencia)

  ACCION EN PAPER:
  - Usar j_per_token como metrica primaria de energia (es justa entre cuantizaciones)
  - Comparar throughput SOLO para output=64 (donde el sesgo es minimo: FP16 3.3% EOS)
  - Documentar como limitacion: INT8 runtime quantization con bitsandbytes
""")

# NVML samples
print(f"NVML sample_count media: {df_ok['nvml_sample_count'].mean():.1f} | "
      f"calls con <5 samples: {(df_ok['nvml_sample_count'] < 5).sum()}")

# ─── RESUMEN NUMÉRICO DE HALLAZGOS ───────────────────────────────────────────
print("\n" + "=" * 60)
print("RESUMEN NUMÉRICO — HALLAZGOS DEL PAPER")
print("=" * 60)

# H1: Escalado superlineal del throughput con batch
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    base = per_config[
        (per_config['vi1_quantization'] == quant) &
        (per_config['vi2_batch_size'] == 1)
    ]['throughput_mean'].mean()
    for bs in [4, 8]:
        val = per_config[
            (per_config['vi1_quantization'] == quant) &
            (per_config['vi2_batch_size'] == bs)
        ]['throughput_mean'].mean()
        pct = (val / base - 1) * 100
        print(f"  {QUANT_LABELS[quant]:12s} batch={bs}: {val:.1f} tok/s (+{pct:.0f}% vs batch=1)")

# H2: Comparación INT8 vs FP16
print("\nINT8 vs FP16 (throughput relativo):")
for bs in [1, 4, 8]:
    fp16 = per_config[(per_config['vi1_quantization'] == 'fp16') &
                      (per_config['vi2_batch_size'] == bs)]['throughput_mean'].mean()
    int8 = per_config[(per_config['vi1_quantization'] == 'int8_w8a16') &
                      (per_config['vi2_batch_size'] == bs)]['throughput_mean'].mean()
    print(f"  batch={bs}: FP16={fp16:.1f} tok/s | INT8={int8:.1f} tok/s | ratio={int8/fp16:.2f}x")

# H3: VRAM por cuantización
print("\nVRAM peak promedio por cuantización (MB):")
for quant in ['fp16', 'int8_w8a16', 'int4_awq']:
    vram = df_ok[df_ok['vi1_quantization'] == quant]['vram_peak_mb'].mean()
    print(f"  {QUANT_LABELS[quant]:12s}: {vram:.0f} MB")

# H5: Impacto de contexto largo
print("\nImpacto contexto largo en throughput (FP16, batch=1, output=256):")
for case in ['A', 'B', 'C']:
    val = per_config[
        (per_config['vi1_quantization'] == 'fp16') &
        (per_config['vi2_batch_size'] == 1) &
        (per_config['vi3_output_length'] == 256) &
        (per_config['vi4_context_case'] == case)
    ]['throughput_mean'].values
    if len(val):
        print(f"  Case {case}: {val[0]:.1f} tok/s")

print("\n" + "=" * 60)
print("ANÁLISIS COMPLETO. Figuras en:", OUT_DIR)
print("=" * 60)

# ─── EXPORTAR DATOS PROCESADOS ───────────────────────────────────────────────
per_config.to_csv('analysis/per_config_summary.csv', index=False)
per_rep.to_csv('analysis/per_rep_summary.csv', index=False)
print("CSVs exportados: analysis/per_config_summary.csv | per_rep_summary.csv")
