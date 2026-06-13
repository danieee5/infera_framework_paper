"""
multiturn_analysis.py
Análisis del Experimento 2: Evolución energética en conversación multi-turno.

Carga los resultados de multiturn_runner.py y produce:
  1. Tabla resumen: J/output_token por (quantization, batch_size, turn)
  2. Test H1: ¿crece J/tok monotonically con turn_number?
  3. Test H2: ¿INT8 batch=4 anomaly amplifies in later turns?
  4. Test H3: ¿AWQ energy advantage erodes vs FP16 as turns increase?
  5. KV-cache proxy: VRAM_peak vs turn_number por esquema
  6. TTFT vs TPOT por turno: ¿prefill domina en turnos tardíos?

USAGE:
  python scripts/multiturn_analysis.py \
    --results-dir results/multiturn/fp16_* results/multiturn/int8_* results/multiturn/awq_*
  
  # O directamente apuntando al directorio padre:
  python scripts/multiturn_analysis.py --results-dir results/multiturn/
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("FATAL: pandas + numpy required. Run: pip install pandas numpy")


# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_results(paths: list[Path]) -> pd.DataFrame:
    """Load all .jsonl and individual .json result files into one DataFrame."""
    records = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            # Try JSONL first
            jsonl_files = list(p.glob("multiturn_results.jsonl"))
            for f in jsonl_files:
                with f.open(encoding="utf-8") as fh:
                    for line in fh:
                        records.append(json.loads(line.strip()))
            # Also individual JSON files (crash recovery)
            for f in p.glob("mt_*.json"):
                records.append(json.loads(f.read_text(encoding="utf-8")))
        elif p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    records.append(json.loads(line.strip()))

    df = pd.DataFrame(records)
    df = df[df["status"] == "success"].copy()

    # Deduplicate (in case both JSONL and individual JSONs were loaded)
    df = df.drop_duplicates(subset=["run_id"])

    print(f"  Loaded {len(df)} successful measurements")
    return df


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean J/output_token per (quantization, batch_size, turn_number)."""
    grp = df.groupby(
        ["vi1_quantization", "vi2_batch_size", "turn_number"]
    )["j_per_output_token"].agg(["mean", "std", "count"]).reset_index()
    grp.columns = ["quantization", "batch_size", "turn", "j_per_tok_mean",
                   "j_per_tok_std", "n_reps"]
    grp["cv_pct"] = (grp["j_per_tok_std"] / grp["j_per_tok_mean"] * 100).round(2)
    return grp


# ── H1: ENERGY ESCALATION ─────────────────────────────────────────────────────

def test_h1(df: pd.DataFrame) -> dict:
    """
    H1: J/output_token increases monotonically with turn number.
    Test: for each (quant, batch), is the Spearman rank correlation
    between turn_number and j_per_output_token positive?
    """
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  [H1] scipy not available — skipping statistical test")
        return {}

    results = {}
    for (q, b), grp in df.groupby(["vi1_quantization", "vi2_batch_size"]):
        turn_means = grp.groupby("turn_number")["j_per_output_token"].mean()
        rho, p = spearmanr(turn_means.index, turn_means.values)
        results[(q, b)] = {"rho": round(rho, 4), "p": round(p, 4),
                            "monotone": rho > 0 and p < 0.05}
    return results


# ── H2: INT8 ANOMALY PERSISTENCE ─────────────────────────────────────────────

def test_h2(df: pd.DataFrame) -> pd.DataFrame:
    """
    H2: INT8 batch=4 energy anomaly (J/tok_b4 > J/tok_b1) amplifies in later turns.
    Output: ratio = (INT8 b4 J/tok) / (INT8 b1 J/tok) per turn.
    Ratio > 1.0 means anomaly is present; trend matters.
    """
    int8 = df[df["vi1_quantization"] == "int8_w8a16"].copy()
    if int8.empty or 4 not in int8["vi2_batch_size"].values:
        print("  [H2] INT8 or batch=4 data missing — skipping")
        return pd.DataFrame()

    b1 = int8[int8["vi2_batch_size"] == 1].groupby("turn_number")["j_per_output_token"].mean()
    b4 = int8[int8["vi2_batch_size"] == 4].groupby("turn_number")["j_per_output_token"].mean()

    ratio = (b4 / b1).reset_index()
    ratio.columns = ["turn", "int8_b4_vs_b1_ratio"]
    ratio["anomaly_present"] = ratio["int8_b4_vs_b1_ratio"] > 1.0
    return ratio


# ── H3: AWQ ADVANTAGE EROSION ────────────────────────────────────────────────

def test_h3(df: pd.DataFrame) -> pd.DataFrame:
    """
    H3: AWQ energy advantage over FP16 erodes as turns increase.
    Output: ratio = (AWQ b1 J/tok) / (FP16 b1 J/tok) per turn.
    Ratio < 1.0 means AWQ is more efficient; trend toward 1.0 means erosion.

    Bridge to static factorial: at short context (T1 ≈ Case_A) we expect
    ~0.49 (AWQ ~51% of FP16 J/tok). At long context (T7 ≈ Case_C) we expect
    the ratio to rise toward ~0.69 (consistent with +122% AWQ vs +59% FP16
    from Case_A to Case_C in the static experiment).
    """
    b1 = df[df["vi2_batch_size"] == 1]
    fp16 = b1[b1["vi1_quantization"] == "fp16"].groupby("turn_number")["j_per_output_token"].mean()
    awq  = b1[b1["vi1_quantization"] == "int4_awq"].groupby("turn_number")["j_per_output_token"].mean()

    if fp16.empty or awq.empty:
        print("  [H3] FP16 or AWQ data missing — skipping")
        return pd.DataFrame()

    ratio = (awq / fp16).reset_index()
    ratio.columns = ["turn", "awq_vs_fp16_ratio"]
    ratio["awq_advantage_pct"] = ((1 - ratio["awq_vs_fp16_ratio"]) * 100).round(1)
    return ratio


# ── KV-CACHE PROXY ───────────────────────────────────────────────────────────

def vram_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    VRAM peak per turn as proxy for KV-cache growth.
    Each quantization scheme loads model weights at different sizes;
    the *delta* VRAM (peak - start) more directly reflects KV-cache.
    """
    df["vram_delta_mb"] = df["vram_peak_mb"] - df["vram_start_mb"]
    grp = df.groupby(
        ["vi1_quantization", "vi2_batch_size", "turn_number"]
    )[["vram_peak_mb", "vram_delta_mb"]].mean().reset_index()
    return grp


# ── TTFT vs TPOT ANALYSIS ────────────────────────────────────────────────────

def prefill_dominance(df: pd.DataFrame) -> pd.DataFrame:
    """
    TTFT vs TPOT per turn: does prefill dominate in later turns?
    TTFT reflects prefill cost. TPOT reflects decode cost per token.
    Rising TTFT/TPOT ratio signals increasing prefill dominance.
    """
    grp = df.groupby(
        ["vi1_quantization", "vi2_batch_size", "turn_number"]
    )[["ttft_ms", "tpot_ms"]].mean().reset_index()
    grp["ttft_tpot_ratio"] = (grp["ttft_ms"] / grp["tpot_ms"]).round(3)
    return grp


# ── BRIDGE TO STATIC EXPERIMENT ──────────────────────────────────────────────

def bridge_summary(df: pd.DataFrame) -> str:
    """
    Map multi-turn turns to approximate static factorial context cases:
      T1 ≈ Case_B (~1024 tokens)
      T4 ≈ Case_B–C transition
      T7 ≈ Case_C (~4096 tokens)

    Compare J/tok at these anchor points vs static experiment values.
    Returns a text summary for inclusion in the paper.
    """
    anchor_map = {1: "~Case_B", 4: "~Case_B/C", 7: "~Case_C"}
    b1 = df[df["vi2_batch_size"] == 1]

    lines = ["Multi-turn anchor comparison (batch=1):"]
    for turn, label in anchor_map.items():
        for q in ["fp16", "int8_w8a16", "int4_awq"]:
            sub = b1[(b1["turn_number"] == turn) & (b1["vi1_quantization"] == q)]
            if sub.empty:
                continue
            mean_jpt = sub["j_per_output_token"].mean()
            lines.append(f"  Turn {turn} ({label}) | {q:15s}: {mean_jpt:.4f} J/tok")

    return "\n".join(lines)


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    print("\n" + "="*70)
    print("  MULTI-TURN ANALYSIS REPORT — INFERA Experimento 2")
    print("="*70)

    # Summary table
    tbl = summary_table(df)
    print("\n── SUMMARY: J/output_token por (quantization, batch, turn) ──")
    for q in sorted(df["vi1_quantization"].unique()):
        print(f"\n  {q}")
        sub = tbl[tbl["quantization"] == q]
        for b in sorted(sub["batch_size"].unique()):
            row = sub[sub["batch_size"] == b]
            vals = "  ".join([f"T{r['turn']}:{r['j_per_tok_mean']:.4f}" for _, r in row.iterrows()])
            print(f"    batch={b}: {vals}")

    # H1
    print("\n── H1: Monotonic energy escalation with turn? ──")
    h1 = test_h1(df)
    for (q, b), res in sorted(h1.items()):
        verdict = "✓ CONFIRMED" if res["monotone"] else "✗ NOT confirmed"
        print(f"  {q} batch={b}: ρ={res['rho']}  p={res['p']}  → {verdict}")

    # H2
    print("\n── H2: INT8 batch=4 anomaly — does it persist/amplify? ──")
    h2 = test_h2(df)
    if not h2.empty:
        for _, row in h2.iterrows():
            flag = "⚠ ANOMALY" if row["anomaly_present"] else "OK"
            print(f"  Turn {int(row['turn'])}: INT8_b4/INT8_b1 = {row['int8_b4_vs_b1_ratio']:.4f}  {flag}")
        early = h2[h2["turn"] <= 2]["int8_b4_vs_b1_ratio"].mean()
        late  = h2[h2["turn"] >= 6]["int8_b4_vs_b1_ratio"].mean()
        print(f"  Mean ratio turns 1–2: {early:.4f}  vs  turns 6–7: {late:.4f}")
        if late > early:
            print("  → Anomaly amplifies in later turns (supports H2)")
        else:
            print("  → Anomaly does NOT amplify (H2 not supported)")

    # H3
    print("\n── H3: AWQ advantage erosion vs FP16? ──")
    h3 = test_h3(df)
    if not h3.empty:
        for _, row in h3.iterrows():
            print(f"  Turn {int(row['turn'])}: AWQ/FP16 = {row['awq_vs_fp16_ratio']:.4f}"
                  f"  → AWQ advantage: {row['awq_advantage_pct']:.1f}%")
        t1_adv = h3[h3["turn"] == 1]["awq_advantage_pct"].values[0]
        t7_adv = h3[h3["turn"] == 7]["awq_advantage_pct"].values[0]
        print(f"\n  AWQ advantage T1: {t1_adv:.1f}%  →  T7: {t7_adv:.1f}%")
        if t7_adv < t1_adv:
            delta = t1_adv - t7_adv
            print(f"  → Advantage eroded {delta:.1f} pp over conversation (supports H3)")
        else:
            print("  → Advantage maintained (H3 not supported)")

    # KV-cache proxy
    print("\n── KV-cache proxy: VRAM delta (peak − start) per turn ──")
    vram = vram_analysis(df)
    for q in sorted(vram["vi1_quantization"].unique()):
        sub = vram[(vram["vi1_quantization"] == q) & (vram["vi2_batch_size"] == 1)]
        deltas = "  ".join([f"T{int(r['turn_number'])}:{r['vram_delta_mb']:.0f}MB"
                             for _, r in sub.iterrows()])
        print(f"  {q}: {deltas}")

    # TTFT vs TPOT
    print("\n── Prefill dominance: TTFT vs TPOT per turn (batch=1) ──")
    pref = prefill_dominance(df[df["vi2_batch_size"] == 1])
    for q in sorted(pref["vi1_quantization"].unique()):
        sub = pref[pref["vi1_quantization"] == q]
        ratios = "  ".join([f"T{int(r['turn_number'])}:{r['ttft_tpot_ratio']:.2f}x"
                             for _, r in sub.iterrows()])
        print(f"  {q}: TTFT/TPOT = {ratios}")

    # Bridge to static
    print("\n── Bridge to static factorial ──")
    print(bridge_summary(df))

    # CV check
    print("\n── Reproducibilidad: CV por (quant, batch, turn) ──")
    tbl2 = summary_table(df)
    max_cv = tbl2["cv_pct"].max()
    pct_ok = (tbl2["cv_pct"] < 5.0).mean() * 100
    print(f"  CV < 5%: {pct_ok:.1f}% of configurations  (max: {max_cv:.2f}%)")

    print("\n" + "="*70 + "\n")


# ── EXPORT CSV ────────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, out_dir: Path):
    """Export key tables as CSV for paper figures."""
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_table(df).to_csv(out_dir / "mt_summary.csv", index=False)

    h2 = test_h2(df)
    if not h2.empty:
        h2.to_csv(out_dir / "mt_h2_anomaly.csv", index=False)

    h3 = test_h3(df)
    if not h3.empty:
        h3.to_csv(out_dir / "mt_h3_awq_erosion.csv", index=False)

    vram_analysis(df).to_csv(out_dir / "mt_vram_proxy.csv", index=False)
    prefill_dominance(df).to_csv(out_dir / "mt_prefill_dominance.csv", index=False)

    print(f"  CSVs exported to: {out_dir}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", nargs="+", required=True,
        help="One or more directories containing multiturn results"
    )
    parser.add_argument(
        "--export-csv", default="results/multiturn/analysis",
        help="Directory to export CSV files"
    )
    args = parser.parse_args()

    print("\n  Loading results...")
    df = load_results([Path(p) for p in args.results_dir])

    if df.empty:
        sys.exit("No successful results found.")

    print_report(df)
    export_csv(df, Path(args.export_csv))
