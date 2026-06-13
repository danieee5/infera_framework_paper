"""
infera_analysis.py  (v3 — Fase 2a)
Analisis del Protocolo C: sobre conjunto energia x calidad + recuperacion por compactacion.

Lee todos los .jsonl de results/ y produce:
  1. Curvas energia/tarea y calidad vs contexto acumulado (brazo naive, por cuantizacion).
  2. Deteccion del CODO usando tareas SONDA (faciles por diseno, lookup de KB puro):
     si la calidad de una SONDA cae < 0.5 es context-rot real, no dificultad intrinseca.
     Fallback al detector clasico si no hay sondas en los datos.
  3. [v3 NUEVO] threshold_bracket: localiza el bracket exacto [lo, hi] entre dos SONDAS
     consecutivas donde quality transiciona de >=0.5 a <0.5 en el brazo naive.
     Cuantifica el ancho de la ventana en tokens.
  4. Figura insignia: doble eje (J/tarea y calidad) sobre contexto acumulado.
  5. [v3 NUEVO] Curva dosis-respuesta: calidad de SONDA vs contexto acumulado.
     Una linea por quant x arm (x session si hay varias). Esta es la figura central
     del hallazgo de context-rot dependiente de cuantizacion.
  6. Recuperacion: naive vs compaction (energia acumulada, calidad, impuesto, break-even).
  7. Desglose de calidad por task_id x quant x arm.
  8. Regresion OLS energy_j ~ alpha*ctx_tokens + beta*completion_tokens.

Compatibilidad: los archivos de salida de v2 (knees.json, energy_regression.json,
recovery_naive_vs_compaction.csv, quality_by_task.csv, sonda_recall_summary.csv)
se generan igual. Las nuevas salidas son adicionales.

Uso:
  python infera_analysis.py --results results --out results/analysis
  python infera_analysis.py --results results --out results/analysis --session v3
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Carga — con soporte multi-sesion
# ---------------------------------------------------------------------------

def load_runs(results_dir: str) -> pd.DataFrame:
    """
    Lee todos los .jsonl de results_dir.
    Extrae session_tag del nombre del archivo:
      run_v3_AWQ_naive_rep1.jsonl       -> session_tag = "v3"
      run_v3_filler_AWQ_naive_rep1.jsonl -> session_tag = "v3_filler"
      run_AWQ_naive_rep1.jsonl           -> session_tag = "v2"  (compat hacia atras)
    """
    rows = []
    for fp in Path(results_dir).glob("*.jsonl"):
        # Extraer session_tag entre "run_" y "_(AWQ|FP16|INT8)"
        m = re.match(r"run_(.+?)_(AWQ|FP16|INT8)", fp.stem)
        session_tag = m.group(1) if m else "v2"

        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["session_tag"] = session_tag
                rows.append(row)

    if not rows:
        raise SystemExit(f"No se encontraron .jsonl en {results_dir}")
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Deteccion del CODO — igual que v2
# ---------------------------------------------------------------------------

def detect_knee_sonda(df: pd.DataFrame, quant: str):
    sub = df[
        (df["quant"] == quant) &
        (df["arm"] == "naive") &
        (~df["is_compaction"]) &
        (df["task_type"] == "SONDA") &
        df["quality"].notna()
    ].copy()
    if sub.empty:
        return None
    agg = (sub.groupby("task_id")
           .agg(ctx=("accumulated_prompt_tokens", "mean"),
                quality=("quality", "mean"))
           .reset_index()
           .sort_values("ctx"))
    for _, row in agg.iterrows():
        if row["quality"] < 0.5:
            return {
                "task_id": str(row["task_id"]),
                "accumulated_tokens": int(round(row["ctx"])),
                "quality": round(float(row["quality"]), 4),
                "detector": "SONDA",
                "note": "Tarea facil (KB lookup puro) que falla: signal limpia de context-rot.",
            }
    return None


def detect_knee_classic(curve: pd.DataFrame, quality_drop_frac: float = 0.85):
    c = curve[curve["task_type"] != "CONSTRAINT"].sort_values("accumulated_prompt_tokens")
    c = c[c["quality"].notna()]
    if len(c) < 3:
        return None
    n_third = max(1, len(c) // 3)
    base_q = c["quality"].iloc[:n_third].mean()
    med_jpt = c["j_per_completion_token"].median()
    for _, row in c.iterrows():
        if (row["quality"] < quality_drop_frac * base_q
                and (row.get("j_per_completion_token") or 0) >= (med_jpt or 0)):
            return {
                "task_id": str(row.get("task_id", "?")),
                "accumulated_tokens": int(row["accumulated_prompt_tokens"]),
                "quality": round(float(row["quality"]), 4),
                "base_quality": round(float(base_q), 3),
                "detector": "clasico",
                "note": f"Heuristica: calidad < {quality_drop_frac:.0%} de la base y J/tok >= mediana.",
            }
    return None


def detect_knee(df: pd.DataFrame, quant: str):
    knee = detect_knee_sonda(df, quant)
    if knee is not None:
        return knee
    sub = df[(df["quant"] == quant) & (df["arm"] == "naive") & (~df["is_compaction"])]
    if sub.empty:
        return None
    curve = sub.groupby("task_id").agg(
        accumulated_prompt_tokens=("accumulated_prompt_tokens", "mean"),
        quality=("quality", "mean"),
        j_per_completion_token=("j_per_completion_token", "mean"),
        task_type=("task_type", "first"),
    ).reset_index()
    return detect_knee_classic(curve)


# ---------------------------------------------------------------------------
# [v3 NUEVO] Bracket del umbral: ventana exacta entre dos SONDAS consecutivas
# ---------------------------------------------------------------------------

def threshold_bracket(df: pd.DataFrame) -> dict:
    """
    Para cada cuantizacion, encuentra el par [lo, hi] de sondas consecutivas
    (brazo naive) donde quality transiciona de >=0.5 a <0.5.

    Devuelve:
      {quant: {"lo": {"task_id", "ctx", "quality"},
               "hi": {"task_id", "ctx", "quality"},
               "window_tokens": hi.ctx - lo.ctx,
               "note": str} or None si no hay transicion}

    Con las sondas densas de v3, la ventana deberia ser ~200-400 tokens
    vs ~1036 tokens en v2 (solo SONDA_B/SONDA_C). Eso es el objetivo.
    """
    results = {}
    for quant in sorted(df["quant"].unique()):
        sub = df[
            (df["quant"] == quant) &
            (df["arm"] == "naive") &
            (~df["is_compaction"]) &
            (df["task_type"] == "SONDA") &
            df["quality"].notna()
        ]
        if sub.empty:
            results[quant] = None
            continue

        agg = (sub.groupby("task_id")
               .agg(ctx=("accumulated_prompt_tokens", "mean"),
                    quality=("quality", "mean"))
               .reset_index()
               .sort_values("ctx"))

        lo = None
        hi = None
        for _, row in agg.iterrows():
            if row["quality"] >= 0.5:
                lo = {
                    "task_id": str(row["task_id"]),
                    "ctx": int(round(row["ctx"])),
                    "quality": round(float(row["quality"]), 4),
                }
            else:
                hi = {
                    "task_id": str(row["task_id"]),
                    "ctx": int(round(row["ctx"])),
                    "quality": round(float(row["quality"]), 4),
                }
                break  # primer fallo encontrado

        if lo and hi:
            window = hi["ctx"] - lo["ctx"]
            results[quant] = {
                "lo": lo,
                "hi": hi,
                "window_tokens": window,
                "note": (
                    f"Umbral acotado: ({lo['ctx']}, {hi['ctx']}) tokens "
                    f"(ventana de {window} tokens). "
                    f"Lo={lo['task_id']} Q={lo['quality']}, "
                    f"Hi={hi['task_id']} Q={hi['quality']}."
                ),
            }
        elif hi and not lo:
            results[quant] = {
                "lo": None, "hi": hi,
                "window_tokens": None,
                "note": "Todas las sondas fallan — no hay sondas pasantes para acotar el lado inferior.",
            }
        else:
            results[quant] = None  # no hay fallo, no hay codo en este rango
    return results


# ---------------------------------------------------------------------------
# [v3 NUEVO] Curva dosis-respuesta de SONDAS
# ---------------------------------------------------------------------------

def dose_response_plot(df: pd.DataFrame, out_path: Path):
    """
    Figura central del paper: calidad de SONDA vs contexto acumulado.
    Una curva por quant x arm. Si hay multiples session_tags, diferencia
    con marcador de forma.

    Esta figura muestra visualmente:
    - El codo de AWQ (naive) entre dos sondas consecutivas.
    - Que FP16 no cae en el mismo rango.
    - Que compaction mantiene quality=1.0 para AWQ (al reiniciar ctx < umbral).
    """
    sub = df[(df["task_type"] == "SONDA") & (~df["is_compaction"])].copy()
    if sub.empty:
        return

    sessions = sorted(sub["session_tag"].unique()) if "session_tag" in sub.columns else ["default"]
    quants   = sorted(sub["quant"].unique())
    arms     = sorted(sub["arm"].unique())

    # Paleta reproducible
    quant_colors = {"AWQ": "tab:orange", "FP16": "tab:blue", "INT8": "tab:green"}
    arm_styles   = {"naive": "-",        "compaction": "--"}
    session_markers = {s: m for s, m in zip(sessions, ["o", "s", "^", "D"])}

    fig, ax = plt.subplots(figsize=(11, 6))

    for session in sessions:
        for quant in quants:
            for arm in arms:
                mask = (
                    (sub["quant"] == quant) &
                    (sub["arm"] == arm)
                )
                if "session_tag" in sub.columns:
                    mask &= (sub["session_tag"] == session)
                s = sub[mask]
                if s.empty:
                    continue

                agg = (s.groupby("task_id")
                       .agg(ctx=("accumulated_prompt_tokens", "mean"),
                            quality=("quality", "mean"))
                       .reset_index()
                       .sort_values("ctx"))
                if agg.empty:
                    continue

                # Etiqueta de leyenda: omite session si solo hay una
                label = f"{quant} {arm}"
                if len(sessions) > 1:
                    label += f" [{session}]"

                ax.plot(
                    agg["ctx"], agg["quality"],
                    linestyle=arm_styles.get(arm, "-"),
                    color=quant_colors.get(quant, "gray"),
                    marker=session_markers.get(session, "o"),
                    markersize=6,
                    linewidth=2,
                    label=label,
                )

    # Linea de umbral de context-rot
    ax.axhline(0.5, color="red", linestyle=":", linewidth=1.5,
               label="Umbral context-rot (quality=0.5)")

    ax.set_xlabel("Contexto acumulado (tokens de prompt)", fontsize=12)
    ax.set_ylabel("Calidad en SONDA (0=fallo / 1=correcto)", fontsize=12)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.set_title(
        "Curva dosis-respuesta de context-rot: calidad de SONDA vs contexto acumulado",
        fontsize=11,
    )
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  [OK] Curva dosis-respuesta -> {out_path}")


# ---------------------------------------------------------------------------
# Figura insignia: doble eje energia x calidad — igual que v2
# ---------------------------------------------------------------------------

def plot_envelope(df: pd.DataFrame, quant: str, out_path: Path, knee=None):
    sub = df[(df["quant"] == quant) & (df["arm"] == "naive") & (~df["is_compaction"])]
    if sub.empty:
        return
    agg = sub.groupby("task_id").agg(
        ctx=("accumulated_prompt_tokens", "mean"),
        energy=("energy_j", "mean"),
        quality=("quality", "mean"),
        task_type=("task_type", "first"),
    ).reset_index().sort_values("ctx")

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel("Contexto acumulado (tokens de prompt)")
    ax1.set_ylabel("Energia por tarea (J)", color="tab:red")
    ax1.plot(agg["ctx"], agg["energy"], "o-", color="tab:red", label="Energia/tarea (J)")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    sondas = agg[agg["task_type"] == "SONDA"]
    if not sondas.empty:
        ax1.scatter(sondas["ctx"], sondas["energy"], marker="^", s=100,
                    color="darkred", zorder=5, label="SONDA (energia)")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Calidad de tarea (0-1)", color="tab:blue")
    ax2.plot(agg["ctx"], agg["quality"], "s--", color="tab:blue", label="Calidad")
    ax2.set_ylim(-0.05, 1.10)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    if not sondas.empty:
        ax2.scatter(sondas["ctx"], sondas["quality"], marker="^", s=100,
                    color="navy", zorder=5, label="SONDA (calidad)")

    if knee:
        ax1.axvline(knee["accumulated_tokens"], color="gray", linestyle=":", linewidth=1.5)
        ax1.annotate(
            f"CODO ~{knee['accumulated_tokens']} tok\n(pagas mas, calidad cae)",
            xy=(knee["accumulated_tokens"], ax1.get_ylim()[1] * 0.9),
            fontsize=9, color="gray",
        )

    plt.title(f"Sobre conjunto Energia x Calidad vs contexto acumulado — {quant} (naive)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Desglose de calidad por tarea — igual que v2
# ---------------------------------------------------------------------------

def quality_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    tasks = df[~df["is_compaction"]].copy()
    tbl = (tasks.groupby(["task_id", "task_type", "quant", "arm"])
           .agg(
               ctx_media=("accumulated_prompt_tokens", "mean"),
               quality_media=("quality", "mean"),
               quality_std=("quality", "std"),
               n=("quality", "count"),
           )
           .round(4)
           .reset_index()
           .sort_values(["task_id", "quant", "arm"]))
    return tbl


def sonda_recall_summary(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["task_type"].isin(["SONDA", "RECALL"]) & ~df["is_compaction"]].copy()
    if sub.empty:
        return pd.DataFrame()
    tbl = (sub.groupby(["task_id", "task_type", "quant", "arm"])
           .agg(
               ctx_media=("accumulated_prompt_tokens", "mean"),
               quality_media=("quality", "mean"),
               n=("quality", "count"),
           )
           .round(3)
           .reset_index()
           .sort_values(["task_id", "quant", "arm"]))
    return tbl


# ---------------------------------------------------------------------------
# Regresion OLS: descomposicion prefill vs decode — igual que v2
# ---------------------------------------------------------------------------

def energy_regression(df: pd.DataFrame) -> dict:
    """
    OLS:  energy_j = alpha * accumulated_prompt_tokens
                   + beta  * completion_tokens
                   + intercept

    alpha = coste marginal de prefill (J por token de contexto acumulado).
    beta  = coste marginal de decode  (J por token generado).

    Solo corridas naive (sin compactacion) para no confundir con reinicios de contexto.
    Reporta coeficientes, R2 y n por cuantizacion y global.
    """
    results = {}
    base = df[~df["is_compaction"] & df["completion_tokens"].notna()].copy()
    base = base[base["completion_tokens"] > 0]

    quants_list = sorted(base["quant"].unique().tolist()) + ["GLOBAL"]
    for quant in quants_list:
        sub = base if quant == "GLOBAL" else base[base["quant"] == quant]
        if len(sub) < 5:
            continue
        X = np.column_stack([
            sub["accumulated_prompt_tokens"].values.astype(float),
            sub["completion_tokens"].values.astype(float),
            np.ones(len(sub)),
        ])
        y = sub["energy_j"].values.astype(float)
        coefs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coefs
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        results[quant] = {
            "alpha_j_per_ctx_token": round(float(coefs[0]), 5),
            "beta_j_per_completion_token": round(float(coefs[1]), 5),
            "intercept_j": round(float(coefs[2]), 2),
            "r2": round(float(r2), 4),
            "n": int(len(sub)),
            "nota": (
                "alpha=J/token-de-contexto (prefill). "
                "beta=J/token-generado (decode). "
                "R2 mide cuanto de la varianza de energy_j explica el modelo lineal."
            ),
        }
    return results


# ---------------------------------------------------------------------------
# Tabla de recuperacion naive vs compaction — igual que v2
# ---------------------------------------------------------------------------

def recovery_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quant in sorted(df["quant"].unique()):
        for arm in ["naive", "compaction"]:
            sub = df[(df["quant"] == quant) & (df["arm"] == arm)]
            if sub.empty:
                continue
            tasks = sub[~sub["is_compaction"]]
            comp  = sub[sub["is_compaction"]]
            per_rep     = sub.groupby("rep")["energy_j"].sum()
            tax_per_rep = comp.groupby("rep")["energy_j"].sum() if not comp.empty else None
            rows.append({
                "quant": quant,
                "arm": arm,
                "energia_total_J_media": round(per_rep.mean(), 2),
                "calidad_media": round(tasks["quality"].mean(), 3),
                "impuesto_compactacion_J": round(tax_per_rep.mean(), 2) if tax_per_rep is not None else 0.0,
                "n_reps": int(sub["rep"].nunique()),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/analisis")
    ap.add_argument("--session", default=None,
                    help="Filtrar por session_tag (ej: v3, v3_filler). Default: todos.")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.results)

    # Mostrar que sesiones se encontraron
    sessions_found = sorted(df["session_tag"].unique()) if "session_tag" in df.columns else []
    print(f"[OK] {len(df)} registros de {df['run_id'].nunique() if 'run_id' in df.columns else '?'} corridas.")
    if sessions_found:
        print(f"     Sesiones encontradas: {sessions_found}")

    # Filtrar por sesion si se especifica
    if args.session:
        df = df[df["session_tag"] == args.session]
        if df.empty:
            raise SystemExit(f"No hay datos para session_tag='{args.session}'. "
                             f"Sesiones disponibles: {sessions_found}")
        print(f"     Filtrando a session_tag='{args.session}': {len(df)} registros.")

    df.to_csv(out / "all_runs_long.csv", index=False)

    # ------------------------------------------------------------------
    # 1. Deteccion del CODO (igual que v2)
    # ------------------------------------------------------------------
    print("\n=== Deteccion del CODO (context-rot) ===")
    knees = {}
    for quant in sorted(df["quant"].unique()):
        knee = detect_knee(df, quant)
        knees[quant] = knee
        if knee:
            print(f"  [{knee['detector'].upper()}] {quant}: ~{knee['accumulated_tokens']} tokens "
                  f"| task={knee.get('task_id','?')} | quality={knee['quality']:.4f}")
        else:
            print(f"  {quant}: codo no detectado en el rango actual.")
        plot_envelope(df, quant, out / f"envelope_{quant}.png", knee)

    with open(out / "knees.json", "w", encoding="utf-8") as f:
        json.dump(knees, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 2. [v3 NUEVO] Bracket del umbral — ventana exacta
    # ------------------------------------------------------------------
    print("\n=== Bracket del umbral de context-rot (sondas densas) ===")
    brackets = threshold_bracket(df)
    for quant, br in brackets.items():
        if br is None:
            print(f"  {quant}: sin transicion detectada (todas las sondas pasan o todas fallan).")
        elif br.get("lo") is None:
            print(f"  {quant}: {br['note']}")
        else:
            print(f"  {quant}: [{br['lo']['task_id']} ctx={br['lo']['ctx']} Q={br['lo']['quality']:.2f}]"
                  f" --> [{br['hi']['task_id']} ctx={br['hi']['ctx']} Q={br['hi']['quality']:.2f}]"
                  f"  ventana={br['window_tokens']} tokens")

    with open(out / "threshold_brackets.json", "w", encoding="utf-8") as f:
        json.dump(brackets, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 3. [v3 NUEVO] Curva dosis-respuesta de SONDAS
    # ------------------------------------------------------------------
    print("\n=== Curva dosis-respuesta de SONDAS ===")
    dose_response_plot(df, out / "dose_response_sondas.png")

    # ------------------------------------------------------------------
    # 4. Desglose de calidad por tarea (igual que v2)
    # ------------------------------------------------------------------
    print("\n=== Desglose calidad: SONDA + RECALL x quant x arm ===")
    qb = quality_breakdown(df)
    qb.to_csv(out / "quality_by_task.csv", index=False)

    ss = sonda_recall_summary(df)
    if not ss.empty:
        ss.to_csv(out / "sonda_recall_summary.csv", index=False)
        print(ss.to_string(index=False))
    else:
        print("  (sin tareas SONDA ni RECALL en los datos)")

    # ------------------------------------------------------------------
    # 5. Regresion OLS prefill vs decode (igual que v2)
    # ------------------------------------------------------------------
    print("\n=== Regresion OLS: energy_j ~ alpha*ctx + beta*completion ===")
    reg = energy_regression(df)
    with open(out / "energy_regression.json", "w", encoding="utf-8") as f:
        json.dump(reg, f, indent=2, ensure_ascii=False)
    for quant, r in reg.items():
        print(f"  {quant:8s}  alpha={r['alpha_j_per_ctx_token']:+.5f} J/ctx-tok"
              f"  beta={r['beta_j_per_completion_token']:+.5f} J/compl-tok"
              f"  R2={r['r2']:.4f}  n={r['n']}")

    # ------------------------------------------------------------------
    # 6. Tabla de recuperacion naive vs compaction (igual que v2)
    # ------------------------------------------------------------------
    rec = recovery_table(df)
    rec.to_csv(out / "recovery_naive_vs_compaction.csv", index=False)
    print("\n=== Recuperacion naive vs compaction ===")
    print(rec.to_string(index=False))

    print(f"\n[OK] Figuras y tablas en {out}/")
    print("  Archivos generados:")
    for f_path in sorted(out.iterdir()):
        print(f"    {f_path.name}")


if __name__ == "__main__":
    main()
