"""
infera_analysis.py
Analisis del Protocolo C: sobre conjunto energia x calidad + recuperacion por compactacion.

Lee todos los .jsonl de results/ y produce:
  1. Curvas energia/tarea y calidad vs contexto acumulado (brazo naive, por cuantizacion).
  2. Deteccion del CODO usando tareas SONDA (faciles por diseno, lookup de KB puro):
     si la calidad de una SONDA cae < 0.5 es context-rot real, no dificultad intrinseca.
     Fallback al detector clasico si no hay sondas en los datos.
  3. Figura insignia: doble eje (J/tarea y calidad) sobre contexto acumulado.
  4. Recuperacion: naive vs compaction (energia acumulada, calidad post-handoff,
     impuesto de compactacion, break-even).
  5. [NUEVO v2] Desglose de calidad por task_id x quant x arm — revela la curva
     dosis-respuesta de las sondas y el hallazgo de AWQ.
  6. [NUEVO v2] Regresion OLS energy_j ~ alpha*ctx_tokens + beta*completion_tokens:
     separa el coste marginal de prefill (alpha) del de decode (beta).

Uso:
  python infera_analysis.py --results results --out results/analysis
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load_runs(results_dir: str) -> pd.DataFrame:
    rows = []
    for fp in Path(results_dir).glob("*.jsonl"):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"No se encontraron .jsonl en {results_dir}")
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Deteccion del CODO
# ---------------------------------------------------------------------------

def detect_knee_sonda(df: pd.DataFrame, quant: str):
    """
    Detector primario: usa EXCLUSIVAMENTE tareas tipo SONDA.
    Estas tareas son faciles por diseno (lookup de KB puro, sin historial de sesion) —
    un fallo (quality < 0.5) es siempre context-rot real, no dificultad intrinseca.
    Devuelve dict o None.
    """
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
           .agg(
               ctx=("accumulated_prompt_tokens", "mean"),
               quality=("quality", "mean"),
           )
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
    """
    Detector de respaldo (sin sondas): heuristica sobre el brazo naive.
    Excluye CONSTRAINT del calculo de la base para no sesgar por dificultad intrinseca.
    """
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
    """
    Intenta primero con sondas; si no hay, usa el detector clasico.
    """
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
# Figura insignia: doble eje energia x calidad
# ---------------------------------------------------------------------------

def plot_envelope(df: pd.DataFrame, quant: str, out_path: Path, knee=None):
    """Figura insignia: doble eje J/tarea y calidad vs contexto acumulado (brazo naive)."""
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

    # marcar sondas con triangulo
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
# Desglose de calidad por tarea (hallazgo central)
# ---------------------------------------------------------------------------

def quality_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calidad media por task_id x task_type x quant x arm.
    Esta tabla revela la curva dosis-respuesta de las sondas:
    AWQ falla sondas a contexto alto; FP16 no. Compaction degrada RECALL en AWQ pero no FP16.
    """
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
    """
    Vista enfocada: tareas SONDA y RECALL, comparacion naive vs compaction x quant.
    Tabla principal del hallazgo de context-rot dependiente de cuantizacion.
    """
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
# Regresion OLS: descomposicion prefill vs decode
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

    Interpretacion tipica:
      - beta >> alpha: la energia esta dominada por el largo de la respuesta (decode).
      - alpha significativo: hay un coste de prefill detectable — el contexto importa.
      - R2 cercano a 1: el modelo lineal explica bien la varianza de energia.
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
# Tabla de recuperacion naive vs compaction
# ---------------------------------------------------------------------------

def recovery_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for quant in sorted(df["quant"].unique()):
        for arm in ["naive", "compaction"]:
            sub = df[(df["quant"] == quant) & (df["arm"] == arm)]
            if sub.empty:
                continue
            tasks = sub[~sub["is_compaction"]]
            comp = sub[sub["is_compaction"]]
            per_rep = sub.groupby("rep")["energy_j"].sum()
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
    ap.add_argument("--out", default="results/analysis")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.results)
    df.to_csv(out / "all_runs_long.csv", index=False)
    print(f"[OK] {len(df)} registros cargados de {df['run_id'].nunique()} corridas.")

    # ------------------------------------------------------------------
    # 1. Curvas y codo por cuantizacion
    # ------------------------------------------------------------------
    print("\n=== Deteccion del CODO (context-rot) ===")
    knees = {}
    for quant in sorted(df["quant"].unique()):
        knee = detect_knee(df, quant)
        knees[quant] = knee
        if knee:
            print(f"  [{knee['detector'].upper()}] {quant}: ~{knee['accumulated_tokens']} tokens "
                  f"| task={knee.get('task_id','?')} | quality={knee['quality']:.4f} "
                  f"| {knee.get('note','')}")
        else:
            print(f"  {quant}: codo no detectado en el rango actual de contexto.")
        plot_envelope(df, quant, out / f"envelope_{quant}.png", knee)

    with open(out / "knees.json", "w", encoding="utf-8") as f:
        json.dump(knees, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 2. Desglose de calidad por tarea (hallazgo central)
    # ------------------------------------------------------------------
    print("\n=== Desglose calidad: SONDA + RECALL x quant x arm ===")
    qb = quality_breakdown(df)
    qb.to_csv(out / "quality_by_task.csv", index=False)

    ss = sonda_recall_summary(df)
    if not ss.empty:
        ss.to_csv(out / "sonda_recall_summary.csv", index=False)
        print(ss.to_string(index=False))
    else:
        print("  (sin tareas SONDA ni RECALL en los datos — revisa el tipo de tarea en session_tasks.json)")

    # ------------------------------------------------------------------
    # 3. Regresion prefill vs decode
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
    # 4. Tabla de recuperacion naive vs compaction
    # ------------------------------------------------------------------
    rec = recovery_table(df)
    rec.to_csv(out / "recovery_naive_vs_compaction.csv", index=False)
    print("\n=== Recuperacion naive vs compaction ===")
    print(rec.to_string(index=False))

    print(f"\n[OK] Figuras y tablas en {out}/")
    print("  Archivos generados:")
    for f in sorted(out.iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
