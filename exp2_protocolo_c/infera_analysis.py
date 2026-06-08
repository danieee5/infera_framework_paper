"""
infera_analysis.py
Analisis del Protocolo C: sobre conjunto energia x calidad + recuperacion por compactacion.

Lee todos los .jsonl de results/ y produce:
  1. Curvas energia/tarea y calidad vs contexto acumulado (por cuantizacion, brazo naive).
  2. Deteccion del CODO: donde la calidad cae mientras la energia/token sube.
  3. Figura insignia: doble eje (J/tarea y calidad) sobre contexto acumulado.
  4. Recuperacion: naive vs compaction (energia acumulada, calidad post-handoff,
     impuesto de compactacion, break-even).

Uso:
  python infera_analysis.py --results results --out results/analysis
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def detect_knee(curve: pd.DataFrame, quality_drop_frac: float = 0.85):
    """
    Heuristica transparente del codo sobre el brazo naive:
      - calidad_base = media de la calidad en el primer tercio de la sesion.
      - codo = primera tarea donde la calidad cae por debajo de quality_drop_frac*base
               Y la energia por token de salida es >= la mediana.
    Devuelve dict con task_index, accumulated_tokens, quality, o None.
    """
    c = curve.sort_values("task_index")
    c = c[c["quality"].notna()]
    if len(c) < 3:
        return None
    n_third = max(1, len(c) // 3)
    base_q = c["quality"].iloc[:n_third].mean()
    med_jpt = c["j_per_completion_token"].median()
    for _, row in c.iterrows():
        if (row["quality"] < quality_drop_frac * base_q
                and (row["j_per_completion_token"] or 0) >= (med_jpt or 0)):
            return {"task_index": row["task_index"],
                    "accumulated_tokens": int(row["accumulated_prompt_tokens"]),
                    "quality": float(row["quality"]),
                    "base_quality": round(float(base_q), 3)}
    return None


def plot_envelope(df: pd.DataFrame, quant: str, out_path: Path):
    """Figura insignia: doble eje J/tarea y calidad vs contexto acumulado (brazo naive)."""
    sub = df[(df["quant"] == quant) & (df["arm"] == "naive") & (~df["is_compaction"])]
    if sub.empty:
        return None
    # promediar repeticiones por task_index
    agg = sub.groupby("task_index").agg(
        ctx=("accumulated_prompt_tokens", "mean"),
        energy=("energy_j", "mean"),
        quality=("quality", "mean"),
    ).reset_index().sort_values("ctx")

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.set_xlabel("Contexto acumulado (tokens de prompt)")
    ax1.set_ylabel("Energia por tarea (J)", color="tab:red")
    ax1.plot(agg["ctx"], agg["energy"], "o-", color="tab:red", label="Energia/tarea (J)")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Calidad de tarea (0-1)", color="tab:blue")
    ax2.plot(agg["ctx"], agg["quality"], "s--", color="tab:blue", label="Calidad")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    knee = detect_knee(sub.groupby("task_index").agg(
        accumulated_prompt_tokens=("accumulated_prompt_tokens", "mean"),
        energy_j=("energy_j", "mean"),
        quality=("quality", "mean"),
        j_per_completion_token=("j_per_completion_token", "mean"),
    ).reset_index())
    if knee:
        ax1.axvline(knee["accumulated_tokens"], color="gray", linestyle=":", linewidth=1.5)
        ax1.annotate(f"CODO ~{knee['accumulated_tokens']} tok\n(pagas mas, calidad cae)",
                     xy=(knee["accumulated_tokens"], ax1.get_ylim()[1] * 0.9),
                     fontsize=9, color="gray")

    plt.title(f"Sobre conjunto Energia x Calidad vs contexto acumulado — {quant} (naive)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return knee


def recovery_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compara energia acumulada y calidad media naive vs compaction por cuantizacion."""
    rows = []
    for quant in sorted(df["quant"].unique()):
        for arm in ["naive", "compaction"]:
            sub = df[(df["quant"] == quant) & (df["arm"] == arm)]
            if sub.empty:
                continue
            tasks = sub[~sub["is_compaction"]]
            comp = sub[sub["is_compaction"]]
            # energia total promedio por repeticion
            per_rep = sub.groupby("rep")["energy_j"].sum()
            tax_per_rep = comp.groupby("rep")["energy_j"].sum() if not comp.empty else None
            rows.append({
                "quant": quant, "arm": arm,
                "energia_total_J_media": round(per_rep.mean(), 2),
                "calidad_media": round(tasks["quality"].mean(), 3),
                "impuesto_compactacion_J": round(tax_per_rep.mean(), 2) if tax_per_rep is not None else 0.0,
                "n_reps": sub["rep"].nunique(),
            })
    return pd.DataFrame(rows)


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

    # Figura insignia por cuantizacion (brazo naive)
    for quant in sorted(df["quant"].unique()):
        knee = plot_envelope(df, quant, out / f"envelope_{quant}.png")
        if knee:
            print(f"[CODO] {quant}: ~{knee['accumulated_tokens']} tokens "
                  f"(calidad {knee['quality']:.2f} vs base {knee['base_quality']:.2f})")
        else:
            print(f"[CODO] {quant}: no detectado con el umbral actual "
                  f"(quiza la sesion es muy corta; extiende el numero de tareas).")

    # Tabla de recuperacion naive vs compaction
    rec = recovery_table(df)
    rec.to_csv(out / "recovery_naive_vs_compaction.csv", index=False)
    print("\n=== Recuperacion naive vs compaction ===")
    print(rec.to_string(index=False))
    print(f"\n[OK] Figuras y tablas en {out}/")


if __name__ == "__main__":
    main()
