"""
infera_session_runner.py
Runner principal de una sesión incremental INFERA.

Corre UNA sesion incremental por configuracion:
  (cuantizacion x brazo x repeticion)

Brazos:
  --arm naive        : el contexto crece sin compactar (baseline).
  --arm compaction   : al cruzar el umbral de tokens, compacta y continua.

Por cada tarea registra: energia (J, NVML), tokens (prompt/completion via usage),
calidad (infera_quality), y marca eventos de compactacion con su costo energetico
(el "impuesto de compactacion").

Requisitos de medición:
  - GPUPowerMonitor (NVML 500ms buffer / 100ms / trapezoidal).
  - vLLM sirviendo /v1/chat/completions (plantilla de chat LLaMA 3.1 interna).

Uso tipico:
  python infera_session_runner.py \
      --vllm-url http://localhost:8000/v1/chat/completions \
      --model /models/llama3.1-8b-instruct \
      --quant FP16 --arm naive --rep 1 \
      --kb-dir kb --tasks config/session_tasks.example.json \
      --out results/runs/mi_corrida/run_mi_corrida_FP16_naive_rep1.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from gpu_power_monitor import GPUPowerMonitor
from infera_kb import build_fixed_context
from infera_quality import score_task
from infera_compaction import build_compaction_messages, apply_handoff


def call_vllm(url, model, messages, temperature, max_tokens, seed, timeout=600):
    """Llama a vLLM (OpenAI-compatible). Devuelve (text, prompt_tokens, completion_tokens, ok, err)."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        return (text,
                int(usage.get("prompt_tokens", 0)),
                int(usage.get("completion_tokens", 0)),
                True, "")
    except Exception as e:
        return ("", 0, 0, False, str(e))


def measured_call(monitor, url, model, messages, temperature, max_tokens, seed):
    """Envuelve una llamada vLLM con medicion de energia NVML."""
    monitor.start_monitoring()              # incluye buffer pre 500ms
    t0 = time.time()
    text, ptok, ctok, ok, err = call_vllm(url, model, messages, temperature, max_tokens, seed)
    wall = time.time() - t0
    energy = monitor.stop_monitoring()      # incluye buffer post 500ms
    return {
        "text": text, "prompt_tokens": ptok, "completion_tokens": ctok,
        "ok": ok, "err": err, "wall_s": wall,
        "energy_j": energy.energy_j, "duration_s": energy.duration_s,
        "avg_power_w": energy.avg_power_w, "peak_power_w": energy.peak_power_w,
        "vram_peak_mb": energy.vram_used_mb_peak, "nvml_samples": energy.sample_count,
        "nvml_available": energy.nvml_available,
    }


def run_session(args):
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise SystemExit(
            f"La salida ya existe y no se sobrescribirá: {out}. "
            "Usa otra etiqueta o añade --overwrite de forma explícita."
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    kb_system = build_fixed_context(args.kb_dir)
    tasks_doc = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    dec = tasks_doc.get("decoding", {"temperature": 0.0, "max_tokens": 512, "seed": 42})
    temperature = dec.get("temperature", 0.0)
    max_tokens = dec.get("max_tokens", 512)
    seed = dec.get("seed", 42)
    tasks = tasks_doc["tasks"]

    monitor = GPUPowerMonitor(device_index=args.device_index)
    if not monitor.available:
        raise SystemExit(
            "NVML no está disponible. Se cancela la medición para evitar "
            "generar resultados energéticos inválidos."
        )

    # --- Calentamiento: descartar N peticiones antes de medir ---
    for _ in range(args.warmup):
        call_vllm(args.vllm_url, args.model,
                  [{"role": "system", "content": kb_system},
                   {"role": "user", "content": "Responde OK."}],
                  temperature, 8, seed)

    history = []          # turnos SIN el system KB (se antepone en cada llamada)
    records = []
    cumulative_energy = 0.0
    compaction_count = 0

    for idx, task in enumerate(tasks):
        messages = [{"role": "system", "content": kb_system}] + history + \
                   [{"role": "user", "content": task["prompt"]}]

        res = measured_call(monitor, args.vllm_url, args.model, messages,
                            temperature, max_tokens, seed)

        status = "ok" if res["ok"] else "error"
        q = score_task(res["text"], task.get("verify", {}), judge_fn=None) \
            if res["ok"] else {"quality": 0.0, "subscores": {}, "forbidden_penalty": 1.0}

        cumulative_energy += res["energy_j"]
        j_per_tok = (res["energy_j"] / res["completion_tokens"]) if res["completion_tokens"] else None

        rec = {
            "run_id": f"{args.quant}_{args.arm}_rep{args.rep}",
            "quant": args.quant, "arm": args.arm, "rep": args.rep,
            "task_index": idx, "task_id": task["id"], "task_type": task["type"],
            "accumulated_prompt_tokens": res["prompt_tokens"],
            "completion_tokens": res["completion_tokens"],
            "energy_j": round(res["energy_j"], 4),
            "j_per_completion_token": round(j_per_tok, 5) if j_per_tok else None,
            "duration_s": round(res["duration_s"], 4),
            "avg_power_w": round(res["avg_power_w"], 2),
            "peak_power_w": round(res["peak_power_w"], 2),
            "vram_peak_mb": round(res["vram_peak_mb"], 1),
            "quality": q["quality"],
            "quality_subscores": q["subscores"],
            "prompt_text": task["prompt"],
            "response_text": res["text"],
            "cumulative_energy_j": round(cumulative_energy, 4),
            "is_compaction": False,
            "compaction_index": compaction_count,
            "status": status, "error": res["err"],
            "nvml_samples": res["nvml_samples"], "nvml_available": res["nvml_available"],
        }
        records.append(rec)
        print(f"  [{task['id']:>4} {task['type']:<10}] "
              f"ctx={res['prompt_tokens']:>5}tok  E={res['energy_j']:>7.2f}J  "
              f"Q={q['quality']:.2f}  ({status})")

        # actualizar historial
        history.append({"role": "user", "content": task["prompt"]})
        history.append({"role": "assistant", "content": res["text"]})

        # --- Brazo de compactacion: si cruza el umbral, compactar ---
        if (args.arm == "compaction"
                and res["prompt_tokens"] >= args.compaction_threshold
                and idx < len(tasks) - 1):
            comp_msgs = build_compaction_messages(kb_system, history)
            comp = measured_call(monitor, args.vllm_url, args.model, comp_msgs,
                                temperature, max_tokens, seed)
            cumulative_energy += comp["energy_j"]
            compaction_count += 1
            records.append({
                "run_id": f"{args.quant}_{args.arm}_rep{args.rep}",
                "quant": args.quant, "arm": args.arm, "rep": args.rep,
                "task_index": idx + 0.5, "task_id": f"COMPACT_{compaction_count}",
                "task_type": "COMPACTION",
                "accumulated_prompt_tokens": comp["prompt_tokens"],
                "completion_tokens": comp["completion_tokens"],
                "energy_j": round(comp["energy_j"], 4),
                "j_per_completion_token": None,
                "duration_s": round(comp["duration_s"], 4),
                "avg_power_w": round(comp["avg_power_w"], 2),
                "peak_power_w": round(comp["peak_power_w"], 2),
                "vram_peak_mb": round(comp["vram_peak_mb"], 1),
                "quality": None,
                "quality_subscores": {},
                "prompt_text": "[COMPACTION_INSTRUCTION] resumir sesion en handoff estructurado",
                "response_text": comp["text"],
                "cumulative_energy_j": round(cumulative_energy, 4),
                "is_compaction": True,
                "compaction_index": compaction_count,
                "status": "ok" if comp["ok"] else "error", "error": comp["err"],
                "nvml_samples": comp["nvml_samples"], "nvml_available": comp["nvml_available"],
            })
            print(f"  [COMPACT #{compaction_count}] tax E={comp['energy_j']:.2f}J "
                  f"(ctx era {comp['prompt_tokens']}tok) -> reinicio de contexto")
            # reiniciar contexto a KB + handoff
            history = apply_handoff(kb_system, comp["text"])[1:]  # quitar el system (se reañade en cada llamada)

    monitor.cleanup()

    with open(out, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[OK] {len(records)} registros -> {out}")
    print(f"[OK] Energia total de la sesion: {cumulative_energy:.2f} J | compactaciones: {compaction_count}")


def main():
    ap = argparse.ArgumentParser(
        description="INFERA — runner de una sesión incremental"
    )
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", required=True, help="ruta o id del modelo servido por vLLM")
    ap.add_argument("--quant", required=True, help="etiqueta de cuantizacion (FP16/INT8/AWQ)")
    ap.add_argument("--arm", choices=["naive", "compaction"], default="naive")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--kb-dir", default="kb")
    ap.add_argument("--tasks", default="config/session_tasks.example.json")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--compaction-threshold",
        type=int,
        default=4500,
        help=(
            "tokens del prompt medido que disparan compactación; es una "
            "regla operativa de la política, no un umbral óptimo"
        ),
    )
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="permite sobrescribir la salida indicada; desactivado por defecto",
    )
    args = ap.parse_args()

    try:
        import requests as requests_module
    except ImportError as exc:
        raise SystemExit(
            "Falta requests. Instala requirements-gpu.txt antes de ejecutar "
            "una medición."
        ) from exc
    globals()["requests"] = requests_module

    print(f"=== INFERA | {args.quant} | {args.arm} | rep {args.rep} ===")
    run_session(args)


if __name__ == "__main__":
    main()
