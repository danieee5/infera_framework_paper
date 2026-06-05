"""
build_multiturn_conversation.py  —  INFERA Experimento 2, Fase 1.

Genera la HISTORIA CONVERSACIONAL FIJA (pinned) que usaran los tres esquemas
(FP16 / INT8 / AWQ) en multiturn_runner.py, y valida los tokens reales contra
el tokenizer del propio modelo via el `usage` de vLLM.

POR QUE FASE 1 SEPARADA:
  Con temperature=0 (greedy decoding) la conversacion se pre-genera UNA sola vez
  con FP16 y se congela. Asi el contexto presentado en cada turno es IDENTICO
  para los tres esquemas, y las diferencias de energia en la Fase 2 son
  atribuibles al esquema de cuantizacion, no a respuestas divergentes.

  >>> Esta historia debe generarse UNA vez y copiarse IDENTICA a todos los pods. <<<

QUE PRODUCE:
  - data/multiturn/conversation_history.json   (lo que consume multiturn_runner.py)
  - actualiza data/multiturn/conversation_flow.json con measured_input_tokens reales
  - reporte de validacion: T1 en [1500,2000], T7 en [3500,4500], max+256 < max-model-len

COMPOSICION DEL CONTEXTO:
  system enviado al modelo = system_prompt + "\\n\\n=== DOCUMENTOS ===\\n" + combined_content
  Se reenvia COMPLETO cada turno (prefix caching OFF => re-prefill completo).

USO (con el servidor vLLM FP16 corriendo en otra terminal):
  bash scripts/start_vllm_fp16.sh          # en otra terminal
  python scripts/build_multiturn_conversation.py
  python scripts/build_multiturn_conversation.py --dry-run   # solo medir, sin regenerar
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("FATAL: httpx no instalado. Ejecuta: pip install httpx")

VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_TOKENS  = 256
TEMPERATURE = 0.0
TIMEOUT_S   = 300
MAX_MODEL_LEN = 8192

FLOW_PATH    = Path("data/multiturn/conversation_flow.json")
HISTORY_PATH = Path("data/multiturn/conversation_history.json")

# Bandas objetivo (tokens REALES de entrada por turno)
T1_BAND = (1500, 2000)
T7_BAND = (3500, 4500)


def compose_system(flow: dict) -> str:
    """system efectivo = system_prompt + bloque de documentos."""
    base = flow["system_prompt"].strip()
    docs = flow["documents"]["combined_content"].strip()
    return f"{base}\n\n=== DOCUMENTOS INTERNOS MOSS ===\n{docs}"


def build_messages(system: str, prior_turns: list[dict], user_content: str) -> list[dict]:
    """system + pares (user, assistant) previos + user actual (sin respuesta)."""
    msgs = [{"role": "system", "content": system}]
    for t in prior_turns:
        msgs.append({"role": "user",      "content": t["user_content"]})
        msgs.append({"role": "assistant", "content": t["assistant_content"]})
    msgs.append({"role": "user", "content": user_content})
    return msgs


def call_vllm(client: httpx.Client, messages: list[dict], generate: bool) -> dict:
    """
    Una peticion no-streaming. Si generate=False, pide max_tokens=1 solo para
    leer usage.prompt_tokens (medir contexto sin gastar generacion).
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS if generate else 1,
        "temperature": TEMPERATURE,
        "stream":      False,
    }
    r = client.post(VLLM_URL, json=payload, timeout=TIMEOUT_S)
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage", {})
    text  = data["choices"][0]["message"]["content"] if generate else ""
    return {
        "text":              text.strip(),
        "prompt_tokens":     usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def main():
    ap = argparse.ArgumentParser(description="INFERA Exp.2 Fase 1 — build + validate")
    ap.add_argument("--flow",    default=str(FLOW_PATH))
    ap.add_argument("--history", default=str(HISTORY_PATH))
    ap.add_argument("--dry-run", action="store_true",
                    help="Solo medir tokens (usa historia existente si la hay; no regenera).")
    args = ap.parse_args()

    flow_path    = Path(args.flow)
    history_path = Path(args.history)
    if not flow_path.exists():
        sys.exit(f"FATAL: no existe {flow_path}")

    flow   = json.loads(flow_path.read_text(encoding="utf-8"))
    system = compose_system(flow)
    user_turns = [t["content"] for t in flow["turns"]]

    # En dry-run con historia previa, reusar respuestas fijas para medir contexto exacto.
    existing = None
    if args.dry_run and history_path.exists():
        existing = json.loads(history_path.read_text(encoding="utf-8"))

    print("\n" + "=" * 66)
    print("  INFERA Exp.2 — Fase 1: construccion + validacion de contexto")
    print(f"  Modo: {'DRY-RUN (solo medir)' if args.dry_run else 'GENERAR historia con FP16'}")
    print("=" * 66 + "\n")

    history_turns = []
    measured = {}

    with httpx.Client() as client:
        # sanity
        try:
            client.get("http://localhost:8000/v1/models", timeout=10).raise_for_status()
        except Exception as e:
            sys.exit(f"FATAL: vLLM no responde en localhost:8000 ({e}). "
                     f"Levanta start_vllm_fp16.sh primero.")

        prior = []
        for i, uc in enumerate(user_turns, 1):
            msgs = build_messages(system, prior, uc)

            if args.dry_run and existing:
                # medir contexto real sin generar; respuesta fija de la historia
                res = call_vllm(client, msgs, generate=False)
                assistant = existing["turns"][i - 1]["assistant_content"]
            else:
                res = call_vllm(client, msgs, generate=True)
                assistant = res["text"]

            pt = res["prompt_tokens"]
            measured[i] = pt
            print(f"  T{i}: input_real = {pt:>5} tok"
                  + (f" | output = {res['completion_tokens']} tok" if not args.dry_run else ""))

            history_turns.append({
                "turn_number": i,
                "user_content": uc,
                "assistant_content": assistant,
                "measured_input_tokens": pt,
                "measured_output_tokens": res["completion_tokens"] if not args.dry_run else None,
            })
            prior.append({"user_content": uc, "assistant_content": assistant})

    # ── Escribir historia (solo si generamos) ──────────────────────────────
    if not args.dry_run:
        history = {
            "system_prompt": system,          # incluye documentos -> re-prefill completo cada turno
            "generated_with": "fp16 temperature=0 greedy (pinned)",
            "turns": history_turns,
        }
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n  -> historia fija escrita: {history_path}")

    # actualizar flow con tokens reales
    for t in flow["turns"]:
        t["measured_input_tokens"] = measured.get(t["turn_number"])
    flow_path.write_text(json.dumps(flow, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  -> conversation_flow.json actualizado con tokens reales")

    # ── Reporte de validacion ──────────────────────────────────────────────
    t1, t7 = measured.get(1), measured.get(7)
    max_ctx = max(measured.values())
    print("\n" + "-" * 66)
    print("  VALIDACION DE LA ENVOLVENTE DE TOKENS")
    print("-" * 66)

    def check(label, val, lo, hi):
        ok = lo <= val <= hi
        flag = "OK " if ok else "FUERA"
        print(f"  {label}: {val:>5} tok  (banda {lo}-{hi})  [{flag}]")
        return ok

    ok1 = check("T1 (~Case_B)", t1, *T1_BAND)
    ok7 = check("T7 (~Case_C)", t7, *T7_BAND)
    okm = (max_ctx + MAX_TOKENS) < MAX_MODEL_LEN
    print(f"  Max contexto T7 + {MAX_TOKENS} salida = {max_ctx + MAX_TOKENS} tok  "
          f"(limite max-model-len {MAX_MODEL_LEN})  [{'OK' if okm else 'EXCEDE'}]")

    if not (ok1 and ok7 and okm):
        print("\n  >>> AJUSTE SUGERIDO <<<")
        if not ok1:
            d = (sum(T1_BAND)//2) - t1
            print(f"    T1 fuera de banda: {'agrega' if d>0 else 'quita'} ~{abs(d)} tokens "
                  f"al bloque documents.combined_content (afecta la BASE de todos los turnos).")
        if not ok7 and ok1:
            print(f"    Solo T7 fuera: el largo de las respuestas A1..A6 quedo "
                  f"{'corto' if t7<T7_BAND[0] else 'largo'}. Ajusta max_tokens_per_turn o el "
                  f"detalle pedido en los turnos generativos (T2,T3,T6).")
        if not okm:
            print(f"    Reduce el corpus o sube --max-model-len en los start_vllm_*.sh.")
        print("    Re-ejecuta esta Fase 1 tras ajustar.")
    else:
        print("\n  Envolvente VALIDADA: puente Case_B -> Case_C correcto. Lista para Fase 2.")
    print("-" * 66 + "\n")


if __name__ == "__main__":
    main()
