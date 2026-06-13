"""
multiturn_runner.py
Phase 2 of the multi-turn experiment — Experimento 2 of INFERA.

Measures energy consumption per conversational turn across quantization
schemes (FP16, INT8 W8A16, INT4 AWQ) and batch sizes (1, 4).

DESIGN:
  - 7-turn enterprise conversation (MOSS security company internal assistant)
  - Fixed conversation history pre-generated in Phase 1 (temperature=0)
  - Each turn measured independently with MELODI 500ms buffer protocol
  - 120s cooling between turns to ensure thermal stability
  - Context grows organically: ~1.9k tokens (T1) → ~3.7k tokens (T7)
    (puente Case_B → Case_C del factorial estatico)

MEASUREMENT SCOPE (declaracion explicita — defendible ante jurado):
  - EXP2 mide ENERGIA TOTAL por turno: prefill + decode en UNA sola ventana NVML.
    No se segmenta energia prefill vs decode.
  - TTFT (time-to-first-token) se reporta como PROXY TEMPORAL del prefill.
    TPOT (time-per-output-token) se reporta como PROXY TEMPORAL del decode.
  - Regimen medido: RE-PREFILL COMPLETO por turno. Los servidores vLLM se
    levantan SIN --enable-prefix-caching, por lo que cada turno reprocesa todo
    el contexto acumulado desde cero. Este es el regimen intencional: aisla el
    costo energetico del contexto creciente.
  - Prefix caching (la optimizacion que usaria produccion real) queda FUERA DE
    ALCANCE y se declara como amenaza a la validez externa / trabajo futuro.
  - max_tokens=256 fijo: control para mantener el decode ~constante, de modo que
    el crecimiento de energia por turno sea atribuible al prefill creciente.

EXPERIMENTAL MATRIX:
  VI1 (quantization): fp16, int8_w8a16, int4_awq
  VI2 (batch_size):   1, 4
  Turn:               1–7
  Repetitions:        3
  Total measurements: 3 × 2 × 7 × 3 = 126 turn-level records

HYPOTHESES UNDER TEST:
  H1: J/output_token increases monotonically with turn number (KV-cache pressure)
  H2: INT8 batch=4 anomaly (J/tok_batch4 > J/tok_batch1) amplifies in later turns
  H3: AWQ energy advantage over FP16 erodes as context grows (consistent with
      +122% vs +59% asymmetry found in static factorial, Hallazgo 3)
  H4: VRAM peak grows monotonically with turn, confirming KV-cache accumulation

USAGE:
  # vLLM must be running with appropriate quantization:
  #   bash scripts/start_vllm_fp16.sh     → then run with --quantization fp16
  #   bash scripts/start_vllm_int8.sh     → then run with --quantization int8_w8a16
  #   bash scripts/start_vllm_awq.sh      → then run with --quantization int4_awq
  python scripts/multiturn_runner.py --quantization fp16
  python scripts/multiturn_runner.py --quantization fp16 --batch-sizes 1      # single batch
  python scripts/multiturn_runner.py --quantization fp16 --pilot               # rep=1 only
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    sys.exit("FATAL: httpx not installed. Run: pip install httpx")

# Import existing MELODI-validated monitor
sys.path.insert(0, str(Path(__file__).parent))
from gpu_power_monitor import GPUPowerMonitor

# ── CONFIG ────────────────────────────────────────────────────────────────────
VLLM_URL   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_TOKENS  = 256
TEMPERATURE = 0.0
TIMEOUT_S   = 300

BATCH_SIZES  = [1, 4]   # Reduced from static experiment's [1,4,8]
REPETITIONS  = 3
N_TURNS      = 7
COOLING_S    = 120       # Between turns — same as static factorial
IDLE_WAIT_S  = 5         # Pre-measurement idle (same as static)
WARMUP_TURNS = 2         # Warmup turns before first measured session

HISTORY_PATH  = Path("data/multiturn/conversation_history.json")
RESULTS_DIR   = Path("results/multiturn")


# ── MESSAGE BUILDER ───────────────────────────────────────────────────────────

def build_messages_for_turn(history: dict, turn_number: int) -> list[dict]:
    """
    Build the vLLM messages array for a given turn.
    Includes system prompt + all prior turns + current user message.

    The conversation history is fixed (pre-generated with FP16 temp=0),
    so the same context is presented to all quantization schemes.
    This ensures energy differences are attributable to the scheme,
    not to different response content.
    """
    messages = [
        {"role": "system", "content": history["system_prompt"]}
    ]

    turns = history["turns"]

    for i, turn in enumerate(turns[:turn_number]):
        messages.append({"role": "user",      "content": turn["user_content"]})
        # For turns before the current one: add the pre-generated response
        if i < turn_number - 1:
            messages.append({"role": "assistant", "content": turn["assistant_content"]})
        # For the current turn (i == turn_number - 1): do NOT add the response
        # — this is what we're asking the model to generate now

    return messages


# ── STREAMING REQUEST (with TTFT) ─────────────────────────────────────────────

async def single_request_streaming(
    client: httpx.AsyncClient,
    messages: list[dict],
) -> dict:
    """
    Send one request with stream=True to capture TTFT precisely.
    Returns: status, prompt_tokens, completion_tokens, ttft_ms, total_time_s, text
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream":      True,
        "stream_options": {"include_usage": True},  # vLLM 0.5.3+ returns usage in last chunk
    }

    t_start   = time.perf_counter()
    ttft_ms   = None
    chunks    = []
    pt, ct    = 0, 0

    try:
        async with client.stream("POST", VLLM_URL, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return {"status": "error", "error": f"HTTP {resp.status_code}: {body[:200]}"}

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                delta   = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")

                if content and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t_start) * 1000

                if content:
                    chunks.append(content)

                # Usage is in the final chunk (stream_options)
                if "usage" in chunk and chunk["usage"]:
                    pt = chunk["usage"].get("prompt_tokens", 0)
                    ct = chunk["usage"].get("completion_tokens", 0)

    except httpx.TimeoutException:
        return {"status": "timeout", "error": f"Exceeded {TIMEOUT_S}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}

    t_end = time.perf_counter()
    full_text    = "".join(chunks)
    total_time_s = t_end - t_start
    ct_actual    = ct if ct > 0 else max(1, len(full_text.split()) * 4 // 3)

    return {
        "status":         "success",
        "prompt_tokens":  pt,
        "completion_tokens": ct_actual,
        "ttft_ms":        round(ttft_ms, 2) if ttft_ms else None,
        "total_time_s":   round(total_time_s, 4),
        "text":           full_text,
    }


# ── CONCURRENT BATCH CALL (batch_size requests in parallel) ──────────────────

def call_concurrent(
    messages_list: list[list[dict]],
) -> dict:
    """
    Launch batch_size identical requests simultaneously via asyncio.gather().
    All requests receive the SAME messages (same conversation state at this turn).
    Returns aggregated results.
    """
    async def _run():
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            tasks = [
                single_request_streaming(client, msgs)
                for msgs in messages_list
            ]
            return await asyncio.gather(*tasks)

    results = asyncio.run(_run())

    successes = [r for r in results if r["status"] == "success"]
    if not successes:
        return {
            "status": results[0].get("status", "error"),
            "error":  results[0].get("error", "all requests failed"),
        }

    n = len(successes)
    return {
        "status":            "success",
        "prompt_tokens":     sum(r["prompt_tokens"]      for r in successes),
        "completion_tokens": sum(r["completion_tokens"]  for r in successes),
        "ttft_ms_mean":      sum(r["ttft_ms"] or 0       for r in successes) / n,
        "total_time_s":      max(r["total_time_s"]       for r in successes),
        "n_success":         n,
        "n_batch":           len(results),
        # Texto representativo (primera respuesta exitosa). Todas las peticiones
        # del batch reciben el MISMO contexto, asi que basta una para inspeccion.
        "text":              successes[0].get("text", ""),
    }


# ── WARMUP ────────────────────────────────────────────────────────────────────

def run_warmup(history: dict, batch_size: int):
    """
    Run WARMUP_TURNS turns without NVML monitoring to warm up GPU and KV-cache.
    Uses turn 1 messages repeated.
    """
    msgs  = build_messages_for_turn(history, 1)
    batch = [msgs] * batch_size
    print(f"    [Warmup: {WARMUP_TURNS} turns]", end=" ", flush=True)
    for _ in range(WARMUP_TURNS):
        call_concurrent(batch)
        print(".", end="", flush=True)
    print()


# ── SINGLE TURN MEASUREMENT ───────────────────────────────────────────────────

def measure_turn(
    history:    dict,
    turn_number: int,
    batch_size: int,
    monitor:    GPUPowerMonitor,
) -> dict:
    """
    Measure energy for a single turn with MELODI 500ms buffer protocol.
    Returns the full result dict for this turn measurement.
    """
    messages      = build_messages_for_turn(history, turn_number)
    messages_list = [messages] * batch_size  # identical for all concurrent requests

    time.sleep(IDLE_WAIT_S)

    # MELODI PROTOCOL: 500ms pre-buffer → inference → 500ms post-buffer
    monitor.start_monitoring()
    vllm_result = call_concurrent(messages_list)
    energy = monitor.stop_monitoring()

    return {
        "vllm_result": vllm_result,
        "energy":      energy,
        "input_tokens_per_request": messages_len_approx(messages),
    }


def messages_len_approx(messages: list[dict]) -> int:
    """Rough token count: total chars / 4."""
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4


# ── RESULT BUILDER ────────────────────────────────────────────────────────────

def build_result_record(
    quantization: str,
    batch_size:   int,
    repetition:   int,
    turn_number:  int,
    history:      dict,
    meas:         dict,
) -> dict:
    """Construct the standardized result record for this turn measurement."""

    vr  = meas["vllm_result"]
    en  = meas["energy"]
    run_id = (
        f"mt_{quantization}_b{batch_size}_t{turn_number}"
        f"_rep{repetition}_{int(time.time())}"
    )

    record = {
        "run_id":         run_id,
        "experiment":     "multiturn",
        "timestamp":      datetime.now(timezone.utc).isoformat(),

        # Independent variables
        "vi1_quantization": quantization,
        "vi2_batch_size":   batch_size,
        "turn_number":      turn_number,
        "repetition":       repetition,

        # Context metadata
        "input_tokens_approx":      meas["input_tokens_per_request"],
        "cumulative_turns_in_context": turn_number - 1,

        # Status
        "status": vr.get("status", "unknown"),
        "error":  vr.get("error"),

        # Energy (NVML MELODI protocol)
        "energy_j":           None,
        "duration_s":         None,
        "avg_power_w":        None,
        "peak_power_w":       None,
        "vram_peak_mb":       None,
        "vram_start_mb":      None,
        "nvml_samples":       en.sample_count,
        "nvml_available":     en.nvml_available,

        # Throughput & latency
        "prompt_tokens":      None,
        "completion_tokens":  None,
        "ttft_ms":            None,
        "tpot_ms":            None,
        "throughput_tok_s":   None,
        "j_per_output_token": None,

        # Batch metadata
        "n_batch_success":    vr.get("n_success"),
        "n_batch_total":      vr.get("n_batch"),

        # Generated text (representative; for qualitative inspection only — no
        # formal quality scoring is performed in EXP2).
        "generated_text":     None,
    }

    if vr.get("status") == "success":
        ct  = vr.get("completion_tokens", 0)
        pt  = vr.get("prompt_tokens", 0)
        dur = vr.get("total_time_s", 0)
        ttft = vr.get("ttft_ms_mean")

        tput = ct / dur if dur > 0 else 0.0
        jpt  = en.energy_j / ct if ct > 0 else 0.0
        tpot = (dur / ct * 1000) if ct > 0 else None

        record.update({
            "energy_j":           round(en.energy_j, 6),
            "duration_s":         round(dur, 4),
            "avg_power_w":        round(en.avg_power_w, 2),
            "peak_power_w":       round(en.peak_power_w, 2),
            "vram_peak_mb":       round(en.vram_used_mb_peak, 1),
            "vram_start_mb":      round(en.vram_used_mb_start, 1),
            "prompt_tokens":      pt,
            "completion_tokens":  ct,
            "ttft_ms":            round(ttft, 2) if ttft else None,
            "tpot_ms":            round(tpot, 2) if tpot else None,
            "throughput_tok_s":   round(tput, 4),
            "j_per_output_token": round(jpt, 6),
            "generated_text":     vr.get("text", ""),
        })

    return record


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run_multiturn(
    quantization: str,
    batch_sizes:  list[int],
    pilot:        bool,
    results_dir:  Optional[str],
):
    if not HISTORY_PATH.exists():
        sys.exit(
            f"\nFATAL: Conversation history not found: {HISTORY_PATH}\n"
            f"Run Phase 1 first: python scripts/build_multiturn_conversation.py\n"
        )

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    turns   = list(range(1, N_TURNS + 1))
    reps    = [1] if pilot else list(range(1, REPETITIONS + 1))

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_dir or f"{RESULTS_DIR}/{quantization}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_measurements = len(batch_sizes) * len(turns) * len(reps)

    print(f"\n{'='*65}")
    print(f"  MULTI-TURN BENCHMARK  |  quantization={quantization}")
    print(f"  Batch sizes: {batch_sizes}  |  Turns: {N_TURNS}  |  Reps: {reps}")
    print(f"  Total turn-measurements: {total_measurements}")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")

    monitor = GPUPowerMonitor(device_index=0)
    all_results = []
    measurement_idx = 0

    try:
        for batch_size in batch_sizes:

            print(f"\n{'─'*55}")
            print(f"  Batch size = {batch_size}")

            # Warmup: run 2 turns without measurement to stabilize GPU
            print(f"  Warming up...")
            run_warmup(history, batch_size)

            for rep in reps:
                print(f"\n  Rep {rep}/{len(reps)}:")

                for turn_number in turns:
                    measurement_idx += 1
                    progress = f"[{measurement_idx:>3}/{total_measurements}]"

                    print(f"\n    {progress}  Turn {turn_number}/7"
                          f"  (batch={batch_size}, rep={rep})", end="  ")

                    meas   = measure_turn(history, turn_number, batch_size, monitor)
                    record = build_result_record(
                        quantization, batch_size, rep, turn_number, history, meas
                    )
                    all_results.append(record)

                    # Save each result immediately (crash resilience)
                    rec_path = out_dir / f"{record['run_id']}.json"
                    rec_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))

                    # Console summary
                    if record["status"] == "success":
                        j   = record["energy_j"]
                        jpt = record["j_per_output_token"]
                        tt  = record.get("ttft_ms") or 0
                        vr  = record["vram_peak_mb"]
                        pt  = record["prompt_tokens"]
                        print(f"✓  {j:.1f}J | {jpt:.4f}J/tok | "
                              f"TTFT:{tt:.0f}ms | VRAM:{vr:.0f}MB | "
                              f"ctx:{pt}tok")
                    else:
                        print(f"✗  {record['status']}  {record.get('error','')[:60]}")

                    # Cooling between turns
                    if turn_number < N_TURNS:
                        print(f"    [Cooling {COOLING_S}s...]")
                        time.sleep(COOLING_S)

                # Extra cooling between repetitions
                if rep < len(reps):
                    print(f"\n    [Rep-change cooling {COOLING_S}s...]")
                    time.sleep(COOLING_S)

            # Extra cooling between batch sizes
            if batch_sizes.index(batch_size) < len(batch_sizes) - 1:
                print(f"\n  [Batch-change cooling {COOLING_S * 2}s...]")
                time.sleep(COOLING_S * 2)

    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results saved.")
    finally:
        monitor.cleanup()

        # Write consolidated JSONL
        jsonl_path = out_dir / "multiturn_results.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"\n{'='*65}")
        print(f"  Completed: {len(all_results)}/{total_measurements} measurements")
        n_ok  = sum(1 for r in all_results if r["status"] == "success")
        n_err = len(all_results) - n_ok
        print(f"  Success: {n_ok}  |  Error/OOM: {n_err}")
        print(f"  JSONL: {jsonl_path}")
        print(f"{'='*65}\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-turn LLM energy benchmark (INFERA Exp. 2)"
    )
    parser.add_argument(
        "--quantization", required=True,
        choices=["fp16", "int8_w8a16", "int4_awq"],
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=int, default=BATCH_SIZES,
        help=f"Batch sizes to test (default: {BATCH_SIZES})"
    )
    parser.add_argument(
        "--pilot", action="store_true",
        help="Only rep=1 — quick sanity check before full run"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Override default results directory"
    )
    args = parser.parse_args()

    run_multiturn(
        quantization=args.quantization,
        batch_sizes=args.batch_sizes,
        pilot=args.pilot,
        results_dir=args.results_dir,
    )
