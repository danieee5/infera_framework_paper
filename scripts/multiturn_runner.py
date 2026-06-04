"""
multiturn_runner.py  — Phase 2 of INFERA Experiment 2.

Measures GPU energy consumption per conversational turn across quantization
schemes and batch sizes, using the fixed conversation history from Phase 1.

DESIGN:
  - 7-turn enterprise conversation (MOSS security company internal RAG assistant)
  - Fixed history pre-generated in Phase 1 (FP16, temperature=0, batch=1)
  - Each turn measured independently: MELODI 500ms buffer + 120s cooling
  - Context grows organically: ~1600 tokens (T1) → ~3250 tokens (T7)
  - Bridges to EXP1: T1 ≈ Case_B, T5-T7 ≈ Case_C

EXPERIMENTAL MATRIX:
  VI1 (quantization): fp16 | int8_w8a16 | int4_awq
  VI2 (batch_size):   1 | 4
  Turn:               1–7
  Repetitions:        3
  Total/scheme:       2 × 7 × 3 = 42 measurements
  Grand total:        3 schemes × 42 = 126 measurements

HYPOTHESES:
  H1: J/output_token increases monotonically with turn (KV-cache + prefill cost)
  H2: INT8 batch=4 anomaly (+18% J/tok vs batch=1, found in EXP1) persists/amplifies
  H3: AWQ advantage over FP16 erodes as context grows (vs +122%/+59% in EXP1)
  H4: VRAM_delta grows monotonically with turn (KV-cache accumulation proxy)

KEY METRIC NOTES:
  energy_j            — NVML trapezoidal integration (always accurate)
  j_per_output_token  — energy_j / completion_tokens (main efficiency metric)
  token_count_source  — "api" if vLLM returned exact counts; "estimated" otherwise
  tpot_ms             — wall_time / ct_total (server throughput, consistent with EXP1)
  tpot_ms_per_request — wall_time / ct_per_request (per-user latency, correct for batch>1)
  total_context_tokens — prompt tokens per request (input context per user)

USAGE:
  # Start appropriate vLLM first, then:
  python scripts/multiturn_runner.py --quantization fp16 --batch-sizes 1 --pilot  # 7 turns, 1 rep
  python scripts/multiturn_runner.py --quantization fp16                           # full run
  python scripts/multiturn_runner.py --quantization int8_w8a16
  python scripts/multiturn_runner.py --quantization int4_awq
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

sys.path.insert(0, str(Path(__file__).parent))
from gpu_power_monitor import GPUPowerMonitor

# ── CONFIG ────────────────────────────────────────────────────────────────────
VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_TOKENS  = 256          # fixed across all turns — consistent with EXP1 VI3=256
TEMPERATURE = 0.0
TIMEOUT_S   = 300

BATCH_SIZES  = [1, 4]
REPETITIONS  = 3
N_TURNS      = 7
COOLING_S    = 120         # between turns — same as EXP1
IDLE_WAIT_S  = 5           # pre-measurement idle — same as EXP1
WARMUP_TURNS = 2           # warmup calls before first measured rep (Turn 1 context)

HISTORY_PATH = Path("data/multiturn/conversation_history.json")
RESULTS_DIR  = Path("results/multiturn")


# ── MESSAGE BUILDER ───────────────────────────────────────────────────────────

def build_messages_for_turn(history: dict, turn_number: int) -> list[dict]:
    """
    Reconstruct the vLLM message array for a given turn number.

    Structure for Turn N:
      [system] [user_1] [asst_1] [user_2] [asst_2] ... [user_{N-1}] [asst_{N-1}] [user_N]

    The last message is always the user's question for this turn — no assistant
    response is appended because that is what the model must generate now.

    All user_content and assistant_content come from the fixed pre-generated
    history, ensuring all quantization schemes face identical context.
    """
    messages = [{"role": "system", "content": history["system_prompt"]}]
    turns    = history["turns"]

    for i, turn in enumerate(turns[:turn_number]):
        messages.append({"role": "user",      "content": turn["user_content"]})
        if i < turn_number - 1:
            messages.append({"role": "assistant", "content": turn["assistant_content"]})

    return messages


def messages_token_approx(messages: list[dict]) -> int:
    """
    Rough token count for the message array: total characters / 4.
    Used as fallback display value only; does not affect energy calculations.
    For Spanish technical text, actual ratio is ~3.5-4.5 chars/token (±10-15%).
    """
    return sum(len(m["content"]) for m in messages) // 4


# ── STREAMING REQUEST (TTFT + token counts) ───────────────────────────────────

async def single_request_streaming(
    client:   httpx.AsyncClient,
    messages: list[dict],
) -> dict:
    """
    Single request with stream=True.
    Measures TTFT and attempts exact token counts via stream_options.
    Falls back to word-based estimate (capped at MAX_TOKENS) if usage unavailable.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream":      True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.perf_counter()
    ttft_ms = None
    chunks  = []
    pt, ct  = 0, 0

    try:
        async with client.stream("POST", VLLM_URL, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return {"status": "error",
                        "error": f"HTTP {resp.status_code}: {body[:200]}"}

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

                if chunk.get("usage"):
                    pt = chunk["usage"].get("prompt_tokens", 0)
                    ct = chunk["usage"].get("completion_tokens", 0)

    except httpx.TimeoutException:
        return {"status": "timeout", "error": f"Exceeded {TIMEOUT_S}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}

    t_end        = time.perf_counter()
    full_text    = "".join(chunks)
    total_time_s = t_end - t_start

    token_count_from_api = ct > 0
    if not token_count_from_api:
        # Fallback: word count estimate capped at MAX_TOKENS
        # Capping is justified: temperature=0 typically generates ≤MAX_TOKENS tokens
        ct = min(MAX_TOKENS, max(1, len(full_text.split()) * 4 // 3))
        # pt remains 0 — caller handles via messages_token_approx

    return {
        "status":               "success",
        "prompt_tokens":        pt,
        "completion_tokens":    ct,
        "token_count_from_api": token_count_from_api,
        "ttft_ms":              round(ttft_ms, 2) if ttft_ms else None,
        "total_time_s":         round(total_time_s, 4),
    }


# ── CONCURRENT BATCH ─────────────────────────────────────────────────────────

def call_concurrent(messages_list: list[list[dict]]) -> dict:
    """
    Send batch_size identical requests simultaneously via asyncio.gather().
    Each request carries the same conversation context (same turn, same history).
    Models 'batch_size enterprise users at the same stage of their conversation'.

    total_time_s = max across all requests (wall clock for the full batch).
    prompt_tokens / completion_tokens = sum across all requests.
    ttft_ms_mean = mean TTFT across successful requests.
    """
    async def _run():
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            tasks = [
                single_request_streaming(client, msgs)
                for msgs in messages_list
            ]
            return await asyncio.gather(*tasks)

    results   = asyncio.run(_run())
    successes = [r for r in results if r["status"] == "success"]

    if not successes:
        return {
            "status": results[0].get("status", "error"),
            "error":  results[0].get("error", "all requests failed"),
        }

    n = len(successes)
    any_from_api = any(r["token_count_from_api"] for r in successes)

    return {
        "status":               "success",
        "prompt_tokens":        sum(r["prompt_tokens"]    for r in successes),
        "completion_tokens":    sum(r["completion_tokens"] for r in successes),
        "token_count_from_api": any_from_api,
        "ttft_ms_mean":         sum(r["ttft_ms"] or 0     for r in successes) / n,
        # Wall clock = time until ALL concurrent requests finish (NVML window stays open)
        "total_time_s":         max(r["total_time_s"]     for r in successes),
        "n_success":            n,
        "n_batch":              len(results),
    }


# ── WARMUP ────────────────────────────────────────────────────────────────────

def run_warmup(history: dict, batch_size: int):
    """
    WARMUP_TURNS un-measured inference calls before the first measured rep.
    Uses Turn 1 context (shortest). Note: does NOT warm up the memory allocation
    pattern for later turns (T5-T7). This may cause slightly higher latency on
    first rep of long turns — documented as a known limitation.
    """
    msgs  = build_messages_for_turn(history, 1)
    batch = [msgs] * batch_size
    print(f"    [Warmup {WARMUP_TURNS}×]", end=" ", flush=True)
    for _ in range(WARMUP_TURNS):
        call_concurrent(batch)
        print(".", end="", flush=True)
    print()


# ── SINGLE TURN MEASUREMENT ───────────────────────────────────────────────────

def measure_turn(
    history:     dict,
    turn_number: int,
    batch_size:  int,
    monitor:     GPUPowerMonitor,
    messages_approx_tokens: int,
) -> dict:
    """
    MELODI protocol for one turn:
      idle 5s → start_monitoring (500ms pre-buffer) → inference → stop_monitoring (500ms post-buffer)
    """
    messages      = build_messages_for_turn(history, turn_number)
    messages_list = [messages] * batch_size

    time.sleep(IDLE_WAIT_S)

    monitor.start_monitoring()               # 500ms pre-buffer
    vllm_result = call_concurrent(messages_list)
    energy      = monitor.stop_monitoring()  # 500ms post-buffer

    return {
        "vllm_result":           vllm_result,
        "energy":                energy,
        "input_tokens_approx":   messages_approx_tokens,
    }


# ── RESULT RECORD ─────────────────────────────────────────────────────────────

def build_result_record(
    quantization: str,
    batch_size:   int,
    repetition:   int,
    turn_number:  int,
    meas:         dict,
) -> dict:
    vr  = meas["vllm_result"]
    en  = meas["energy"]

    run_id = (
        f"mt_{quantization}_b{batch_size}_t{turn_number}"
        f"_rep{repetition}_{int(time.time())}"
    )

    record = {
        "run_id":           run_id,
        "experiment":       "multiturn",
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        # ── Independent variables ─────────────────────────────────────────────
        "vi1_quantization": quantization,
        "vi2_batch_size":   batch_size,
        "turn_number":      turn_number,
        "repetition":       repetition,
        # ── Context metadata ──────────────────────────────────────────────────
        "input_tokens_approx":           meas["input_tokens_approx"],
        "cumulative_turns_in_context":   turn_number - 1,
        # ── Status ────────────────────────────────────────────────────────────
        "status":           vr.get("status", "unknown"),
        "error":            vr.get("error"),
        # ── Token count reliability ───────────────────────────────────────────
        # "api"       — vLLM returned exact counts via stream_options
        # "estimated" — fallback: min(MAX_TOKENS, word_count * 4/3)
        "token_count_source": None,
        # ── Energy (NVML MELODI protocol) ─────────────────────────────────────
        "energy_j":         None,
        "duration_s":       None,
        "avg_power_w":      None,
        "peak_power_w":     None,
        "vram_peak_mb":     None,
        "vram_start_mb":    None,
        "vram_delta_mb":    None,    # peak − start: KV-cache footprint proxy
        "nvml_samples":     en.sample_count,
        "nvml_available":   en.nvml_available,
        # ── Throughput & latency ──────────────────────────────────────────────
        "prompt_tokens":                 None,   # aggregate (all concurrent requests)
        "completion_tokens":             None,   # aggregate
        "prompt_tokens_per_request":     None,   # per user (= total_context_tokens)
        "completion_tokens_per_request": None,
        "total_context_tokens":          None,   # input context per user — key for KV analysis
        "ttft_ms":                       None,   # mean across concurrent requests
        # tpot_ms:            wall_time / ct_total  — server throughput (consistent with EXP1)
        # tpot_ms_per_request:wall_time / ct_per_req — per-user latency (correct for batch>1)
        "tpot_ms":                       None,
        "tpot_ms_per_request":           None,
        "throughput_tok_s":              None,
        "j_per_output_token":            None,
        # ── Batch metadata ────────────────────────────────────────────────────
        "n_batch_success":  vr.get("n_success"),
        "n_batch_total":    vr.get("n_batch"),
    }

    if vr.get("status") == "success":
        ct_total = vr.get("completion_tokens", 0)
        pt_total = vr.get("prompt_tokens",     0)
        dur      = vr.get("total_time_s",      0)
        ttft     = vr.get("ttft_ms_mean")
        n_req    = max(1, vr.get("n_success", batch_size))
        from_api = vr.get("token_count_from_api", False)

        ct_per_req = max(1, ct_total // n_req)
        pt_per_req = (
            max(1, pt_total // n_req)
            if pt_total > 0
            else meas["input_tokens_approx"]   # fallback to char-based estimate
        )

        tput         = ct_total / dur if dur > 0 else 0.0
        jpt          = en.energy_j / ct_total if ct_total > 0 else 0.0
        tpot_server  = dur / ct_total * 1000   if ct_total > 0 else None
        tpot_per_req = dur / ct_per_req * 1000 if ct_per_req > 0 else None
        vram_delta   = round(en.vram_used_mb_peak - en.vram_used_mb_start, 1)

        record.update({
            "token_count_source":            "api" if from_api else "estimated",
            "energy_j":                      round(en.energy_j, 6),
            "duration_s":                    round(dur, 4),
            "avg_power_w":                   round(en.avg_power_w, 2),
            "peak_power_w":                  round(en.peak_power_w, 2),
            "vram_peak_mb":                  round(en.vram_used_mb_peak, 1),
            "vram_start_mb":                 round(en.vram_used_mb_start, 1),
            "vram_delta_mb":                 vram_delta,
            "prompt_tokens":                 pt_total,
            "completion_tokens":             ct_total,
            "prompt_tokens_per_request":     pt_per_req,
            "completion_tokens_per_request": ct_per_req,
            "total_context_tokens":          pt_per_req,
            "ttft_ms":                       round(ttft, 2) if ttft else None,
            "tpot_ms":                       round(tpot_server,  2) if tpot_server  else None,
            "tpot_ms_per_request":           round(tpot_per_req, 2) if tpot_per_req else None,
            "throughput_tok_s":              round(tput, 4),
            "j_per_output_token":            round(jpt, 6),
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
            f"\nFATAL: Conversation history not found at {HISTORY_PATH.resolve()}\n"
            f"Run Phase 1 first:\n"
            f"  bash scripts/start_vllm_fp16.sh\n"
            f"  python scripts/build_multiturn_conversation.py\n"
        )

    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))

    # Warn if history was built with estimated token counts
    if not history.get("token_counts_from_api", True):
        print("\n  ⚠  NOTICE: conversation_history.json was built with estimated token counts.")
        print("     j_per_output_token will use word-based estimates (±10-15%).")
        print("     Energy (Joules) is unaffected. Comparisons between schemes remain valid.\n")

    turns = list(range(1, N_TURNS + 1))
    reps  = [1] if pilot else list(range(1, REPETITIONS + 1))

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_dir or f"{RESULTS_DIR}/{quantization}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(batch_sizes) * len(turns) * len(reps)

    print(f"\n{'='*65}")
    print(f"  MULTI-TURN BENCHMARK  |  quantization={quantization}")
    print(f"  Batch sizes: {batch_sizes}  |  Turns: {N_TURNS}  |  Reps: {reps}")
    print(f"  Total measurements this run: {total}")
    if pilot:
        print("  MODE: PILOT (rep=1 only)")
    print(f"  Output: {out_dir}")
    print(f"{'='*65}\n")

    monitor      = GPUPowerMonitor(device_index=0)
    all_results  = []
    idx          = 0
    # Track token count source across run for final warning
    estimated_count_warned = False

    try:
        for batch_size in batch_sizes:

            print(f"\n{'─'*55}")
            print(f"  Batch size = {batch_size}")
            run_warmup(history, batch_size)

            for rep in reps:
                print(f"\n  Rep {rep}/{len(reps)}:")

                for turn_number in turns:
                    idx += 1
                    msgs_approx = messages_token_approx(
                        build_messages_for_turn(history, turn_number)
                    )

                    print(f"\n    [{idx:>3}/{total}]  "
                          f"T{turn_number}/7  batch={batch_size}  rep={rep}  "
                          f"~{msgs_approx}tok_ctx", end="  ")

                    meas   = measure_turn(history, turn_number, batch_size,
                                         monitor, msgs_approx)
                    record = build_result_record(
                        quantization, batch_size, rep, turn_number, meas
                    )
                    all_results.append(record)

                    # ── Immediate save for crash resilience ──────────────────
                    (out_dir / f"{record['run_id']}.json").write_text(
                        json.dumps(record, indent=2, ensure_ascii=False)
                    )

                    # ── Console summary ──────────────────────────────────────
                    if record["status"] == "success":
                        src = "A" if record["token_count_source"] == "api" else "E"
                        print(
                            f"✓  {record['energy_j']:.1f}J | "
                            f"{record['j_per_output_token']:.4f}J/tok({src}) | "
                            f"TTFT:{record.get('ttft_ms') or 0:.0f}ms | "
                            f"VRAM:{record['vram_peak_mb']:.0f}MB "
                            f"(Δ{record['vram_delta_mb']:.0f}) | "
                            f"ctx:{record['total_context_tokens']}tok"
                        )
                        # ── First-measurement token count verification ───────
                        if idx == 1:
                            print(f"\n    ── TOKEN COUNT VERIFICATION ──")
                            print(f"    prompt_tokens:     {record['prompt_tokens']}")
                            print(f"    completion_tokens: {record['completion_tokens']}")
                            print(f"    source:            {record['token_count_source']}")
                            if record["token_count_source"] == "estimated":
                                print(f"    ⚠  stream_options not supported — "
                                      f"using word-count fallback")
                            else:
                                print(f"    ✓  API token counts confirmed")
                            print()

                    else:
                        print(f"✗  {record['status']}  {record.get('error','')[:60]}")

                    if turn_number < N_TURNS:
                        print(f"    [Cooling {COOLING_S}s]")
                        time.sleep(COOLING_S)

                if rep < len(reps):
                    print(f"\n    [Rep-change cooling {COOLING_S}s]")
                    time.sleep(COOLING_S)

            if batch_sizes.index(batch_size) < len(batch_sizes) - 1:
                print(f"\n  [Batch-change cooling {COOLING_S * 2}s]")
                time.sleep(COOLING_S * 2)

    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results saved.")
    finally:
        monitor.cleanup()

        jsonl_path = out_dir / "multiturn_results.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        n_ok  = sum(1 for r in all_results if r["status"] == "success")
        n_est = sum(1 for r in all_results
                    if r.get("token_count_source") == "estimated")

        print(f"\n{'='*65}")
        print(f"  Completed: {len(all_results)}/{total}")
        print(f"  Success: {n_ok}  |  Errors: {len(all_results) - n_ok}")
        if n_est > 0:
            print(f"  ⚠  {n_est} records used estimated token counts")
        print(f"  JSONL: {jsonl_path}")
        print(f"{'='*65}\n")

        if pilot and n_ok == N_TURNS:
            print("  PILOT: All 7 turns succeeded.")
            print("  Verify energy and VRAM grow from T1 to T7 before full run.")
            print("  If token_count_source == 'estimated', decide on fallback strategy.\n")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="INFERA Exp.2 — Multi-turn LLM energy benchmark"
    )
    parser.add_argument("--quantization", required=True,
                        choices=["fp16", "int8_w8a16", "int4_awq"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=BATCH_SIZES)
    parser.add_argument("--pilot", action="store_true",
                        help="rep=1 only — fast validation before full run")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    run_multiturn(
        quantization=args.quantization,
        batch_sizes=args.batch_sizes,
        pilot=args.pilot,
        results_dir=args.results_dir,
    )
