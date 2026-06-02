"""
benchmark_runner.py
Main experiment runner. 81 configurations × 3 repetitions.

CONCURRENCY MODEL — WHY THIS MATTERS FOR VI2:
  VI2 (batch_size) represents concurrent request load. If N requests are
  sent sequentially, vLLM processes them one at a time — there is no
  continuous batching, no shared KV-cache pressure, and the measurement
  is indistinguishable from batch=1 repeated N times. This would invalidate
  VI2 entirely.

  Correct implementation: all N requests in a batch are launched
  simultaneously using asyncio.gather(). They arrive at the vLLM server
  within microseconds of each other and are processed as a genuine
  concurrent batch by vLLM's continuous batching scheduler.

  Implementation: asyncio + httpx.AsyncClient + asyncio.gather()

EXECUTION MODEL — HOW PROMPTS ARE USED:
  Each (config × repetition) is evaluated over ALL 30 valid prompts of
  its case. With batch_size=B, the 30 prompts are processed as
  ceil(30/B) sequential batch calls. Within each batch call, B prompts
  are sent concurrently.

  Batch calls per (config × rep):
    batch=1 → 30 calls (each: 1 concurrent request)
    batch=4 → 8 calls  (7 full + 1 padded; each: 4 concurrent requests)
    batch=8 → 4 calls  (3 full + 1 padded; each: 8 concurrent requests)

  Total batch calls per quantization scheme (27 configs × 3 reps):
    batch=1: 27 × 3 × 30 =  810 batch calls  (810  HTTP requests)
    batch=4: 27 × 3 ×  8 =  216 batch calls  (864  HTTP requests)
    batch=8: 27 × 3 ×  4 =  108 batch calls  (864  HTTP requests)
    Total:                 1,134 batch calls per quantization

  Each batch call = one NVML measurement window = one result JSON file.

PADDING:
  30 is not divisible by 4 or 8. The last batch of each group is padded
  by repeating prompts from the start of the 30-prompt list.
    batch=4: last batch has 2 real + 2 repeated
    batch=8: last batch has 6 real + 2 repeated
  Padded batches are flagged with batch_padded=true in the result JSON.
  Recommendation: exclude padded batches when computing per-prompt averages,
  or weight by real_prompt_count / batch_size.

WARMUP:
  Before the first measured call of each (batch, output_length, case) group,
  WARMUP_REQUESTS=5 concurrent batch calls are sent without NVML monitoring.
  Warmup is per group, not per repetition.

API:
  /v1/chat/completions — vLLM applies the correct LLaMA 3.1 chat template.
  usage.prompt_tokens and usage.completion_tokens are post-template real counts.

USAGE:
  python scripts/benchmark_runner.py --quantization fp16
  python scripts/benchmark_runner.py --quantization fp16 --pilot
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import httpx
except ImportError:
    sys.exit(
        "\nFATAL: httpx not installed.\n"
        "Run: pip install httpx\n"
        "Or:  pip install -r requirements.txt\n"
    )

from gpu_power_monitor import GPUPowerMonitor

VLLM_CHAT_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME    = "meta-llama/Meta-Llama-3.1-8B-Instruct"

BATCH_SIZES      = [1, 4, 8]
OUTPUT_LENGTHS   = [64, 256, 512]
CASES            = ["A", "B", "C"]
REPETITIONS      = 3
PROMPTS_PER_CASE = 30

WARMUP_REQUESTS = 5    # concurrent batch calls before first measured call of each group
COOLING_S       = 120  # seconds between distinct (batch, output_length, case) groups
TIMEOUT_S       = 300  # per-request timeout in seconds
IDLE_WAIT_S     = 5    # idle wait before each measured batch call


# ── CORPUS LOADING + SANITY CHECK ─────────────────────────────────────────────

def load_and_validate_prompts(corpus_path: str) -> dict[str, list]:
    """
    Load corpus. Hard-fail if missing or any case has < 30 valid prompts.
    Returns {"A": [...30 prompts...], "B": [...], "C": [...]}.
    """
    if not Path(corpus_path).exists():
        sys.exit(
            f"\nFATAL: Corpus not found: {corpus_path}\n"
            f"Run: python scripts/build_prompt_dataset.py\n"
        )

    by_case: dict[str, list] = {"A": [], "B": [], "C": []}
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = json.loads(line)
            if p["case"] in by_case and p.get("within_tolerance", True):
                by_case[p["case"]].append(p)

    print("\n=== CORPUS SANITY CHECK ===")
    fail = False
    for case in ["A", "B", "C"]:
        n  = len(by_case[case])
        ok = n >= PROMPTS_PER_CASE
        print(f"  Case {case}: {n} valid prompts  {'✓' if ok else '✗ FAIL (need 30)'}")
        if not ok:
            fail = True

    if fail:
        sys.exit(
            "\nFATAL: One or more cases have fewer than 30 valid prompts.\n"
            "Rebuild: python scripts/build_prompt_dataset.py\n"
        )

    # Deterministic selection of exactly 30
    rng = random.Random(42)
    for case in ["A", "B", "C"]:
        rng.shuffle(by_case[case])
        by_case[case] = by_case[case][:PROMPTS_PER_CASE]

    print("  ✓ All cases ready (30 valid prompts each)\n")
    return by_case


# ── CONCURRENT vLLM CALL ──────────────────────────────────────────────────────

async def _single_chat_request(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_new_tokens: int,
) -> dict:
    """
    Send one chat completion request. Returns usage + status dict.
    Never raises — errors are returned as status dicts.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  max_new_tokens,
        "temperature": 0.0,
        "stream":      False,
    }
    try:
        resp = await client.post(VLLM_CHAT_URL, json=payload)
        body = resp.text.lower()

        if resp.status_code in (413, 507) or "out of memory" in body or "cuda oom" in body:
            return {"status": "oom", "error": resp.text[:200]}
        if resp.status_code != 200:
            return {"status": "error",
                    "error": f"HTTP {resp.status_code}: {resp.text[:150]}"}

        data  = resp.json()
        usage = data.get("usage", {})
        return {
            "status":             "success",
            "prompt_tokens":      usage.get("prompt_tokens", 0),
            "completion_tokens":  usage.get("completion_tokens", 0),
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "error": f"Exceeded {TIMEOUT_S}s"}
    except Exception as e:
        msg = str(e)
        if "out of memory" in msg.lower():
            return {"status": "oom", "error": msg[:200]}
        return {"status": "error", "error": msg[:150]}


async def _call_batch_concurrent(
    batch_messages: list[list[dict]],
    max_new_tokens: int,
) -> dict:
    """
    Launch all N requests in the batch simultaneously via asyncio.gather().

    All requests are created and gathered before any awaits, so they arrive
    at the vLLM server within microseconds of each other. vLLM's continuous
    batching scheduler sees them as concurrent load — this is what VI2 measures.

    Returns aggregate token counts and combined status.
    """
    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        tasks = [
            _single_chat_request(client, msgs, max_new_tokens)
            for msgs in batch_messages
        ]
        # All N requests are launched simultaneously here
        responses = await asyncio.gather(*tasks)

    # Any OOM in any request → whole batch is OOM
    for r in responses:
        if r["status"] == "oom":
            return {"status": "oom", "error": r.get("error", "")}

    errors = [r for r in responses if r["status"] == "error"]
    if errors:
        return {"status": "error",
                "error": "; ".join(e.get("error", "") for e in errors[:2])}

    timeouts = [r for r in responses if r["status"] == "timeout"]
    if timeouts:
        return {"status": "timeout", "error": f"{len(timeouts)} requests timed out"}

    total_pt = sum(r.get("prompt_tokens", 0)     for r in responses)
    total_ct = sum(r.get("completion_tokens", 0) for r in responses)
    return {
        "status":            "success",
        "prompt_tokens":     total_pt,
        "completion_tokens": total_ct,
    }


def call_vllm_concurrent(batch_messages: list[list[dict]], max_new_tokens: int) -> dict:
    """
    Synchronous wrapper around the async concurrent call.
    Records wall-clock time around the full gather (start → all N responses received).
    """
    t0 = time.time()
    result = asyncio.run(_call_batch_concurrent(batch_messages, max_new_tokens))
    result["total_time_s"] = time.time() - t0
    return result


# ── BATCH HELPERS ─────────────────────────────────────────────────────────────

def make_batches(prompts: list[dict], batch_size: int) -> list[list[dict]]:
    """
    Split 30 prompts into sequential groups of batch_size.
    Last group is padded by repeating from the start if 30 % batch_size != 0.
    """
    batches = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        if len(batch) < batch_size:
            pad = batch_size - len(batch)
            batch = batch + prompts[:pad]
        batches.append(batch)
    return batches


def is_padded(batch_idx: int, batch_size: int, total: int = PROMPTS_PER_CASE) -> bool:
    return batch_idx >= (total // batch_size) and (total % batch_size != 0)


# ── WARMUP ────────────────────────────────────────────────────────────────────

def run_warmup(sample_messages: list[list[dict]], max_new_tokens: int,
               n: int = WARMUP_REQUESTS):
    """
    Send n concurrent batch calls (no NVML) to reach GPU steady state.
    Each warmup call is itself concurrent (batch_size requests at once).
    """
    print(f"    [Warmup: {n} concurrent batch calls]", end=" ", flush=True)
    for i in range(n):
        r = call_vllm_concurrent(sample_messages, max_new_tokens)
        if r["status"] == "oom":
            print(f" OOM at call {i+1} — skipping rest")
            return
        print(".", end="", flush=True)
    print()


# ── EXPERIMENT MATRIX ─────────────────────────────────────────────────────────

def build_matrix(quantization: str, pilot: bool) -> list[dict]:
    """
    Full mode:  81 (config × rep) entries, shuffled seed=42.
    Pilot mode: 9 entries (Case A, rep=1, all batch × output combos).
    """
    configs = [
        {"quantization": quantization, "batch_size": b,
         "output_length": o, "context_case": c, "repetition": r}
        for b in BATCH_SIZES
        for o in OUTPUT_LENGTHS
        for c in CASES
        for r in range(1, REPETITIONS + 1)
    ]
    if pilot:
        configs = [x for x in configs
                   if x["context_case"] == "A" and x["repetition"] == 1]
        print(f"PILOT MODE: {len(configs)} (config × rep) entries  [Case A, rep=1]")
        return configs
    random.seed(42)
    random.shuffle(configs)
    print(f"FULL MODE:  {len(configs)} (config × rep) entries  "
          f"[quantization={quantization}]")
    return configs


# ── SINGLE CONFIG × REP ───────────────────────────────────────────────────────

def run_config_rep(
    config:          dict,
    prompts_by_case: dict,
    monitor:         GPUPowerMonitor,
    results_dir:     Path,
) -> list[dict]:
    """
    Run one (config × rep) over all 30 prompts.
    Each batch call launches batch_size concurrent requests.
    Returns list of result dicts, one per batch call.
    """
    case    = config["context_case"]
    batch   = config["batch_size"]
    out_len = config["output_length"]
    quant   = config["quantization"]
    rep     = config["repetition"]

    valid_prompts = prompts_by_case[case]
    batches       = make_batches(valid_prompts, batch)
    n_batches     = len(batches)

    config_key = f"{quant}_b{batch}_o{out_len}_case{case}_rep{rep}"
    print(f"\n  {config_key}  ({n_batches} batch calls × {batch} concurrent requests each)")

    results = []

    for batch_idx, batch_prompts in enumerate(batches):
        padded        = is_padded(batch_idx, batch)
        msgs_in_batch = [p["messages"] for p in batch_prompts]
        prompt_ids    = [p["prompt_id"] for p in batch_prompts]
        vi4_tokens    = int(sum(p.get("measured_input_tokens", p.get("token_count", 0))
                               for p in batch_prompts) / len(batch_prompts))

        run_id = (f"{config_key}"
                  f"_batch{batch_idx+1}of{n_batches}"
                  f"_{int(time.time())}")

        print(f"    [{batch_idx+1}/{n_batches}]", end=" ", flush=True)
        time.sleep(IDLE_WAIT_S)

        # NVML window: start (+ 500ms buffer) → concurrent requests → stop (+ 500ms buffer)
        monitor.start_monitoring()
        vllm_result = call_vllm_concurrent(msgs_in_batch, out_len)
        energy = monitor.stop_monitoring()

        result = {
            "run_id":     run_id,
            "config_key": config_key,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
            # Independent variables
            "vi1_quantization":        quant,
            "vi2_batch_size":          batch,
            "vi3_output_length":       out_len,
            "vi4_context_case":        case,
            "vi4_mean_input_tokens":   vi4_tokens,
            "repetition":              rep,
            # Batch metadata
            "batch_index":             batch_idx + 1,
            "batch_total":             n_batches,
            "batch_actual_size":       len(msgs_in_batch),
            "batch_padded":            padded,
            "prompt_ids_in_batch":     prompt_ids,
            # Status
            "status": vllm_result["status"],
            # Dependent variables
            "energy_j":           None,
            "duration_s":         None,
            "avg_power_w":        None,
            "peak_power_w":       None,
            "vram_peak_mb":       None,
            "vram_used_start_mb": None,
            "vram_total_mb":      None,
            "prompt_tokens":      None,
            "completion_tokens":  None,
            "throughput_tok_s":   None,
            "j_per_token":        None,
            "tpot_ms":            None,
            # Measurement metadata
            "nvml_sample_count":  energy.sample_count,
            "nvml_available":     energy.nvml_available,
            "error_detail":       vllm_result.get("error"),
        }

        if vllm_result["status"] == "success":
            pt  = vllm_result.get("prompt_tokens", 0) or vi4_tokens * batch
            ct  = vllm_result.get("completion_tokens", 0) or out_len * batch
            dur = vllm_result["total_time_s"]

            tput = ct / dur if dur > 0 else 0.0
            jpt  = energy.energy_j / ct if ct > 0 else 0.0

            result.update({
                "energy_j":           round(energy.energy_j, 6),
                "duration_s":         round(dur, 4),
                "avg_power_w":        round(energy.avg_power_w, 2),
                "peak_power_w":       round(energy.peak_power_w, 2),
                "vram_peak_mb":       round(energy.vram_used_mb_peak, 1),
                "vram_used_start_mb": round(energy.vram_used_mb_start, 1),
                "vram_total_mb":      round(energy.vram_total_mb, 1),
                "prompt_tokens":      pt,
                "completion_tokens":  ct,
                "throughput_tok_s":   round(tput, 4),
                "j_per_token":        round(jpt, 6),
                "tpot_ms":            round(dur / ct * 1000, 2) if ct > 0 else None,
            })
            pad_note = " [padded]" if padded else ""
            print(f"✓  {energy.energy_j:.3f} J | {tput:.1f} tok/s | "
                  f"VRAM: {energy.vram_used_mb_peak:.0f} MB | "
                  f"samples: {energy.sample_count}{pad_note}")

        elif vllm_result["status"] == "oom":
            print("⚠  OOM — recorded as inviable")
        else:
            print(f"✗  {vllm_result['status']}: "
                  f"{vllm_result.get('error', '')[:60]}")

        with open(results_dir / f"{run_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        results.append(result)

    return results


# ── MAIN RUNNER ───────────────────────────────────────────────────────────────

def run_benchmark(quantization: str, pilot: bool,
                  corpus_path: str, results_dir: Optional[str]):

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(results_dir or f"results/{quantization}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  BENCHMARK  |  quantization={quantization}  |  output: {out_dir}")
    print(f"{'='*65}")

    prompts_by_case = load_and_validate_prompts(corpus_path)

    # Show batch call breakdown before starting
    print("  Batch calls per (config × rep) for this quantization:")
    for b in BATCH_SIZES:
        n_calls     = math.ceil(PROMPTS_PER_CASE / b)
        n_config_rep = 3 * 3 * 3   # output × cases × reps
        total       = n_config_rep * n_calls
        print(f"    batch={b}: {n_config_rep} config×rep × {n_calls} calls"
              f" = {total} batch calls  ({total * b} concurrent HTTP requests)")
    print()

    configs = build_matrix(quantization, pilot)
    monitor = GPUPowerMonitor(device_index=0)

    summary = {
        "start":                  datetime.now(timezone.utc).isoformat(),
        "quantization":           quantization,
        "pilot":                  pilot,
        "config_rep_total":       len(configs),
        "config_rep_completed":   0,
        "batch_calls_success":    0,
        "batch_calls_oom":        0,
        "batch_calls_error":      0,
        "batch_calls_timeout":    0,
        "warmup_calls_per_group": WARMUP_REQUESTS,
        "concurrency_model":      "asyncio.gather — all batch_size requests launched simultaneously",
    }

    prev_group = None

    try:
        for i, cfg in enumerate(configs):
            group = (cfg["batch_size"], cfg["output_length"], cfg["context_case"])

            if group != prev_group:
                if prev_group is not None:
                    print(f"\n  [Cooling {COOLING_S}s ...]")
                    time.sleep(COOLING_S)

                print(f"\n  {'─'*55}")
                print(f"  New group: batch={group[0]}  output={group[1]}  "
                      f"case={group[2]}")

                # Warmup with concurrent requests matching this group's batch size
                case_prompts = prompts_by_case[cfg["context_case"]]
                warmup_msgs  = [p["messages"]
                                for p in case_prompts[:cfg["batch_size"]]]
                run_warmup(warmup_msgs, cfg["output_length"])
                prev_group = group

            print(f"\n[{i+1:>3}/{len(configs)}]  rep={cfg['repetition']}", end="")
            batch_results = run_config_rep(cfg, prompts_by_case, monitor, out_dir)

            summary["config_rep_completed"] += 1
            for r in batch_results:
                key = f"batch_calls_{r['status']}"
                if key in summary:
                    summary[key] += 1

            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    except KeyboardInterrupt:
        print("\n\nInterrupted — partial results saved.")
    finally:
        monitor.cleanup()
        summary["end"] = datetime.now(timezone.utc).isoformat()
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*65}")
        print(f"  Config×rep completed:  {summary['config_rep_completed']}/{len(configs)}")
        print(f"  Batch calls — success: {summary['batch_calls_success']}  "
              f"oom: {summary['batch_calls_oom']}  "
              f"error: {summary['batch_calls_error']}")
        print(f"  Results: {out_dir}")
        print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization", required=True,
                        choices=["fp16", "int8_w8a16", "int4_awq"])
    parser.add_argument("--pilot", action="store_true",
                        help="Case A only, rep=1, all batch×output combos")
    parser.add_argument("--corpus", default="data/prompts/prompt_corpus.jsonl")
    parser.add_argument("--results-dir", default=None)
    args = parser.parse_args()

    run_benchmark(args.quantization, args.pilot, args.corpus, args.results_dir)
