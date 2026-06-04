"""
build_multiturn_conversation.py  — Phase 1 of INFERA Experiment 2.

Generates the FIXED conversation history used by Phase 2 (multiturn_runner.py).
Runs 7 turns against vLLM (FP16, batch=1, temperature=0) and saves the full
conversation to data/multiturn/conversation_history.json.

WHY FIXED HISTORY:
  With temperature=0, all quantization schemes generate identical responses for
  the same input. Pre-generating once and fixing the history makes the design
  auditable: all schemes receive exactly the same sequence of user_content and
  assistant_content strings, ensuring energy differences are attributable only
  to the quantization scheme, not to content variation.

USAGE:
  # vLLM FP16 must be running:
  #   bash scripts/start_vllm_fp16.sh
  python scripts/build_multiturn_conversation.py
  # Output: data/multiturn/conversation_history.json
"""

import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import httpx
except ImportError:
    sys.exit("FATAL: httpx not installed. Run: pip install httpx")

# ── CONFIG ────────────────────────────────────────────────────────────────────
VLLM_URL    = "http://localhost:8000/v1/chat/completions"
MODEL_NAME  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_TOKENS  = 256
TEMPERATURE = 0.0
TIMEOUT_S   = 300

FLOW_PATH   = Path("data/multiturn/conversation_flow.json")
OUTPUT_PATH = Path("data/multiturn/conversation_history.json")


# ── SINGLE TURN (streaming for TTFT + usage) ─────────────────────────────────

async def send_turn_streaming(messages: list[dict]) -> dict:
    """
    Send one turn with stream=True.
    Measures TTFT (time to first content chunk).
    Attempts to get exact token counts via stream_options.include_usage.
    If vLLM does not support stream_options, token counts fall back to estimation.
    """
    payload = {
        "model":       MODEL_NAME,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream":      True,
        "stream_options": {"include_usage": True},  # vLLM ≥0.4.x; silently ignored if unsupported
    }

    t_start = time.perf_counter()
    ttft_ms = None
    chunks  = []
    pt, ct  = 0, 0           # filled from usage chunk if vLLM supports stream_options

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            async with client.stream("POST", VLLM_URL, json=payload) as resp:
                if resp.status_code != 200:
                    return {"status": "error",
                            "error": f"HTTP {resp.status_code}: {(await resp.aread())[:200]}"}

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

                    # stream_options: usage arrives in the final data chunk
                    if chunk.get("usage"):
                        pt = chunk["usage"].get("prompt_tokens", 0)
                        ct = chunk["usage"].get("completion_tokens", 0)

    except Exception as e:
        return {"status": "error", "error": str(e)[:200]}

    t_end        = time.perf_counter()
    full_text    = "".join(chunks)
    total_time_s = t_end - t_start

    # Fallback token count: word-based estimate, capped at MAX_TOKENS
    # Used only if vLLM did not return usage (stream_options not supported)
    token_count_from_api = ct > 0
    if not token_count_from_api:
        ct = min(MAX_TOKENS, max(1, len(full_text.split()) * 4 // 3))

    return {
        "status":              "success",
        "full_text":           full_text,
        "prompt_tokens":       pt,
        "completion_tokens":   ct,
        "token_count_from_api": token_count_from_api,
        "ttft_ms":             round(ttft_ms, 2) if ttft_ms else None,
        "total_time_s":        round(total_time_s, 4),
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

async def build_conversation():
    if not FLOW_PATH.exists():
        sys.exit(f"FATAL: conversation_flow.json not found at {FLOW_PATH}\n"
                 f"Expected at: {FLOW_PATH.resolve()}")

    flow          = json.loads(FLOW_PATH.read_text(encoding="utf-8"))
    system_prompt = flow["system_prompt"]
    doc_content   = flow["documents"]["combined_content"]
    turns_def     = flow["turns"]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*65)
    print("  BUILD MULTITURN CONVERSATION — FP16 baseline pre-generation")
    print("="*65)
    print(f"  Model:      {MODEL_NAME}")
    print(f"  Turns:      {len(turns_def)}  |  max_tokens: {MAX_TOKENS}  |  temp: {TEMPERATURE}")
    print(f"  Output:     {OUTPUT_PATH}")
    print("="*65)

    vllm_messages      = [{"role": "system", "content": system_prompt}]
    generated_turns    = []
    token_counts_ok    = True   # tracks whether vLLM returned API token counts

    for turn_def in turns_def:
        t_num    = turn_def["turn_number"]
        user_msg = turn_def["content"]

        # Turn 1: embed documents in the first user message (RAG pattern)
        user_content = (
            f"Documentos de referencia:\n\n{doc_content}\n\n---\n\n{user_msg}"
            if t_num == 1 else user_msg
        )

        vllm_messages.append({"role": "user", "content": user_content})

        print(f"\n  Turno {t_num}/7  ({len(vllm_messages)} msgs in context)")
        print(f"  Q: {user_msg[:80]}...", flush=True)

        result = await send_turn_streaming(vllm_messages)

        if result["status"] != "success":
            sys.exit(f"\nFATAL at Turn {t_num}: {result.get('error')}")

        response_text       = result["full_text"].strip()
        token_count_from_api = result["token_count_from_api"]
        if not token_count_from_api:
            token_counts_ok = False

        print(f"  A: {response_text[:100].replace(chr(10), ' ')}...")
        print(f"  TTFT: {result['ttft_ms'] or 'N/A'}ms  |  "
              f"{result['total_time_s']:.1f}s  |  "
              f"ct={result['completion_tokens']} "
              f"({'API' if token_count_from_api else 'estimated'})")

        # Append assistant response to grow the context for the next turn
        vllm_messages.append({"role": "assistant", "content": response_text})

        generated_turns.append({
            "turn_number":          t_num,
            "user_content":         user_content,
            "assistant_content":    response_text,
            "prompt_tokens":        result["prompt_tokens"],
            "completion_tokens":    result["completion_tokens"],
            "token_count_from_api": token_count_from_api,
            "ttft_ms":              result["ttft_ms"],
            "total_time_s":         result["total_time_s"],
        })

        if t_num < len(turns_def):
            time.sleep(2)   # brief pause between turns during pre-generation

    # ── WARNINGS ─────────────────────────────────────────────────────────────
    if not token_counts_ok:
        print("\n  ⚠  WARNING: stream_options.include_usage not supported by this vLLM.")
        print("     Token counts in this file are word-based estimates (±10-15%).")
        print("     The same limitation will apply in Phase 2 (multiturn_runner.py).")
        print("     Energy (Joules) is unaffected. j_per_output_token will use estimates.")

    # ── SAVE ─────────────────────────────────────────────────────────────────
    output = {
        "generated_at":          datetime.now(timezone.utc).isoformat(),
        "model":                 MODEL_NAME,
        "quantization":          "fp16",
        "temperature":           TEMPERATURE,
        "max_tokens":            MAX_TOKENS,
        "token_counts_from_api": token_counts_ok,
        "system_prompt":         system_prompt,
        "document_content":      doc_content,
        "turns":                 generated_turns,
        # Full message array stored for audit: verify against build_messages_for_turn output
        "vllm_messages_final":   vllm_messages,
        "note": (
            "Fixed history used by Phase 2. "
            "All quantization schemes receive identical user_content and assistant_content. "
            "Energy differences in Phase 2 are attributable only to the quantization scheme."
        ),
    }

    OUTPUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*65}")
    print(f"  ✓ Saved: {OUTPUT_PATH}")
    print(f"  Turns: {len(generated_turns)}  |  "
          f"Token counts: {'API (exact)' if token_counts_ok else 'estimated'}")
    print(f"  Context at T7: ~{sum(len(t['user_content']) + len(t['assistant_content']) for t in generated_turns) // 4} tokens (rough)")
    print(f"{'='*65}\n")
    print("  Next step:")
    print("    python scripts/multiturn_runner.py --quantization fp16 --batch-sizes 1 --pilot")


if __name__ == "__main__":
    asyncio.run(build_conversation())
