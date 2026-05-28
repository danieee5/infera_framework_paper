"""
build_prompt_dataset.py
Builds the 90-prompt evaluation corpus from context files.

THREE SCENARIOS (context-augmented prompting — NOT full RAG):
  Case A — Customer support chatbot        target: ~256 input tokens
  Case B — Memory-augmented assistant      target: ~1024 input tokens
  Case C — Long document analysis          target: ~4096 input tokens

  These scenarios simulate the inference workload AFTER context has been
  retrieved and assembled — the computationally relevant phase for energy
  measurement. The retrieval step is not evaluated.

TOKENIZATION POLICY:
  By default, requires AutoTokenizer from HuggingFace (real LLaMA tokenizer).
  If the model cannot be loaded and --allow-tokenizer-fallback is NOT set,
  the script aborts. This is intentional: approximate token counts invalidate
  J/token and throughput calculations, which are the primary dependent variables.

  Token count heuristics (tiktoken, word split) are ONLY permitted when
  --allow-tokenizer-fallback is explicitly passed. This flag is provided for
  development convenience only and must NOT be used for thesis data collection.

  To pre-download the tokenizer without downloading full model weights:
    python -c "
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    "

PROMPT FORMAT:
  Prompts are stored as lists of message dicts (OpenAI chat format):
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
  The benchmark runner sends these to /v1/chat/completions, which lets vLLM
  apply the correct LLaMA 3.1 chat template internally. This avoids manually
  reconstructing <|begin_of_text|>...<|eot_id|> special tokens.

  Token counts in the corpus are estimated pre-template using the tokenizer.
  The actual prompt_tokens reported by vLLM (post-template) will be slightly
  higher (~10–20 tokens for template overhead). This difference is documented
  and consistent across all configurations, so it does not affect comparisons.

VALIDITY TOLERANCE:
  ±15% of target token count per case (documented in methodology).
  Out-of-range prompts are flagged. The script ABORTS if any case has fewer
  than 30 valid prompts — it will not save a broken corpus.

USAGE:
  # Normal (requires real tokenizer):
  python scripts/build_prompt_dataset.py
  python scripts/build_prompt_dataset.py --verify-only

  # Development only (approximate counts — DO NOT USE FOR THESIS DATA):
  python scripts/build_prompt_dataset.py --allow-tokenizer-fallback

  # Custom model ID (if tokenizer is cached elsewhere):
  python scripts/build_prompt_dataset.py --model-id meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import json
import random
import sys
from pathlib import Path

RANDOM_SEED      = 42
PROMPTS_PER_CASE = 30
OUTPUT_PATH      = "data/prompts/prompt_corpus.jsonl"

TARGETS = {
    "A": {"target": 256,  "min": 218,  "max": 294},
    "B": {"target": 1024, "min": 870,  "max": 1178},
    "C": {"target": 4096, "min": 3482, "max": 4710},
}

CONTEXT_FILES = {
    "profile":   "data/context/company_profile.md",
    "policies":  "data/context/company_policies.md",
    "contract":  "data/context/sample_contract.md",
    "faq":       "data/context/internal_faq.md",
    "histories": "data/conversations/conversation_histories.jsonl",
}


# ── TOKENIZER ─────────────────────────────────────────────────────────────────

def load_tokenizer(model_id: str, allow_fallback: bool):
    """
    Load the real LLaMA tokenizer.

    If allow_fallback=False (default) and the tokenizer cannot be loaded,
    the script aborts. This is the correct behavior for thesis data collection.

    If allow_fallback=True (--allow-tokenizer-fallback flag), falls back to
    tiktoken then word heuristic. FOR DEVELOPMENT ONLY.
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_id)
        print(f"[Tokenizer] ✓ Loaded: {model_id}")
        return tok, model_id
    except Exception as e:
        if not allow_fallback:
            sys.exit(
                f"\nFATAL: Could not load tokenizer for '{model_id}'.\n"
                f"Error: {e}\n\n"
                f"The real model tokenizer is required for valid token counts.\n"
                f"Fix options:\n"
                f"  1. Run setup first: bash scripts/setup_runpod.sh\n"
                f"     (downloads the model and tokenizer to /workspace/models/)\n"
                f"  2. Pre-download tokenizer only:\n"
                f"     python -c \"from transformers import AutoTokenizer; "
                f"AutoTokenizer.from_pretrained('{model_id}')\"\n"
                f"  3. Development only (DO NOT use for thesis data):\n"
                f"     python scripts/build_prompt_dataset.py --allow-tokenizer-fallback\n"
            )

        # Fallback path — only reached with --allow-tokenizer-fallback
        print(f"[Tokenizer] WARNING: Could not load {model_id}: {e}")
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
            class _TiktokenWrap:
                def encode(self, t): return enc.encode(t)
            print("[Tokenizer] FALLBACK: tiktoken cl100k_base")
            print("[Tokenizer] ⚠ Token counts are approximate (~4% error vs LLaMA)")
            print("[Tokenizer] ⚠ DO NOT use these counts for thesis data collection")
            return _TiktokenWrap(), "tiktoken_cl100k_base_FALLBACK"
        except ImportError:
            class _WordWrap:
                def encode(self, t): return t.split()
            print("[Tokenizer] FALLBACK: word split heuristic (very approximate)")
            print("[Tokenizer] ⚠ DO NOT use these counts for thesis data collection")
            return _WordWrap(), "word_split_heuristic_FALLBACK"


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens using the loaded tokenizer."""
    return len(tokenizer.encode(text))


# ── CONTEXT LOADING ───────────────────────────────────────────────────────────

def load_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"\nFATAL: Context file not found: {path}\n"
            f"See docs/context_guide.md for instructions on replacing context files.\n"
        )
    return p.read_text(encoding="utf-8")


def load_histories(path: str) -> list:
    p = Path(path)
    if not p.exists():
        sys.exit(
            f"\nFATAL: Conversation histories not found: {path}\n"
            f"See docs/context_guide.md.\n"
        )
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ── PROMPT BUILDERS — return OpenAI chat message lists ───────────────────────
#
# Each function returns a list of {"role": ..., "content": ...} dicts.
# This is the format expected by /v1/chat/completions (OpenAI-compatible API).
# vLLM applies the correct LLaMA 3.1 chat template (with <|begin_of_text|>,
# <|eot_id|>, etc.) internally — we do not reconstruct it manually.
#
# For token counting in the corpus, we count the concatenated content text
# (system + user). The actual post-template token count reported by vLLM will
# be ~10–20 tokens higher due to template overhead tokens. This offset is
# constant across all configurations and does not affect relative comparisons.

QUESTIONS_A = [
    "¿Cuál es el horario de atención del soporte técnico?",
    "¿Cómo solicito una visita técnica a domicilio?",
    "¿Qué garantía tienen los equipos que instalan?",
    "¿Ofrecen contratos de mantenimiento mensual?",
    "¿Hacen envíos fuera de la ciudad?",
    "¿Cómo puedo pagar mis facturas en línea?",
    "¿En cuánto tiempo reparan un equipo que entrego?",
    "¿Tienen descuentos para empresas con múltiples equipos?",
    "¿Qué pasa si el equipo se daña durante la reparación?",
    "¿Puedo hacer seguimiento del estado de mi solicitud?",
    "¿Tienen cobertura en provincias fuera de Quito?",
    "¿Qué incluye el diagnóstico gratuito?",
    "¿Cuánto demora una instalación de red empresarial?",
    "¿Cómo cancelo un servicio contratado?",
    "¿Qué hace falta para solicitar soporte remoto?",
]


def build_case_a(index: int, profile: str, policies: str) -> list[dict]:
    """Case A: ~256 input tokens. System context + short user question."""
    system_content = (
        "Eres el asistente virtual de atención al cliente de TechSolutions Ecuador, "
        "empresa especializada en soluciones tecnológicas empresariales. "
        "Responde de forma clara, amigable y concisa.\n\n"
        f"INFORMACIÓN DE LA EMPRESA:\n{profile[:600]}\n\n"
        f"POLÍTICAS DE SERVICIO:\n{policies[:400]}"
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": QUESTIONS_A[index % len(QUESTIONS_A)]},
    ]


def build_case_b(index: int, profile: str, history: dict) -> list[dict]:
    """
    Case B: ~1024 input tokens. System + company profile + serialized history + question.
    Only turns and new_question from the history dict enter the prompt.
    conversation_id and scenario_tag are stored as corpus metadata only.
    """
    history_text = "\n".join(
        f"[{'Usuario' if t['role'] == 'user' else 'Asistente'}]: {t['content']}"
        for t in history.get("turns", [])[:8]
    )
    system_content = (
        "Eres el asistente de soporte empresarial de TechSolutions Ecuador. "
        "Tienes acceso al historial completo de esta conversación y al perfil de la empresa. "
        "Usa el historial para dar respuestas contextualizadas y coherentes.\n\n"
        f"PERFIL DE LA EMPRESA:\n{profile[:900]}\n\n"
        f"HISTORIAL DE CONVERSACIÓN:\n{history_text}"
    )
    question = history.get("new_question",
                           "¿Puede resumir los problemas reportados y su estado actual?")
    return [
        {"role": "system", "content": system_content},
        {"role": "user",   "content": question},
    ]


TASKS_C = [
    ("contract",
     "Analiza este contrato e identifica: (1) obligaciones de cada parte, "
     "(2) condiciones de pago, (3) penalidades por incumplimiento, "
     "(4) mecanismo de resolución de conflictos."),
    ("contract",
     "Resume los puntos clave: monto total, plazos, forma de pago, garantías "
     "y condiciones de terminación anticipada."),
    ("contract",
     "¿Cuáles son los riesgos financieros más importantes para el CONTRATANTE? "
     "Explica cada uno con referencia a cláusulas específicas."),
    ("contract",
     "Crea un calendario de pagos con las condiciones que deben cumplirse "
     "para cada desembolso."),
    ("contract",
     "¿Bajo qué condiciones puede terminarse este contrato anticipadamente? "
     "Explica las consecuencias económicas para cada parte."),
    ("faq",
     "Resume los derechos laborales principales: jornada, remuneración, "
     "vacaciones y permisos con sueldo."),
    ("faq",
     "¿Cuáles son las causas de terminación del contrato laboral y cuál es "
     "el proceso disciplinario completo?"),
    ("faq",
     "Un trabajador con 5 años de antigüedad quiere saber sus beneficios. "
     "Extrae y calcula la respuesta exacta."),
    ("faq",
     "¿Cómo funcionan las horas extras? ¿Cuándo se pagan, con qué recargo "
     "y requieren autorización previa?"),
    ("faq",
     "Crea una guía de bolsillo para empleado nuevo: derechos y obligaciones "
     "más importantes en lenguaje simple."),
]


def build_case_c(index: int, contract: str, faq: str) -> list[dict]:
    """Case C: ~4096 input tokens. Full document + analysis instruction."""
    task_key, instruction = TASKS_C[index % len(TASKS_C)]
    document = contract if task_key == "contract" else faq
    return [
        {"role": "system",
         "content": "Eres un asistente especializado en análisis de documentos empresariales. "
                    "Analiza el documento completo antes de responder."},
        {"role": "user",
         "content": f"{instruction}\n\nDOCUMENTO:\n{document}"},
    ]


def messages_text_for_counting(messages: list[dict]) -> str:
    """Concatenate all message content for pre-template token estimation."""
    return " ".join(m["content"] for m in messages)


# ── CORPUS BUILDER ────────────────────────────────────────────────────────────

def build_corpus(tokenizer, tokenizer_id: str, output_path: str, verify_only: bool):
    profile   = load_file(CONTEXT_FILES["profile"])
    policies  = load_file(CONTEXT_FILES["policies"])
    contract  = load_file(CONTEXT_FILES["contract"])
    faq       = load_file(CONTEXT_FILES["faq"])
    histories = load_histories(CONTEXT_FILES["histories"])

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(histories)

    prompts      = []
    out_of_range = []

    for i in range(PROMPTS_PER_CASE):

        # ── Case A ────────────────────────────────────────────────────────────
        msgs_a = build_case_a(i, profile, policies)
        tok_a  = count_tokens(messages_text_for_counting(msgs_a), tokenizer)
        in_a   = TARGETS["A"]["min"] <= tok_a <= TARGETS["A"]["max"]
        prompts.append({
            "prompt_id":             f"A_{i:02d}",
            "case":                  "A",
            "vi4_level":             "short",
            "vi4_target_tokens":     256,
            "measured_input_tokens": tok_a,
            "within_tolerance":      in_a,
            "tokenizer_used":        tokenizer_id,
            "messages":              msgs_a,   # list of {role, content} dicts
        })
        if not in_a:
            out_of_range.append(f"A_{i:02d}: {tok_a} tokens (target 218–294)")

        # ── Case B ────────────────────────────────────────────────────────────
        hist   = histories[i % len(histories)]
        msgs_b = build_case_b(i, profile, hist)
        tok_b  = count_tokens(messages_text_for_counting(msgs_b), tokenizer)
        in_b   = TARGETS["B"]["min"] <= tok_b <= TARGETS["B"]["max"]
        prompts.append({
            "prompt_id":             f"B_{i:02d}",
            "case":                  "B",
            "vi4_level":             "medium",
            "vi4_target_tokens":     1024,
            "measured_input_tokens": tok_b,
            "within_tolerance":      in_b,
            "tokenizer_used":        tokenizer_id,
            # conversation_id and scenario_tag: corpus metadata ONLY.
            # Neither field appears in msgs_b or is sent to the model.
            "source_conversation_id": hist.get("conversation_id", ""),
            "source_scenario_tag":    hist.get("scenario_tag", ""),
            "messages":               msgs_b,
        })
        if not in_b:
            out_of_range.append(f"B_{i:02d}: {tok_b} tokens (target 870–1178)")

        # ── Case C ────────────────────────────────────────────────────────────
        msgs_c = build_case_c(i, contract, faq)
        tok_c  = count_tokens(messages_text_for_counting(msgs_c), tokenizer)
        in_c   = TARGETS["C"]["min"] <= tok_c <= TARGETS["C"]["max"]
        prompts.append({
            "prompt_id":             f"C_{i:02d}",
            "case":                  "C",
            "vi4_level":             "long",
            "vi4_target_tokens":     4096,
            "measured_input_tokens": tok_c,
            "within_tolerance":      in_c,
            "tokenizer_used":        tokenizer_id,
            "messages":              msgs_c,
        })
        if not in_c:
            out_of_range.append(f"C_{i:02d}: {tok_c} tokens (target 3482–4710)")

    # ── Validation report ─────────────────────────────────────────────────────
    print("\n=== CORPUS VALIDATION ===")
    for case in ["A", "B", "C"]:
        cp     = [p for p in prompts if p["case"] == case]
        tokens = [p["measured_input_tokens"] for p in cp]
        t      = TARGETS[case]
        valid  = sum(1 for p in cp if p["within_tolerance"])
        print(f"  Case {case}: n={len(tokens)} | "
              f"mean={sum(tokens)/len(tokens):.0f} | "
              f"min={min(tokens)} max={max(tokens)} | "
              f"target={t['target']} ±15%=[{t['min']},{t['max']}] | "
              f"valid={valid}/{len(tokens)}")

    if out_of_range:
        print(f"\n  OUT OF RANGE ({len(out_of_range)} prompts) — details:")
        for item in out_of_range:
            print(f"    {item}")
    else:
        print("\n  ✓ All prompts within ±15% tolerance")

    # ── Sanity check — hard fail before saving ────────────────────────────────
    print("\n=== SANITY CHECK ===")
    fail = False
    for case in ["A", "B", "C"]:
        valid_count = sum(1 for p in prompts
                          if p["case"] == case and p["within_tolerance"])
        ok = valid_count >= PROMPTS_PER_CASE
        print(f"  Case {case}: {valid_count}/{PROMPTS_PER_CASE} valid  "
              f"{'✓' if ok else '✗ FAIL'}")
        if not ok:
            fail = True

    if fail:
        sys.exit(
            "\nFATAL: One or more cases have fewer than 30 valid prompts.\n"
            "Fix your context files (see docs/context_guide.md) and re-run:\n"
            "  python scripts/build_prompt_dataset.py --verify-only\n"
            "  python scripts/build_prompt_dataset.py\n"
            "Corpus NOT saved.\n"
        )

    if verify_only:
        print("\n  --verify-only: corpus not saved.")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\n  ✓ Corpus saved: {output_path} ({len(prompts)} prompts)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the 90-prompt evaluation corpus for the LLM energy benchmark"
    )
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="HuggingFace model ID for the real tokenizer (default: LLaMA 3.1 8B Instruct)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check token counts and run sanity check without saving corpus"
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH,
        help=f"Output path for prompt corpus (default: {OUTPUT_PATH})"
    )
    parser.add_argument(
        "--allow-tokenizer-fallback",
        action="store_true",
        help=(
            "DEVELOPMENT ONLY. Allow approximate token counting if real tokenizer "
            "cannot be loaded. DO NOT use for thesis data collection."
        )
    )
    args = parser.parse_args()

    tokenizer, tokenizer_id = load_tokenizer(
        args.model_id,
        allow_fallback=args.allow_tokenizer_fallback,
    )
    build_corpus(tokenizer, tokenizer_id, args.output, args.verify_only)
