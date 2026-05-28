# Context Files Guide

How to replace the placeholder context files with your own domain content.

---

## Why replace the context files?

The files in `data/context/` and `data/conversations/` are fictional placeholder content
representing a Ecuadorian tech company (TechSolutions Ecuador). They exist only to
demonstrate the structure and token budgets required for each experimental case.

For your own deployment evaluation, replace them with content from your actual domain.
The energy measurements are language- and content-agnostic: what matters is token count,
not semantics.

---

## Token budgets per case

The prompt builder validates these automatically. Prompts outside ±15% are flagged but
not discarded — they are recorded with their actual measured token count.

| Case | Role | Target input tokens | Valid range | What to put here |
|------|------|---------------------|-------------|-----------------|
| A | Short context chatbot | ~256 tokens | 218–294 | System prompt + FAQ excerpt |
| B | Memory-augmented assistant | ~1024 tokens | 870–1178 | System + profile + conversation history |
| C | Long document analysis | ~4096 tokens | 3482–4710 | Full document (contract, policy, report) |

---

## File by file

### `data/context/company_profile.md`
**Used in:** Case A (partial), Case B (partial)
**Target contribution:** ~180–250 tokens in the final prompt

Write a description of your organization or system that an assistant would need as
background to answer user questions. Include: who you are, what you do, key services or
products, and contact/process information.

**Token check:**
```bash
python -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
text = open('data/context/company_profile.md').read()
print(f'{len(tok.encode(text))} tokens')
"
```

---

### `data/context/company_policies.md`
**Used in:** Case A (partial)
**Target contribution:** ~100–200 tokens in the final prompt (truncated by build script)

Write your operational policies, FAQ, or service terms. The build script uses only the
first ~400 characters to stay within Case A's budget.

---

### `data/context/sample_contract.md`
**Used in:** Case C
**Target contribution:** ~3800–4000 tokens (nearly the full document)

This is your long document. It should be 3500–4200 tokens on its own. Good options:
- A real contract or agreement (anonymized)
- An internal policy document
- A technical specification
- A research paper or report

**Validate length before running:**
```bash
python scripts/build_prompt_dataset.py --verify-only
```

---

### `data/context/internal_faq.md`
**Used in:** Case C (alternated with sample_contract.md)
**Target contribution:** ~3800–4000 tokens

A second long document for Case C analysis prompts. Half the Case C prompts use
`sample_contract.md` and half use `internal_faq.md`. Having two documents increases
prompt variety and reduces repetition effects.

---

### `data/conversations/conversation_histories.jsonl`
**Used in:** Case B
**Format:** One JSON object per line

Each object must have:
```json
{
  "conversation_id": "T-001",
  "scenario_tag": "descriptive_label",
  "turns": [
    {"role": "user", "content": "First user message"},
    {"role": "assistant", "content": "First assistant response"},
    {"role": "user", "content": "Second user message"}
  ],
  "new_question": "The follow-up question for this conversation"
}
```

You need at least **30 objects**. The build script uses them round-robin for the 30 Case B prompts.

The `turns` field should contain 4–8 turns totaling ~600–750 tokens when serialized.
Use realistic conversations from your domain: support tickets, customer queries,
help desk interactions, etc.

**Validate your histories:**
```bash
python -c "
import json
from pathlib import Path
lines = [json.loads(l) for l in Path('data/conversations/conversation_histories.jsonl').read_text().splitlines() if l.strip()]
print(f'Loaded {len(lines)} conversation histories')
for h in lines[:3]:
    print(f'  {h[\"conversation_id\"]}: {len(h[\"turns\"])} turns | new_question: {h[\"new_question\"][:60]}...')
"
```

---

## After replacing files

Always re-validate before running the benchmark:

```bash
python scripts/build_prompt_dataset.py --verify-only
```

This shows token counts per case without saving anything. Check that all cases are
within the valid range. If Case C documents are too short, add more content. If they are
too long, trim them.

Then build the corpus:

```bash
python scripts/build_prompt_dataset.py
```

---

## Language note

The included placeholders are in Spanish. You can use any language. The energy
measurement methodology is language-agnostic — energy consumption depends on token
count, not language semantics.

However, note that the LLaMA 3.1 tokenizer is less token-efficient for Spanish than
English: the same content expressed in Spanish typically requires 15–20% more tokens.
This affects the absolute VI4 levels but not the relative comparisons across
configurations, which is what this benchmark measures.

If you switch languages, re-run `--verify-only` to confirm your files hit the token targets.
