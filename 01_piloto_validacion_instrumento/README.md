# Etapa 1 — Piloto de validación del instrumento (EXP1)

Diseño factorial 3⁴ × 3 = **243 corridas** con peticiones aisladas (sin
sesión incremental). Su propósito es **validar el protocolo de medición de
energía** (NVML, buffer de muestreo de 500 ms, integración trapezoidal) y
**anclar la relación entre energía y contexto** para contextos cortos
(~256–4096 tokens). No mide calidad de respuesta — eso se incorpora en las
etapas 2 y 3.

Referencia en el paper: §5.3 (diseño) y §6.1 (resultados del piloto, CV por
configuración).

> **Todos los comandos de esta sección asumen que el directorio de trabajo es
> esta carpeta** (`cd 01_piloto_validacion_instrumento`).

---

## Diseño factorial

| Factor | Niveles | Valores |
|---|---|---|
| Esquema de cuantización | 3 | FP16, INT8 (W8A16), AWQ INT4 |
| Tamaño de batch (concurrencia) | 3 | 1, 4, 8 |
| Longitud de salida (tokens) | 3 | 64, 256, 512 |
| Carga contextual | 3 | Caso A (~256 tok), Caso B (~1024 tok), Caso C (~4096 tok) |

3⁴ = 81 configuraciones × 3 repeticiones = **243 corridas**.

---

## Ejecución completa (~7 horas)

```bash
# 1. Contexto: los archivos en data/context/ son placeholders ficticios
#    (empresa "TechSolutions Ecuador"). Ver docs/context_guide.md si se
#    quieren reemplazar por contenido propio.

# 2. Entorno (RunPod RTX 4090 o equivalente)
export HF_TOKEN=hf_...   # token de HuggingFace con acceso a Meta-LLaMA
bash scripts/setup_runpod.sh
python scripts/generate_reproducibility_info.py

# 3. Construir el corpus de prompts (90 prompts: 30 por caso)
python scripts/build_prompt_dataset.py --verify-only
python scripts/build_prompt_dataset.py

# 4. [Terminal 2] Levantar el servidor vLLM para un esquema
bash scripts/start_vllm_fp16.sh

# 5. [Terminal 1] Piloto rápido de humo (9 configs, ~15 min)
python scripts/benchmark_runner.py --quantization fp16 --pilot

# 6. Corrida completa de este esquema (81 configs × 3 reps = 243 corridas)
python scripts/benchmark_runner.py --quantization fp16

# 7. Repetir 4-6 para int8_w8a16 y int4_awq (cambiando el servidor vLLM)
python scripts/benchmark_runner.py --quantization int8_w8a16
python scripts/benchmark_runner.py --quantization int4_awq

# 8. Consolidar y analizar
python scripts/consolidate_results.py
python analysis/analyze.py
```

---

## Reproducción más rápida (`--reps`)

El experimento original corrió con **3 repeticiones por configuración** y
obtuvo, sobre las 81 configuraciones, un **coeficiente de variación (CV)
promedio de 0.7%** y un **máximo de 3.7%**, ambos muy por debajo del umbral
habitual del 15% (§6.1 del paper). Esto indica que, para este instrumento y
este hardware, la varianza entre repeticiones es muy baja.

Para una reproducción más corta, `benchmark_runner.py` acepta `--reps N`:

```bash
# 1 repetición por configuración: 81 corridas (~2.3 h en vez de ~7 h)
python scripts/benchmark_runner.py --quantization fp16 --reps 1

# 2 repeticiones: 162 corridas (~4.6 h), término medio
python scripts/benchmark_runner.py --quantization fp16 --reps 2
```

**Criterio para aceptar una reproducción reducida:** calcular el CV
(desviación estándar / media) de `energy_j` por configuración con
`scripts/consolidate_results.py`. Si con `--reps 1` o `--reps 2` el CV se
mantiene por debajo de ~10%, la reproducción reducida es representativa del
comportamiento medido con 3 repeticiones. Si supera ese umbral, repetir con
`--reps 3` (configuración original).

---

## Estructura

```
01_piloto_validacion_instrumento/
├── README.md
├── scripts/
│   ├── benchmark_runner.py      ← runner principal (acepta --reps)
│   ├── build_prompt_dataset.py  ← construye el corpus de 90 prompts
│   ├── gpu_power_monitor.py     ← medición NVML (energía + VRAM)
│   ├── generate_reproducibility_info.py
│   ├── consolidate_results.py
│   ├── setup_runpod.sh
│   └── start_vllm_{fp16,int8,awq}.sh
├── docs/
│   ├── methodology.md           ← protocolo de medición y decisiones de diseño
│   ├── context_guide.md         ← cómo reemplazar los archivos de contexto
│   └── runpod_guide.md          ← guía paso a paso de ejecución en RunPod
├── data/
│   ├── context/                 ← documentos de contexto (placeholders)
│   ├── conversations/           ← historiales de conversación para el Caso B
│   └── prompts/                 ← corpus generado (prompt_corpus.jsonl, no versionado)
├── analysis/                    ← figuras y tablas del piloto
├── results/                     ← una carpeta por corrida + CSVs consolidados
└── logs/                        ← bitácoras de ejecución
```

---

## Reemplazar los archivos de contexto

Los archivos en `data/context/` son placeholders ficticios de una empresa
("TechSolutions Ecuador"). Ver `docs/context_guide.md` para los presupuestos
de tokens por caso y el procedimiento de reemplazo. El constructor del
corpus valida automáticamente que cada prompt quede dentro de ±15% del
objetivo de tokens por caso.
