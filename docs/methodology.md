# Methodology and Design Decisions

Technical documentation of measurement protocol and experimental design decisions.
For use in paper sections 3.x and as reference during thesis defense.

---

## Measurement protocol

### Energy measurement (primary)

**Method:** NVML `nvmlDeviceGetPowerUsage()` via pynvml
**Sampling rate:** 100 ms intervals (10 Hz)
**Integration method:** Trapezoidal rule applied to power-time profile → Joules

**Pre/post-inference buffer: 500 ms**

Husom et al. (2026) [MELODI] empirically tested multiple buffer values:
- 200 ms → 0% capture completeness rate → discarded
- 400 ms → 48% CCR → insufficient
- 500 ms → 100% CCR → adopted

This buffer ensures the sampling window captures the full power profile of each
inference call, including GPU state transitions at the beginning and end.

**Requirement for valid NVML measurements:**
NVML measures total GPU power usage. This requires exclusive GPU operation — no other
processes running on the GPU during measurement. A dedicated RunPod instance satisfies
this requirement. Shared instances invalidate energy measurements (documented in MELODI).

### VRAM tracking

**Method:** `nvmlDeviceGetMemoryInfo()` called at each sampling interval

Three values tracked:
- `vram_used_mb_start`: VRAM at beginning of monitoring window
- `vram_used_mb_peak`: maximum VRAM observed during inference
- `vram_total_mb`: total GPU VRAM capacity

VRAM pressure is directly relevant to quantization effects: FP16 model weights occupy
~16 GB, leaving ~8 GB for KV cache. INT4 AWQ weights occupy ~4–5 GB, leaving ~19 GB.
This difference determines which batch size × context length combinations are feasible.

### Token counting

Token counts are obtained from the vLLM API response field `usage.prompt_tokens` and
`usage.completion_tokens`. These are real counts from the model's actual tokenizer,
not estimates.

The corpus builder also validates token counts using `AutoTokenizer` from HuggingFace
during prompt construction, to confirm VI4 levels are within ±15% tolerance before
the benchmark runs.

**Word count (len(text.split())) is never used for token counting in production.**
Word count can differ 20–40% from real token count, which would invalidate J/token
and throughput calculations.

### CO₂eq triangulation

CodeCarbon runs in parallel and records CO₂-equivalent emissions. This is a secondary
triangulation source. CodeCarbon has documented underestimation of 10–30% compared to
physical power meters (Kappa-Energy Index paper; MELODI). It is not used as the primary
energy measurement — NVML is the primary source.

---

## OOM as experimental result

Out-of-memory events are recorded as valid experimental outcomes with `status: "oom"`,
not as benchmark failures.

OOM results represent the operational viability boundary — configurations that exceed
available VRAM cannot be deployed in practice regardless of their theoretical efficiency.

In the results analysis, OOM is reported as a binary operational feasibility dimension:
a configuration is either viable (success) or not viable (oom) for a given hardware
context. This is scientifically relevant because INT4 AWQ, by reducing model weight
memory from ~16 GB to ~4–5 GB, may enable configurations that are infeasible under FP16.

---

## GPU clock control

**Attempted:** `nvidia-smi -pm 1` and `nvidia-smi --lock-gpu-clocks`
**Result:** RunPod virtualized instances do not grant the root privileges required to
lock GPU clocks via the hypervisor.

This means GPU frequency may vary during execution due to DVFS (Dynamic Voltage and
Frequency Scaling), which is a known source of run-to-run variance for consumer-grade
GPU benchmarks (Bhatia et al., 2025, arXiv:2501.08219).

**Mitigation:** Three repetitions per configuration quantify and report this variance.
The MLPerf Power 60-second minimum run rule is not adopted — it applies to cross-system
comparisons, not within-system configuration comparisons on the same hardware.

---

## Randomization and order effects

The experiment matrix (81 configurations × 3 repetitions = 243 runs) is shuffled
with a fixed random seed (42) before execution. This distributes any temporal effects
(thermal drift, memory fragmentation, performance state changes) uniformly across
configurations rather than confounding them with specific variable levels.

A 2-minute cooling period between configurations allows GPU temperature to stabilize.

---

## The 1/W Law and VI4

Chen et al. (2026) demonstrate analytically that energy per token scales inversely with
context window size (J/token ~ 1/context_window) due to the KV-cache mechanism. This
framework empirically tests this prediction on consumer-grade hardware (RTX 4090) with
concurrent requests — a context not covered by existing benchmarks.

The mechanism: longer context → larger KV cache → each decode step reads more memory →
decode is more memory-bandwidth-bound → utilization of GPU compute stays low → energy
per token increases. This effect interacts with batch size because concurrent requests
share VRAM for KV cache, compressing the available budget.

---

## INT8 overhead at low batch

INT8 W8A16 (via bitsandbytes) stores weights as INT8 and dequantizes to FP16 for
computation. This reduces memory bandwidth for weight loading but adds dequantization
overhead. At low batch sizes (batch=1), this overhead can make INT8 slower and more
energy-intensive than FP16.

This is a documented result (TokenPowerBench, AAAI 2026; MLPerf Power) and is expected
to appear in the data. It is not a measurement error — it is a correct characterization
of INT8 behavior on consumer hardware under low concurrency.

---

## Academic contribution framing

This framework is best described as a **reproducible inference energy benchmark
for consumer-grade GPU deployment**.

It differs from prior work as follows:

| Aspect | MLPerf Power | MELODI | ML.ENERGY | This work |
|--------|-------------|--------|-----------|-----------|
| Hardware target | Datacenter | Single GPU | Datacenter | Consumer GPU |
| Concurrency | High (256+ batch) | None (batch=1) | Mixed | Low-mid (1–8) |
| Context window as IV | No | No | No | Yes |
| OOM as formal result | No | No | No | Yes |
| Practitioner-oriented output | No | No | Partial | Yes |

The practitioner-oriented output is the key differentiator: results are organized into
decision matrices that directly answer "for my use case (latency-sensitive / batch / green),
which configuration should I deploy?"
