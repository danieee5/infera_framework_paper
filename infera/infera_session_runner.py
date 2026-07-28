"""
infera_session_runner.py
Runner principal de una sesión incremental INFERA.

Corre UNA sesion incremental por configuracion:
  (cuantizacion x brazo x repeticion)

Brazos:
  --brazo completo   : el historial crece sin intervencion (referencia).
  --brazo resumen    : al cruzar el umbral de tokens, el modelo resume el historial
                       y la sesion continua sobre ese resumen.
  --brazo descarte   : al cruzar el mismo umbral, se descartan los pares de turno
                       mas antiguos y se conservan los N mas recientes. No emite
                       ninguna peticion, por lo que su costo de inferencia es cero.

Resumen y descarte comparten disparador. K=4 no garantiza igual ocupacion:
la ocupacion realizada debe medirse y toda comparacion queda condicionada a
ese parametro.

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
      --quant FP16 --brazo completo --rep 1 \
      --kb-dir kb --tasks config/session_tasks.example.json \
      --out results/runs/mi_corrida/run_FP16_completo_rep1.jsonl
"""

import argparse
import hashlib
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gpu_power_monitor import GPUPowerMonitor
from infera_kb import build_fixed_context
from infera_quality import score_task
from infera_compaction import build_compaction_messages, apply_handoff


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    """Escribe JSON completo y lo publica mediante rename atómico."""
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    with temporary.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def validate_complete_pairs(history: list[dict[str, str]]) -> None:
    if len(history) % 2:
        raise ValueError("el historial no contiene un número entero de pares")
    for index in range(0, len(history), 2):
        roles = (history[index].get("role"), history[index + 1].get("role"))
        if roles != ("user", "assistant"):
            raise ValueError(
                f"par {index // 2}: roles inválidos {roles}; "
                "se esperaban user/assistant"
            )


def retain_recent_complete_pairs(
    history: list[dict[str, str]],
    task_ids: list[str],
    pairs_to_keep: int,
) -> tuple[list[dict[str, str]], list[str], int]:
    """Devuelve copias del sufijo formado por los K pares completos recientes."""
    if pairs_to_keep <= 0:
        raise ValueError("pairs_to_keep debe ser positivo")
    validate_complete_pairs(history)
    pair_count = len(history) // 2
    if len(task_ids) != pair_count:
        raise ValueError("task_ids no corresponde con los pares del historial")
    discarded = max(0, pair_count - pairs_to_keep)
    start = discarded * 2
    return (
        [dict(message) for message in history[start:]],
        list(task_ids[discarded:]),
        discarded,
    )


class TokenBudgetGuard:
    """Cuenta el prompt con la plantilla real antes de emitir una petición."""

    def __init__(self, tokenizer: Any, reference: str):
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("el tokenizer no declara chat_template")
        self.tokenizer = tokenizer
        self.reference = reference
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None or not hasattr(backend, "to_str"):
            raise ValueError("se requiere un tokenizer fast serializable")
        special = getattr(tokenizer, "special_tokens_map", {})
        self.fingerprint = {
            "reference": reference,
            "resolved_commit": tokenizer.init_kwargs.get("_commit_hash"),
            "tokenizer_class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "backend_sha256": sha256_text(backend.to_str()),
            "chat_template_sha256": sha256_text(tokenizer.chat_template),
            "special_tokens_sha256": sha256_text(canonical_json(special)),
        }

    @classmethod
    def from_reference(cls, reference: str) -> "TokenBudgetGuard":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Falta transformers; instala requirements-gpu.txt"
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(
            reference,
            local_files_only=Path(reference).exists(),
            use_fast=True,
        )
        return cls(tokenizer, reference)

    def count_prompt(self, messages: list[dict[str, str]]) -> int:
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens)

    def check(
        self,
        messages: list[dict[str, str]],
        requested_max_tokens: int,
        max_model_len: int,
    ) -> int:
        prompt_tokens = self.count_prompt(messages)
        if prompt_tokens + requested_max_tokens > max_model_len:
            raise RuntimeError(
                "presupuesto excedido antes de llamar a vLLM: "
                f"{prompt_tokens} + {requested_max_tokens} > {max_model_len}"
            )
        return prompt_tokens


def call_vllm(
    url,
    model,
    messages,
    temperature,
    max_tokens,
    seed,
    timeout,
):
    """Llama una vez a vLLM. Un fallo aborta; no se reintenta una inferencia."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]
        usage = data["usage"]
        prompt_tokens = int(usage["prompt_tokens"])
        completion_tokens = int(usage["completion_tokens"])
        response_metadata = {
            "response_id": data.get("id"),
            "response_created": data.get("created"),
            "response_model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
        }
    except Exception as exc:
        raise RuntimeError(f"falló la petición a vLLM: {exc}") from exc
    if not isinstance(text, str):
        raise RuntimeError("vLLM devolvió contenido no textual")
    if prompt_tokens <= 0 or completion_tokens < 0:
        raise RuntimeError(f"usage inválido devuelto por vLLM: {usage!r}")
    return text, prompt_tokens, completion_tokens, response_metadata


def validate_nvml_trace(
    trace: Any,
    sample_count: int,
    context: str,
) -> dict[str, float]:
    """Impide publicar mediciones sin evidencia o con procesos GPU ajenos."""
    if not isinstance(trace, list) or len(trace) != sample_count:
        raise RuntimeError(
            f"{context}: traza NVML ausente/incompleta "
            f"({len(trace) if isinstance(trace, list) else 'no-lista'} "
            f"!= {sample_count})"
        )
    if sample_count < 2:
        raise RuntimeError(f"{context}: menos de dos muestras NVML")
    process_fraction = sum(
        sample.get("process_query_available") is True for sample in trace
    ) / sample_count
    if process_fraction < 0.9:
        raise RuntimeError(
            f"{context}: NVML verificó procesos solo en "
            f"{process_fraction:.1%} de las muestras"
        )
    foreign = sorted({
        int(pid)
        for sample in trace
        for pid in (sample.get("foreign_compute_pids") or [])
    })
    if foreign:
        raise RuntimeError(
            f"{context}: procesos GPU ajenos detectados durante la llamada: "
            f"{foreign}"
        )
    timestamps = [
        float(sample["timestamp_monotonic_s"]) for sample in trace
    ]
    if any(
        timestamps[index] <= timestamps[index - 1]
        for index in range(1, len(timestamps))
    ):
        raise RuntimeError(f"{context}: timestamps NVML no son crecientes")
    duration = timestamps[-1] - timestamps[0]
    expected = max(
        2,
        int(duration / (GPUPowerMonitor.SAMPLING_MS / 1000.0)) + 1,
    )
    sampling_fraction = sample_count / expected
    if sampling_fraction < 0.9:
        raise RuntimeError(
            f"{context}: cobertura temporal NVML insuficiente "
            f"({sample_count}/{expected})"
        )
    return {
        "sampling_fraction": sampling_fraction,
        "trace_duration_s": duration,
    }


def measured_call(
    monitor,
    guard,
    url,
    model,
    messages,
    temperature,
    max_tokens,
    seed,
    timeout,
    max_model_len,
    baseline_power_w,
):
    """Envuelve una llamada vLLM con medicion de energia NVML."""
    preflight_prompt_tokens = guard.check(messages, max_tokens, max_model_len)
    monitor.start_monitoring()              # incluye buffer pre 500ms
    t0 = time.time()
    try:
        text, ptok, ctok, response_metadata = call_vllm(
            url,
            model,
            messages,
            temperature,
            max_tokens,
            seed,
            timeout,
        )
    finally:
        wall = time.time() - t0
        energy = monitor.stop_monitoring(
            baseline_power_w=baseline_power_w,
        )
    if ptok != preflight_prompt_tokens:
        raise RuntimeError(
            "el conteo del tokenizer no coincide con vLLM: "
            f"preflight={preflight_prompt_tokens}, usage={ptok}"
        )
    if ptok + max_tokens > max_model_len:
        raise RuntimeError(
            f"vLLM ejecutó una llamada fuera de presupuesto: "
            f"{ptok} + {max_tokens} > {max_model_len}"
        )
    trace_validation = validate_nvml_trace(
        energy.samples,
        energy.sample_count,
        "llamada vLLM",
    )
    buffer_coverage_s = trace_validation["trace_duration_s"] - wall
    if buffer_coverage_s < 0.75:
        raise RuntimeError(
            "la traza NVML no cubre los buffers pre/post declarados: "
            f"duración={trace_validation['trace_duration_s']:.3f}s, "
            f"petición={wall:.3f}s"
        )
    return {
        "text": text, "prompt_tokens": ptok, "completion_tokens": ctok,
        "preflight_prompt_tokens": preflight_prompt_tokens,
        "requested_max_tokens": max_tokens,
        "token_budget_ok": True,
        "wall_s": wall,
        "energy_j": energy.energy_j, "duration_s": energy.duration_s,
        "avg_power_w": energy.avg_power_w, "peak_power_w": energy.peak_power_w,
        "vram_peak_mb": energy.vram_used_mb_peak, "nvml_samples": energy.sample_count,
        "nvml_available": energy.nvml_available,
        "baseline_power_w": energy.baseline_power_w,
        "energy_above_baseline_j": energy.energy_above_baseline_j,
        "nvml_trace": energy.samples,
        "nvml_sampling_fraction": trace_validation["sampling_fraction"],
        "nvml_buffer_coverage_s": buffer_coverage_s,
        **response_metadata,
    }


def run_session(args):
    out = Path(args.out)
    partial = out.with_name(out.name + ".partial")
    manifest_path = Path(str(out) + ".manifiesto.json")
    occupied = [path for path in (out, partial, manifest_path) if path.exists()]
    if occupied and not args.overwrite:
        raise SystemExit(
            "La sesión colisiona con salidas existentes: "
            + ", ".join(str(path) for path in occupied)
            + ". "
            "Usa otra etiqueta o añade --overwrite de forma explícita."
        )
    if args.overwrite:
        for path in occupied:
            path.unlink()
    out.parent.mkdir(parents=True, exist_ok=True)

    kb_system = build_fixed_context(args.kb_dir)
    tasks_doc = json.loads(Path(args.tasks).read_text(encoding="utf-8"))
    dec = tasks_doc.get("decoding", {"temperature": 0.0, "max_tokens": 512, "seed": 42})
    temperature = float(dec.get("temperature", 0.0))
    max_tokens = int(dec.get("max_tokens", 512))
    seed = int(dec.get("seed", 42))
    tasks = tasks_doc["tasks"]
    expected_tasks = args.expected_tasks or len(tasks)
    if len(tasks) != expected_tasks:
        raise SystemExit(
            f"se esperaban {expected_tasks} tareas y el escenario contiene "
            f"{len(tasks)}"
        )
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise SystemExit(f"tarea {index}: se esperaba un objeto")
        for field in ("id", "type", "prompt"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise SystemExit(
                    f"tarea {index}: {field} debe ser texto no vacío"
                )
    task_ids = [task["id"] for task in tasks]
    if len(set(task_ids)) != len(task_ids):
        raise SystemExit("los ids de tarea deben ser únicos")
    if (
        max_tokens <= 0
        or args.max_model_len <= 0
        or args.compaction_threshold <= 0
    ):
        raise SystemExit(
            "max_tokens, max_model_len y compaction_threshold deben ser positivos"
        )
    if args.compaction_threshold + max_tokens > args.max_model_len:
        raise SystemExit(
            "el umbral no deja la salida reservada dentro del presupuesto: "
            f"{args.compaction_threshold} + {max_tokens} > {args.max_model_len}"
        )
    if args.pares_conservados <= 0:
        raise SystemExit("--pares-conservados debe ser positivo")
    if args.segundos_reposo <= 0:
        raise SystemExit("--segundos-reposo debe ser positivo")
    if args.post_warmup_settle_s < 0:
        raise SystemExit("--post-warmup-settle-s no puede ser negativo")
    if args.warmup < 0:
        raise SystemExit("--warmup no puede ser negativo")
    if args.request_timeout_s <= 0:
        raise SystemExit("--request-timeout-s debe ser positivo")
    if args.device_index < 0:
        raise SystemExit("--device-index no puede ser negativo")
    expected_server_pgid = getattr(args, "expected_server_pgid", None)
    if expected_server_pgid is None or expected_server_pgid <= 0:
        raise SystemExit(
            "--expected-server-pgid debe identificar el grupo de vLLM"
        )
    if not 0 < args.min_baseline_sample_fraction <= 1:
        raise SystemExit(
            "--min-baseline-sample-fraction debe estar en (0, 1]"
        )

    tokenizer_reference = args.tokenizer or args.model
    guard = TokenBudgetGuard.from_reference(tokenizer_reference)
    run_id = f"{args.quant}_{args.arm}_rep{args.rep}"
    runner_hash = sha256_file(Path(__file__))
    base_manifest = {
        "schema_version": 3,
        "status": "running",
        "run_id": run_id,
        "quant": args.quant,
        "brazo": args.arm,
        "rep": args.rep,
        "iniciado_utc": utc_now(),
        "umbral_tokens": args.compaction_threshold,
        "pares_conservados": (
            args.pares_conservados if args.arm == "descarte" else None
        ),
        "escenario": str(Path(args.tasks).resolve()),
        "escenario_sha256": sha256_file(Path(args.tasks)),
        "kb_dir": str(Path(args.kb_dir).resolve()),
        "kb_sha256": sha256_text(kb_system),
        "runner_sha256": runner_hash,
        "modelo_servido": args.model,
        "modelo_fuente": args.model_source or args.model,
        "tokenizer": guard.fingerprint,
        "temperatura": temperature,
        "semilla": seed,
        "max_tokens": max_tokens,
        "max_model_len": args.max_model_len,
        "request_timeout_s": args.request_timeout_s,
        "calentamiento_solicitado": args.warmup,
        "post_warmup_settle_s": args.post_warmup_settle_s,
        "expected_tasks": expected_tasks,
        "require_intervention": args.require_intervention,
        "expected_server_pgid": expected_server_pgid,
    }
    atomic_write_json(manifest_path, base_manifest)

    records: list[dict[str, Any]] = []
    cumulative_energy = 0.0
    cumulative_mechanism_energy = 0.0
    intervention_count = 0
    discard_count = 0
    history: list[dict[str, str]] = []
    history_task_ids: list[str] = []
    intervention_armed = True
    awaiting_reset = False
    reposo: dict[str, Any] | None = None
    warmup_records: list[dict[str, Any]] = []
    monitor = None
    partial_fh = None

    def publish(record: dict[str, Any]) -> None:
        records.append(record)
        assert partial_fh is not None
        partial_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        partial_fh.flush()
        os.fsync(partial_fh.fileno())

    try:
        partial_fh = partial.open("x", encoding="utf-8")
        monitor = GPUPowerMonitor(
            device_index=args.device_index,
            allowed_process_group=expected_server_pgid,
        )
        if not monitor.available:
            raise RuntimeError(
                "NVML no está disponible; no se publicará una sesión energética"
            )
        base_manifest["gpu"] = monitor.device_metadata()
        base_manifest["telemetry_probe"] = monitor.telemetry_probe(
            require_complete=True
        )
        atomic_write_json(manifest_path, base_manifest)

        warmup_messages = [
            {"role": "system", "content": kb_system},
            {"role": "user", "content": "Responde OK."},
        ]
        for warmup_index in range(args.warmup):
            warmup_prompt = guard.check(
                warmup_messages,
                8,
                args.max_model_len,
            )
            warmup_started = time.time()
            _, usage_prompt, usage_completion, warmup_metadata = call_vllm(
                args.vllm_url,
                args.model,
                warmup_messages,
                temperature,
                8,
                seed,
                args.request_timeout_s,
            )
            warmup_records.append({
                "index": warmup_index + 1,
                "prompt_tokens": usage_prompt,
                "completion_tokens": usage_completion,
                "request_wall_s": round(time.time() - warmup_started, 4),
                "recorded_utc": utc_now(),
                **warmup_metadata,
            })
            if usage_prompt != warmup_prompt:
                raise RuntimeError(
                    "tokenizer/vLLM difieren durante warmup "
                    f"{warmup_index + 1}: {warmup_prompt} != {usage_prompt}"
                )
            if warmup_metadata.get("finish_reason") == "length":
                raise RuntimeError(
                    f"warmup {warmup_index + 1} terminó por max_tokens"
                )
        if args.post_warmup_settle_s:
            print(
                "  estabilización posterior al warmup: "
                f"{args.post_warmup_settle_s:.0f} s"
            )
            time.sleep(args.post_warmup_settle_s)

        reposo = monitor.medir_reposo(args.segundos_reposo)
        if not reposo.get("disponible"):
            raise RuntimeError("NVML no produjo muestras durante el reposo")
        if reposo["fraccion_muestras"] < args.min_baseline_sample_fraction:
            raise RuntimeError(
                "muestreo de reposo incompleto: "
                f"{reposo['muestras']}/{reposo['muestras_esperadas']} "
                f"(< {args.min_baseline_sample_fraction:.0%})"
            )
        if float(reposo.get("process_query_fraction", 0.0)) < 0.9:
            raise RuntimeError(
                "NVML verificó exclusividad en menos del 90% del reposo"
            )
        if reposo.get("foreign_compute_pids"):
            raise RuntimeError(
                "procesos GPU ajenos detectados durante el reposo: "
                f"{reposo['foreign_compute_pids']}"
            )
        baseline_power_w = float(reposo["potencia_reposo_mediana_w"])
        print(
            f"  reposo: {baseline_power_w:.1f} W "
            f"({reposo['muestras']}/{reposo['muestras_esperadas']} muestras)"
        )

        for idx, task in enumerate(tasks):
            messages = (
                [{"role": "system", "content": kb_system}]
                + history
                + [{"role": "user", "content": task["prompt"]}]
            )
            res = measured_call(
                monitor,
                guard,
                args.vllm_url,
                args.model,
                messages,
                temperature,
                max_tokens,
                seed,
                args.request_timeout_s,
                args.max_model_len,
                baseline_power_w,
            )
            if not res["nvml_available"] or res["nvml_samples"] < 2:
                raise RuntimeError(f"{task['id']}: medición NVML inválida")
            q = score_task(
                res["text"],
                task.get("verify", {}),
                judge_fn=None,
            )
            cumulative_energy += res["energy_j"]
            j_per_tok = (
                res["energy_j"] / res["completion_tokens"]
                if res["completion_tokens"]
                else None
            )
            rec = {
                "schema_version": 3,
                "run_id": run_id,
                "quant": args.quant,
                "arm": args.arm,
                "rep": args.rep,
                "task_index": idx,
                "task_id": task["id"],
                "task_type": task["type"],
                "accumulated_prompt_tokens": res["prompt_tokens"],
                "preflight_prompt_tokens": res["preflight_prompt_tokens"],
                "completion_tokens": res["completion_tokens"],
                "requested_max_tokens": res["requested_max_tokens"],
                "max_model_len": args.max_model_len,
                "token_budget_ok": res["token_budget_ok"],
                "energy_j": round(res["energy_j"], 4),
                "mechanism_energy_j": 0.0,
                "energy_above_baseline_j": round(
                    res["energy_above_baseline_j"], 4
                ),
                "baseline_power_w": round(res["baseline_power_w"], 3),
                "j_per_completion_token": (
                    round(j_per_tok, 5) if j_per_tok else None
                ),
                "duration_s": round(res["duration_s"], 4),
                "request_wall_s": round(res["wall_s"], 4),
                "avg_power_w": round(res["avg_power_w"], 2),
                "peak_power_w": round(res["peak_power_w"], 2),
                "vram_peak_mb": round(res["vram_peak_mb"], 1),
                "quality": q["quality"],
                "quality_subscores": q["subscores"],
                "quality_is_programmatic": True,
                "prompt_text": task["prompt"],
                "response_text": res["text"],
                "finish_reason": res["finish_reason"],
                "vllm_response_id": res["response_id"],
                "vllm_response_created": res["response_created"],
                "vllm_response_model": res["response_model"],
                "messages_sha256": sha256_text(canonical_json(messages)),
                "cumulative_energy_j": round(cumulative_energy, 4),
                "cumulative_mechanism_energy_j": round(
                    cumulative_mechanism_energy, 4
                ),
                "is_compaction": False,
                "is_mechanism_event": False,
                "es_descarte": False,
                "intervention_index": intervention_count,
                "compaction_index": intervention_count,
                "status": "ok",
                "error": "",
                "nvml_samples": res["nvml_samples"],
                "nvml_available": res["nvml_available"],
                "nvml_trace": res["nvml_trace"],
                "nvml_sampling_fraction": round(
                    res["nvml_sampling_fraction"], 4
                ),
                "nvml_buffer_coverage_s": round(
                    res["nvml_buffer_coverage_s"], 4
                ),
                "recorded_utc": utc_now(),
                "threshold_tokens": args.compaction_threshold,
                "pares_conservados_config": (
                    args.pares_conservados
                    if args.arm == "descarte"
                    else None
                ),
            }
            publish(rec)
            print(
                f"  [{task['id']:>4} {task['type']:<10}] "
                f"ctx={res['prompt_tokens']:>5}tok  "
                f"E={res['energy_j']:>7.2f}J  "
                f"Q={q['quality']:.2f}  (ok)"
            )

            history.append({"role": "user", "content": task["prompt"]})
            history.append({"role": "assistant", "content": res["text"]})
            history_task_ids.append(task["id"])

            if args.arm in ("resumen", "descarte", "compaction"):
                if awaiting_reset:
                    if res["prompt_tokens"] >= args.compaction_threshold:
                        raise RuntimeError(
                            f"{task['id']}: la intervención previa no bajó "
                            f"el prompt de {args.compaction_threshold}; se "
                            "aborta para evitar intervenir en cada turno"
                        )
                    awaiting_reset = False
                    intervention_armed = True

                should_intervene = (
                    intervention_armed
                    and res["prompt_tokens"] >= args.compaction_threshold
                    and idx < len(tasks) - 1
                )
                if not should_intervene:
                    continue

                if args.arm in ("resumen", "compaction"):
                    comp_msgs = build_compaction_messages(kb_system, history)
                    comp = measured_call(
                        monitor,
                        guard,
                        args.vllm_url,
                        args.model,
                        comp_msgs,
                        temperature,
                        max_tokens,
                        seed,
                        args.request_timeout_s,
                        args.max_model_len,
                        baseline_power_w,
                    )
                    if not comp["text"].strip():
                        raise RuntimeError("el resumen medido quedó vacío")
                    if not comp["nvml_available"] or comp["nvml_samples"] < 2:
                        raise RuntimeError("medición NVML inválida en resumen")
                    cumulative_energy += comp["energy_j"]
                    cumulative_mechanism_energy += comp["energy_j"]
                    intervention_count += 1
                    publish({
                        "schema_version": 3,
                        "run_id": run_id,
                        "quant": args.quant,
                        "arm": args.arm,
                        "rep": args.rep,
                        "task_index": idx + 0.5,
                        "task_id": f"COMPACT_{intervention_count}",
                        "task_type": "COMPACTION",
                        "accumulated_prompt_tokens": comp["prompt_tokens"],
                        "preflight_prompt_tokens": comp[
                            "preflight_prompt_tokens"
                        ],
                        "completion_tokens": comp["completion_tokens"],
                        "requested_max_tokens": comp[
                            "requested_max_tokens"
                        ],
                        "max_model_len": args.max_model_len,
                        "token_budget_ok": comp["token_budget_ok"],
                        "energy_j": round(comp["energy_j"], 4),
                        "mechanism_energy_j": round(comp["energy_j"], 4),
                        "energy_above_baseline_j": round(
                            comp["energy_above_baseline_j"], 4
                        ),
                        "baseline_power_w": round(
                            comp["baseline_power_w"], 3
                        ),
                        "j_per_completion_token": None,
                        "duration_s": round(comp["duration_s"], 4),
                        "request_wall_s": round(comp["wall_s"], 4),
                        "avg_power_w": round(comp["avg_power_w"], 2),
                        "peak_power_w": round(comp["peak_power_w"], 2),
                        "vram_peak_mb": round(comp["vram_peak_mb"], 1),
                        "quality": None,
                        "quality_subscores": {},
                        "quality_is_programmatic": True,
                        "prompt_text": (
                            "[COMPACTION_INSTRUCTION] resumir sesión en "
                            "handoff estructurado"
                        ),
                        "response_text": comp["text"],
                        "finish_reason": comp["finish_reason"],
                        "vllm_response_id": comp["response_id"],
                        "vllm_response_created": comp[
                            "response_created"
                        ],
                        "vllm_response_model": comp["response_model"],
                        "messages_sha256": sha256_text(
                            canonical_json(comp_msgs)
                        ),
                        "trigger_prompt_tokens": res["prompt_tokens"],
                        "cumulative_energy_j": round(cumulative_energy, 4),
                        "cumulative_mechanism_energy_j": round(
                            cumulative_mechanism_energy, 4
                        ),
                        "is_compaction": True,
                        "is_mechanism_event": True,
                        "es_descarte": False,
                        "intervention_index": intervention_count,
                        "compaction_index": intervention_count,
                        "status": "ok",
                        "error": "",
                        "nvml_samples": comp["nvml_samples"],
                        "nvml_available": comp["nvml_available"],
                        "nvml_trace": comp["nvml_trace"],
                        "nvml_sampling_fraction": round(
                            comp["nvml_sampling_fraction"], 4
                        ),
                        "nvml_buffer_coverage_s": round(
                            comp["nvml_buffer_coverage_s"], 4
                        ),
                        "recorded_utc": utc_now(),
                        "threshold_tokens": args.compaction_threshold,
                        "pares_conservados_config": None,
                    })
                    print(
                        f"  [RESUMEN #{intervention_count}] "
                        f"E={comp['energy_j']:.2f}J "
                        f"(trigger={res['prompt_tokens']} tok)"
                    )
                    history = apply_handoff(kb_system, comp["text"])[1:]
                    history_task_ids = []
                else:
                    history_before_hash = sha256_text(
                        canonical_json(history)
                    )
                    retained, retained_ids, discarded = (
                        retain_recent_complete_pairs(
                            history,
                            history_task_ids,
                            args.pares_conservados,
                        )
                    )
                    if discarded <= 0:
                        raise RuntimeError(
                            "el umbral se disparó pero no hay pares antiguos "
                            "que descartar; revise K y el escenario"
                        )
                    history = retained
                    history_task_ids = retained_ids
                    intervention_count += 1
                    discard_count += 1
                    publish({
                        "schema_version": 3,
                        "run_id": run_id,
                        "quant": args.quant,
                        "arm": args.arm,
                        "rep": args.rep,
                        "task_index": idx + 0.5,
                        "task_id": f"DESCARTE_{discard_count}",
                        "task_type": "DESCARTE",
                        # No hubo petición: estos campos no pueden duplicar los
                        # tokens de la tarea que activó el recorte.
                        "accumulated_prompt_tokens": 0,
                        "preflight_prompt_tokens": None,
                        "completion_tokens": 0,
                        "requested_max_tokens": 0,
                        "max_model_len": args.max_model_len,
                        "token_budget_ok": True,
                        "context_tokens_before_intervention": res[
                            "prompt_tokens"
                        ],
                        "energy_j": 0.0,
                        "mechanism_energy_j": 0.0,
                        "energy_above_baseline_j": 0.0,
                        "baseline_power_w": baseline_power_w,
                        "j_per_completion_token": None,
                        "duration_s": 0.0,
                        "request_wall_s": 0.0,
                        "avg_power_w": None,
                        "peak_power_w": None,
                        "vram_peak_mb": None,
                        "quality": None,
                        "quality_subscores": {},
                        "quality_is_programmatic": True,
                        "prompt_text": (
                            "[DESCARTE] recorte local; no se emitió petición"
                        ),
                        "response_text": "",
                        "finish_reason": None,
                        "vllm_response_id": None,
                        "vllm_response_created": None,
                        "vllm_response_model": None,
                        "cumulative_energy_j": round(cumulative_energy, 4),
                        "cumulative_mechanism_energy_j": round(
                            cumulative_mechanism_energy, 4
                        ),
                        "is_compaction": False,
                        "is_mechanism_event": True,
                        "es_descarte": True,
                        "pares_descartados": discarded,
                        "pares_conservados": len(history) // 2,
                        "tareas_conservadas_ids": list(history_task_ids),
                        "historial_antes_sha256": history_before_hash,
                        "historial_conservado_sha256": sha256_text(
                            canonical_json(history)
                        ),
                        "intervention_index": intervention_count,
                        "compaction_index": intervention_count,
                        "status": "ok",
                        "error": "",
                        "nvml_samples": None,
                        "nvml_available": None,
                        "nvml_trace": [],
                        "nvml_sampling_fraction": None,
                        "nvml_buffer_coverage_s": None,
                        "recorded_utc": utc_now(),
                        "threshold_tokens": args.compaction_threshold,
                        "pares_conservados_config": args.pares_conservados,
                    })
                    print(
                        f"  [DESCARTE #{discard_count}] retira {discarded} "
                        f"pares; conserva {len(history) // 2} recientes; 0 J"
                    )
                intervention_armed = False
                awaiting_reset = True

        task_records = [
            record for record in records
            if record["task_type"] not in ("COMPACTION", "DESCARTE")
        ]
        if len(task_records) != expected_tasks:
            raise RuntimeError(
                f"sesión incompleta: {len(task_records)} tareas medidas"
            )
        if any(record["status"] != "ok" for record in task_records):
            raise RuntimeError("la sesión contiene tareas con error")
        if (
            args.require_intervention
            and args.arm in ("resumen", "descarte", "compaction")
            and intervention_count == 0
        ):
            raise RuntimeError(
                "el umbral nunca se disparó; la sesión no responde la pregunta"
            )

        partial_fh.close()
        partial_fh = None
        os.replace(partial, out)
        output_directory_fd = os.open(out.parent, os.O_RDONLY)
        try:
            os.fsync(output_directory_fd)
        finally:
            os.close(output_directory_fd)
        final_manifest = {
            **base_manifest,
            "status": "complete",
            "finalizado_utc": utc_now(),
            "reposo": reposo,
            "calentamiento_completado": args.warmup,
            "warmup_records": warmup_records,
            "filas": len(records),
            "tareas_medidas": len(task_records),
            "intervenciones": intervention_count,
            "descartes": discard_count,
            "energia_total_j": round(cumulative_energy, 4),
            "energia_mecanismo_j": round(
                cumulative_mechanism_energy, 4
            ),
            "raw_sha256": sha256_file(out),
            "raw_bytes": out.stat().st_size,
        }
        atomic_write_json(manifest_path, final_manifest)
    except BaseException as exc:
        failure_manifest = {
            **base_manifest,
            "status": "failed",
            "finalizado_utc": utc_now(),
            "reposo": reposo,
            "warmup_records": warmup_records,
            "filas_parciales": len(records),
            "tareas_parciales": sum(
                record["task_type"] not in ("COMPACTION", "DESCARTE")
                for record in records
            ),
            "intervenciones_parciales": intervention_count,
            "energia_parcial_j": round(cumulative_energy, 4),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
            "partial_path": str(partial) if partial.exists() else None,
        }
        atomic_write_json(manifest_path, failure_manifest)
        raise
    finally:
        if partial_fh is not None:
            partial_fh.close()
        if monitor is not None:
            monitor.cleanup()

    print(f"\n[OK] {len(records)} registros completos -> {out}")
    print(
        f"[OK] Energía total: {cumulative_energy:.2f} J | "
        f"intervenciones: {intervention_count}"
    )


def main():
    ap = argparse.ArgumentParser(
        description="INFERA — runner de una sesión incremental"
    )
    ap.add_argument("--vllm-url", default="http://localhost:8000/v1/chat/completions")
    ap.add_argument("--model", required=True, help="ruta o id del modelo servido por vLLM")
    ap.add_argument("--quant", required=True, help="etiqueta de cuantizacion (FP16/INT8/AWQ)")
    ap.add_argument(
        "--brazo",
        dest="arm",
        choices=["completo", "resumen", "descarte"],
        default=None,
    )
    ap.add_argument(
        "--arm",
        dest="legacy_arm",
        choices=["naive", "compaction"],
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--kb-dir", default="kb")
    ap.add_argument("--tasks", default="config/session_tasks.example.json")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "tokenizer real usado para comprobar el presupuesto antes de cada "
            "petición; por defecto usa --model"
        ),
    )
    ap.add_argument(
        "--model-source",
        default=None,
        help="checkpoint fuente; útil si --model es un nombre servido",
    )
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--expected-tasks", type=int, default=None)
    ap.add_argument("--request-timeout-s", type=float, default=600.0)
    ap.add_argument(
        "--compaction-threshold",
        type=int,
        default=4500,
        help=(
            "tokens del prompt medido que disparan compactación; es una "
            "regla operativa de la política, no un umbral óptimo"
        ),
    )
    ap.add_argument(
        "--pares-conservados",
        dest="pares_conservados",
        type=int,
        default=4,
        help=(
            "pares de turno completos y recientes que conserva el descarte. "
            "K=4 es un parámetro predeclarado, no optimizado; no garantiza "
            "igual ocupación que el resumen y debe reportarse la ocupación real"
        ),
    )
    ap.add_argument(
        "--segundos-reposo",
        dest="segundos_reposo",
        type=float,
        default=30.0,
        help=(
            "segundos de medición en reposo al iniciar la sesión. Queda en el "
            "manifiesto y permite informar una sensibilidad con línea base "
            "declarada en lugar de suponerla después"
        ),
    )
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument(
        "--expected-server-pgid",
        type=int,
        required=True,
        help=(
            "grupo de procesos de vLLM permitido en la GPU; cualquier PID "
            "de cómputo fuera de este grupo invalida la sesión"
        ),
    )
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument(
        "--post-warmup-settle-s",
        type=float,
        default=30.0,
        help=(
            "espera sin peticiones después del warmup y antes del baseline; "
            "evita usar el enfriamiento inmediato como potencia de reposo"
        ),
    )
    ap.add_argument(
        "--min-baseline-sample-fraction",
        type=float,
        default=0.9,
        help="fracción mínima de muestras NVML esperadas durante el reposo",
    )
    ap.add_argument(
        "--require-intervention",
        action="store_true",
        help="aborta si un brazo gestionado nunca cruza el umbral",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="permite sobrescribir la salida indicada; desactivado por defecto",
    )
    args = ap.parse_args()
    if args.arm is not None and args.legacy_arm is not None:
        ap.error("usa --brazo o --arm, no ambos")
    if args.legacy_arm is not None:
        args.arm = args.legacy_arm
    elif args.arm is None:
        args.arm = "completo"
    if args.rep <= 0:
        ap.error("--rep debe ser positivo")
    if args.expected_tasks is not None and args.expected_tasks <= 0:
        ap.error("--expected-tasks debe ser positivo")
    if args.request_timeout_s <= 0:
        ap.error("--request-timeout-s debe ser positivo")
    if args.warmup < 0:
        ap.error("--warmup no puede ser negativo")
    if args.post_warmup_settle_s < 0:
        ap.error("--post-warmup-settle-s no puede ser negativo")
    if args.device_index < 0:
        ap.error("--device-index no puede ser negativo")
    if not 0 < args.min_baseline_sample_fraction <= 1:
        ap.error("--min-baseline-sample-fraction debe estar en (0, 1]")

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
