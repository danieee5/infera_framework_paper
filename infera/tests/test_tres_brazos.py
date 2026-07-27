from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


INFERA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(INFERA_ROOT))

import analiza_tres_brazos as analyzer  # noqa: E402
import escribe_manifiesto_campana as campaign_manifest  # noqa: E402
import infera_session_runner as runner  # noqa: E402
from escribe_manifiesto_campana import flattened_schedule  # noqa: E402


def synthetic_trace(energy_j, power_w=None, count=10, pid=321):
    duration = (count - 1) * 0.1
    if power_w is None:
        power_w = energy_j / duration
    step = duration / (count - 1)
    return [
        {
            "timestamp_monotonic_s": round(index * step, 9),
            "timestamp_unix_s": round(1700000000.0 + index * step, 6),
            "power_w": power_w,
            "vram_used_mb": 1000.0,
            "vram_free_mb": 23000.0,
            "vram_total_mb": 24000.0,
            "temperature_c": 40.0,
            "graphics_clock_mhz": 2500,
            "sm_clock_mhz": 2500,
            "memory_clock_mhz": 10000,
            "gpu_utilization_pct": 50,
            "memory_utilization_pct": 10,
            "throttle_reasons_mask": 0,
            "throttle_reasons_active": [],
            "performance_state": 2,
            "process_query_available": True,
            "process_query_fraction": 1.0,
            "compute_pids": [pid],
            "compute_process_groups": {str(pid): 123},
            "foreign_compute_pids": [],
        }
        for index in range(count)
    ]


class FakeGuard:
    fingerprint = {
        "reference": "fake",
        "resolved_commit": "fake-commit",
        "tokenizer_class": "Fake",
        "vocab_size": 1,
        "backend_sha256": "a" * 64,
        "chat_template_sha256": "b" * 64,
        "special_tokens_sha256": "c" * 64,
    }

    def check(self, messages, requested_max_tokens, max_model_len):
        self.last_count = 100
        if self.last_count + requested_max_tokens > max_model_len:
            raise RuntimeError("budget")
        return self.last_count


class FakeMonitor:
    available = True

    def __init__(self, device_index=0, allowed_process_group=None):
        self.device_index = device_index
        self.allowed_process_group = allowed_process_group

    def telemetry_probe(self, require_complete=False):
        return {
            "complete": True,
            "missing": [],
            "sample": synthetic_trace(1.0)[0],
        }

    def medir_reposo(self, seconds):
        return {
            "disponible": True,
            "segundos": seconds,
            "muestras": 10,
            "muestras_esperadas": 10,
            "fraccion_muestras": 1.0,
            "potencia_reposo_media_w": 30.0,
            "potencia_reposo_mediana_w": 30.0,
            "potencia_reposo_sd_w": 0.0,
            "potencia_reposo_cv_pct": 0.0,
            "potencia_reposo_min_w": 30.0,
            "potencia_reposo_max_w": 30.0,
            "temperatura_media_c": 40.0,
            "temperatura_min_c": 40.0,
            "temperatura_max_c": 40.0,
            "process_query_available": True,
            "process_query_fraction": 1.0,
            "foreign_compute_pids": [],
            "trace": synthetic_trace(1.0, power_w=30.0),
        }

    def device_metadata(self):
        return {
            "disponible": True,
            "device_index": self.device_index,
            "name": "Fake GPU",
            "uuid": "GPU-fake",
            "driver_version": "fake",
            "nvml_version": "fake",
            "memory_total_bytes": 1,
            "power_limit_w": 1.0,
        }

    def cleanup(self):
        return None


def fake_measured_call(*args, **kwargs):
    messages = args[4]
    is_summary = (
        messages[-1]["role"] == "user"
        and "COMPACTAR" in messages[-1]["content"]
    )
    if is_summary:
        prompt_tokens = 5500
        energy = 5.0
        text = "HANDOFF válido"
    else:
        prompt_tokens = 5000 if len(messages) > 10 else 3000
        energy = 1.0
        text = "respuesta"
    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "preflight_prompt_tokens": prompt_tokens,
        "completion_tokens": 2,
        "requested_max_tokens": 512,
        "token_budget_ok": True,
        "wall_s": 0.2,
        "energy_j": energy,
        "duration_s": 1.0,
        "avg_power_w": 40.0,
        "peak_power_w": 50.0,
        "vram_peak_mb": 1000.0,
        "nvml_samples": 10,
        "nvml_available": True,
        "baseline_power_w": 30.0,
        "energy_above_baseline_j": max(0.0, energy - 1.0),
        "nvml_trace": synthetic_trace(energy),
        "nvml_sampling_fraction": 1.0,
        "nvml_buffer_coverage_s": 0.8,
        "finish_reason": "stop",
        "response_id": "chatcmpl-fake",
        "response_created": 1700000000,
        "response_model": "infera-awq",
    }


def fake_warmup_call(*args, **kwargs):
    return "OK", 100, 1, {
        "response_id": "warmup",
        "response_created": 1700000000,
        "response_model": "infera-awq",
        "finish_reason": "stop",
    }


class PairRetentionTests(unittest.TestCase):
    def test_retains_complete_most_recent_pairs_without_mutating_source(self):
        history = []
        for index in range(7):
            history.extend((
                {"role": "user", "content": f"q{index}"},
                {"role": "assistant", "content": f"a{index}"},
            ))
        original = json.loads(json.dumps(history))
        retained, ids, discarded = runner.retain_recent_complete_pairs(
            history,
            [f"T{index}" for index in range(7)],
            4,
        )
        self.assertEqual(discarded, 3)
        self.assertEqual(ids, ["T3", "T4", "T5", "T6"])
        self.assertEqual(retained[0]["content"], "q3")
        self.assertEqual(retained[-1]["content"], "a6")
        self.assertEqual(history, original)

    def test_rejects_odd_or_misordered_history(self):
        with self.assertRaises(ValueError):
            runner.retain_recent_complete_pairs(
                [{"role": "user", "content": "q"}],
                [],
                4,
            )
        with self.assertRaises(ValueError):
            runner.retain_recent_complete_pairs(
                [
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "q"},
                ],
                ["T0"],
                4,
            )


class FrozenScheduleTests(unittest.TestCase):
    def test_shell_schedule_matches_campaign_manifest_schedule(self):
        shell = (INFERA_ROOT / "run_campana_tres_brazos.sh").read_text(
            encoding="utf-8"
        )
        observed = []
        for match in re.finditer(
            r'^bloque (AWQ|FP16)\s+"\$(AWQ_MODEL|FP16_MODEL)"\s+'
            r"([123])\s+(.+)$",
            shell,
            flags=re.MULTILINE,
        ):
            quant, model_variable, rep, arms_text = match.groups()
            self.assertEqual(model_variable, f"{quant}_MODEL")
            arms = arms_text.split()
            self.assertEqual(sorted(arms), ["completo", "descarte", "resumen"])
            observed.extend(
                (quant, arm, int(rep)) for arm in arms
            )
        expected = [
            (item["quant"], item["arm"], item["rep"])
            for item in flattened_schedule()
        ]
        self.assertEqual(observed, expected)


class TokenBudgetGuardTests(unittest.TestCase):
    def test_enforces_prompt_plus_requested_max_at_8192(self):
        class Backend:
            @staticmethod
            def to_str():
                return "{}"

        class Tokenizer:
            chat_template = "{{ messages }}"
            backend_tokenizer = Backend()
            special_tokens_map = {}
            init_kwargs = {}

            @staticmethod
            def apply_chat_template(*args, **kwargs):
                return list(range(8000))

            @staticmethod
            def __len__():
                return 1

        guard = runner.TokenBudgetGuard(Tokenizer(), "fake")
        self.assertEqual(guard.check([], 192, 8192), 8000)
        with self.assertRaisesRegex(RuntimeError, "presupuesto excedido"):
            guard.check([], 193, 8192)


class RunnerMockIntegrationTests(unittest.TestCase):
    def test_all_arms_publish_exactly_29_tasks_and_valid_mechanism_costs(self):
        tasks = {
            "decoding": {
                "temperature": 0.0,
                "max_tokens": 512,
                "seed": 42,
            },
            "tasks": [
                {
                    "id": f"T{index:02}",
                    "type": "FACT",
                    "prompt": f"pregunta {index}",
                    "verify": {},
                }
                for index in range(29)
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            tasks_path = temporary_path / "tasks.json"
            tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
            for arm in ("completo", "resumen", "descarte"):
                with self.subTest(arm=arm):
                    out = temporary_path / f"run_{arm}.jsonl"
                    args = SimpleNamespace(
                        out=str(out),
                        overwrite=False,
                        kb_dir=str(INFERA_ROOT / "kb"),
                        tasks=str(tasks_path),
                        expected_tasks=29,
                        max_model_len=8192,
                        compaction_threshold=4500,
                        pares_conservados=4,
                        segundos_reposo=30.0,
                        tokenizer="fake",
                        model="served",
                        model_source="source",
                        quant="AWQ",
                        arm=arm,
                        rep=1,
                        request_timeout_s=600.0,
                        warmup=1,
                        post_warmup_settle_s=0.0,
                        require_intervention=arm != "completo",
                        device_index=0,
                        expected_server_pgid=123,
                        min_baseline_sample_fraction=0.9,
                        vllm_url="http://fake",
                    )
                    with (
                        mock.patch.object(
                            runner.TokenBudgetGuard,
                            "from_reference",
                            return_value=FakeGuard(),
                        ),
                        mock.patch.object(
                            runner,
                            "GPUPowerMonitor",
                            FakeMonitor,
                        ),
                        mock.patch.object(
                            runner,
                            "measured_call",
                            side_effect=fake_measured_call,
                        ),
                        mock.patch.object(
                            runner,
                            "call_vllm",
                            side_effect=fake_warmup_call,
                        ),
                    ):
                        runner.run_session(args)
                    rows = analyzer.read_jsonl(out)
                    self.assertEqual(len(analyzer.tareas(rows)), 29)
                    self.assertFalse(Path(str(out) + ".partial").exists())
                    manifest = json.loads(
                        Path(str(out) + ".manifiesto.json").read_text(
                            encoding="utf-8"
                        )
                    )
                    self.assertEqual(manifest["status"], "complete")
                    events = analyzer.eventos(rows)
                    if arm == "completo":
                        self.assertEqual(events, [])
                    elif arm == "resumen":
                        self.assertGreater(len(events), 0)
                        self.assertTrue(
                            all(event["mechanism_energy_j"] > 0 for event in events)
                        )
                    else:
                        self.assertGreater(len(events), 0)
                        for event in events:
                            self.assertEqual(event["energy_j"], 0.0)
                            self.assertEqual(event["mechanism_energy_j"], 0.0)
                            self.assertEqual(event["pares_conservados"], 4)

    def test_aborts_if_discard_does_not_reset_below_threshold(self):
        tasks = {
            "decoding": {
                "temperature": 0.0,
                "max_tokens": 512,
                "seed": 42,
            },
            "tasks": [
                {
                    "id": f"T{index:02}",
                    "type": "FACT",
                    "prompt": f"pregunta {index}",
                    "verify": {},
                }
                for index in range(29)
            ],
        }
        task_calls = 0

        def measured_without_reset(*args, **kwargs):
            nonlocal task_calls
            result = fake_measured_call(*args, **kwargs)
            messages = args[4]
            is_summary = (
                messages[-1]["role"] == "user"
                and "COMPACTAR" in messages[-1]["content"]
            )
            if not is_summary:
                task_calls += 1
                prompt = 5000 if task_calls >= 6 else 3000
                result["prompt_tokens"] = prompt
                result["preflight_prompt_tokens"] = prompt
            return result

        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            tasks_path = temporary_path / "tasks.json"
            tasks_path.write_text(json.dumps(tasks), encoding="utf-8")
            out = temporary_path / "run_discard.jsonl"
            args = SimpleNamespace(
                out=str(out),
                overwrite=False,
                kb_dir=str(INFERA_ROOT / "kb"),
                tasks=str(tasks_path),
                expected_tasks=29,
                max_model_len=8192,
                compaction_threshold=4500,
                pares_conservados=4,
                segundos_reposo=30.0,
                tokenizer="fake",
                model="served",
                model_source="source",
                quant="AWQ",
                arm="descarte",
                rep=1,
                request_timeout_s=600.0,
                warmup=1,
                post_warmup_settle_s=0.0,
                require_intervention=True,
                device_index=0,
                expected_server_pgid=123,
                min_baseline_sample_fraction=0.9,
                vllm_url="http://fake",
            )
            with (
                mock.patch.object(
                    runner.TokenBudgetGuard,
                    "from_reference",
                    return_value=FakeGuard(),
                ),
                mock.patch.object(runner, "GPUPowerMonitor", FakeMonitor),
                mock.patch.object(
                    runner,
                    "measured_call",
                    side_effect=measured_without_reset,
                ),
                mock.patch.object(
                    runner,
                    "call_vllm",
                    side_effect=fake_warmup_call,
                ),
                self.assertRaisesRegex(
                    RuntimeError,
                    "evitar intervenir en cada turno",
                ),
            ):
                runner.run_session(args)
            self.assertFalse(out.exists())
            self.assertTrue(Path(str(out) + ".partial").is_file())
            manifest = json.loads(
                Path(str(out) + ".manifiesto.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["status"], "failed")


def synthetic_task_row(quant, arm, rep, index, cumulative):
    trace = synthetic_trace(1.0)
    power = float(trace[0]["power_w"])
    duration = (
        trace[-1]["timestamp_monotonic_s"]
        - trace[0]["timestamp_monotonic_s"]
    )
    expected_samples = int(duration / 0.1) + 1
    return {
        "schema_version": 3,
        "run_id": f"{quant}_{arm}_rep{rep}",
        "quant": quant,
        "arm": arm,
        "rep": rep,
        "task_index": index,
        "task_id": f"T{index:02}",
        "task_type": "FACT",
        "accumulated_prompt_tokens": 100 + index,
        "preflight_prompt_tokens": 100 + index,
        "completion_tokens": 2,
        "requested_max_tokens": 10,
        "max_model_len": 8192,
        "threshold_tokens": 4500,
        "token_budget_ok": True,
        "energy_j": 1.0,
        "mechanism_energy_j": 0.0,
        "energy_above_baseline_j": 0.0,
        "baseline_power_w": 30.0,
        "quality": 1.0,
        "quality_subscores": {},
        "quality_is_programmatic": True,
        "prompt_text": f"pregunta {index}",
        "response_text": f"respuesta {index}",
        "finish_reason": "stop",
        "vllm_response_id": f"chatcmpl-{quant}-{arm}-{rep}-{index}",
        "vllm_response_created": 1700000000,
        "vllm_response_model": f"infera-{quant.lower()}",
        "cumulative_energy_j": cumulative,
        "cumulative_mechanism_energy_j": 0.0,
        "status": "ok",
        "is_mechanism_event": False,
        "is_compaction": False,
        "es_descarte": False,
        "nvml_samples": 10,
        "nvml_available": True,
        "nvml_trace": trace,
        "nvml_sampling_fraction": round(10 / expected_samples, 4),
        "nvml_buffer_coverage_s": round(duration - 0.1, 4),
        "duration_s": round(duration, 4),
        "request_wall_s": 0.1,
        "avg_power_w": round(power, 2),
        "peak_power_w": round(power, 2),
        "vram_peak_mb": 1000.0,
    }


def write_synthetic_campaign(root: Path):
    raw_dir = root / "raw"
    raw_dir.mkdir()
    scenario = {
        "decoding": {"temperature": 0.0, "max_tokens": 10, "seed": 42},
        "tasks": [
            {
                "id": f"T{index:02}",
                "type": "FACT",
                "prompt": f"pregunta {index}",
                "verify": {},
            }
            for index in range(29)
        ],
    }
    scenario_path = root / "tasks.json"
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    scenario_hash = analyzer.sha256_file(scenario_path)
    kb_dir = INFERA_ROOT / "kb"
    kb_hash = analyzer.sha256_text(runner.build_fixed_context(str(kb_dir)))
    runner_hash = analyzer.sha256_file(INFERA_ROOT / "infera_session_runner.py")
    model_paths = {}
    for quant in ("AWQ", "FP16"):
        model_path = root / "models" / quant.lower()
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text(
            json.dumps({"quant": quant}),
            encoding="utf-8",
        )
        (model_path / "weights.safetensors").write_bytes(
            f"fake-{quant}".encode("utf-8")
        )
        model_paths[quant] = model_path
    code_names = (
        "infera_session_runner.py",
        "gpu_power_monitor.py",
        "analiza_tres_brazos.py",
        "preflight_campana_tres_brazos.py",
        "escribe_manifiesto_campana.py",
        "run_campana_tres_brazos.sh",
    )
    preflight = {
        "status": "ready_without_gpu_inference",
        "task_count": 29,
        "tasks": str(scenario_path.resolve()),
        "tasks_sha256": scenario_hash,
        "task_ids_sha256": analyzer.sha256_text(
            json.dumps(
                [task["id"] for task in scenario["tasks"]],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        ),
        "kb_dir": str(kb_dir.resolve()),
        "kb_sha256": kb_hash,
        "threshold_tokens": 4500,
        "pairs_kept": 4,
        "max_model_len": 8192,
        "baseline_seconds": 30.0,
        "post_warmup_settle_seconds": 30.0,
        "warmup_count": 5,
        "cooldown_seconds": 120.0,
        "request_timeout_seconds": 600.0,
        "server_start_attempts": 2,
        "requested_max_tokens": 10,
        "temperature": 0.0,
        "seed": 42,
        "prefix_caching_enabled": False,
        "runtime_guards": {
            "raw_nvml_trace_required": True,
            "thermal_clock_utilization_trace_required": True,
            "continuous_compute_process_check": True,
        },
        "tokenizers": {
            quant: {
                **FakeGuard.fingerprint,
                "reference": str(model_paths[quant]),
            }
            for quant in ("AWQ", "FP16")
        },
        "models": {
            quant: {
                "reference": str(model_paths[quant]),
                "core_config": {"architecture": "fake"},
                "quantization_config": (
                    {"quant_method": "awq"} if quant == "AWQ" else None
                ),
                "inventory": {
                    "inventory_sha256": quant.lower() * 32,
                    "files": [
                        {
                            "path": "weights.safetensors",
                            "bytes": 8,
                            "sha256": "d" * 64,
                        }
                    ],
                },
            }
            for quant in ("AWQ", "FP16")
        },
        "code_sha256": {
            name: analyzer.sha256_file(INFERA_ROOT / name)
            for name in code_names
        },
    }
    preflight_path = root / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    campaign = {
        "schema_version": 1,
        "campaign": "tres_brazos",
        "status": "running",
        "started_utc": "2026-07-27T00:00:00+00:00",
        "schedule": flattened_schedule(),
        "preflight": str(preflight_path.resolve()),
        "preflight_sha256": analyzer.sha256_file(preflight_path),
        "gpu": {
            "uuid": "GPU-fake",
            "telemetry_probe": {"complete": True, "missing": []},
        },
    }
    (root / "manifiesto_campana.json").write_text(
        json.dumps(campaign),
        encoding="utf-8",
    )
    for position, item in enumerate(flattened_schedule(), start=1):
        quant, arm, rep = item["quant"], item["arm"], item["rep"]
        rows = []
        cumulative = 0.0
        mechanism = 0.0
        history = []
        active_ids = []
        for index in range(29):
            cumulative += 1.0
            row = synthetic_task_row(
                quant,
                arm,
                rep,
                index,
                cumulative,
            )
            row["cumulative_mechanism_energy_j"] = mechanism
            rows.append(row)
            history.extend((
                {"role": "user", "content": row["prompt_text"]},
                {"role": "assistant", "content": row["response_text"]},
            ))
            active_ids.append(row["task_id"])
            if index != 5 or arm == "completo":
                continue
            if arm == "resumen":
                cumulative += 2.0
                mechanism += 2.0
                summary_trace = synthetic_trace(2.0)
                summary_power = float(summary_trace[0]["power_w"])
                summary_duration = (
                    summary_trace[-1]["timestamp_monotonic_s"]
                    - summary_trace[0]["timestamp_monotonic_s"]
                )
                summary_expected = int(summary_duration / 0.1) + 1
                rows.append({
                    "schema_version": 3,
                    "run_id": f"{quant}_{arm}_rep{rep}",
                    "quant": quant,
                    "arm": arm,
                    "rep": rep,
                    "task_type": "COMPACTION",
                    "accumulated_prompt_tokens": 200,
                    "preflight_prompt_tokens": 200,
                    "completion_tokens": 5,
                    "requested_max_tokens": 10,
                    "max_model_len": 8192,
                    "threshold_tokens": 4500,
                    "token_budget_ok": True,
                    "energy_j": 2.0,
                    "mechanism_energy_j": 2.0,
                    "energy_above_baseline_j": 0.0,
                    "baseline_power_w": 30.0,
                    "cumulative_energy_j": cumulative,
                    "cumulative_mechanism_energy_j": mechanism,
                    "status": "ok",
                    "is_mechanism_event": True,
                    "is_compaction": True,
                    "es_descarte": False,
                    "nvml_samples": 10,
                    "nvml_available": True,
                    "nvml_trace": summary_trace,
                    "nvml_sampling_fraction": round(
                        10 / summary_expected, 4
                    ),
                    "nvml_buffer_coverage_s": round(
                        summary_duration - 0.1, 4
                    ),
                    "duration_s": round(summary_duration, 4),
                    "request_wall_s": 0.1,
                    "avg_power_w": round(summary_power, 2),
                    "peak_power_w": round(summary_power, 2),
                    "vram_peak_mb": 1000.0,
                    "finish_reason": "stop",
                    "vllm_response_id": "chatcmpl-summary",
                    "vllm_response_created": 1700000000,
                    "vllm_response_model": f"infera-{quant.lower()}",
                    "intervention_index": 1,
                    "trigger_prompt_tokens": 5000,
                })
            else:
                before_hash = analyzer.sha256_text(
                    analyzer.canonical_json(history)
                )
                discarded = len(active_ids) - 4
                history = history[discarded * 2:]
                active_ids = active_ids[-4:]
                rows.append({
                    "schema_version": 3,
                    "run_id": f"{quant}_{arm}_rep{rep}",
                    "quant": quant,
                    "arm": arm,
                    "rep": rep,
                    "task_type": "DESCARTE",
                    "accumulated_prompt_tokens": 0,
                    "preflight_prompt_tokens": None,
                    "completion_tokens": 0,
                    "requested_max_tokens": 0,
                    "max_model_len": 8192,
                    "threshold_tokens": 4500,
                    "token_budget_ok": True,
                    "energy_j": 0.0,
                    "mechanism_energy_j": 0.0,
                    "energy_above_baseline_j": 0.0,
                    "baseline_power_w": 30.0,
                    "cumulative_energy_j": cumulative,
                    "cumulative_mechanism_energy_j": mechanism,
                    "status": "ok",
                    "is_mechanism_event": True,
                    "is_compaction": False,
                    "es_descarte": True,
                    "nvml_samples": None,
                    "nvml_available": None,
                    "nvml_trace": [],
                    "nvml_sampling_fraction": None,
                    "nvml_buffer_coverage_s": None,
                    "finish_reason": None,
                    "vllm_response_id": None,
                    "vllm_response_created": None,
                    "vllm_response_model": None,
                    "pares_descartados": discarded,
                    "pares_conservados": 4,
                    "tareas_conservadas_ids": list(active_ids),
                    "historial_antes_sha256": before_hash,
                    "historial_conservado_sha256": analyzer.sha256_text(
                        analyzer.canonical_json(history)
                    ),
                    "intervention_index": 1,
                    "context_tokens_before_intervention": 5000,
                })
        path = raw_dir / f"run_{quant}_{arm}_rep{rep}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 3,
            "status": "complete",
            "quant": quant,
            "brazo": arm,
            "rep": rep,
            "umbral_tokens": 4500,
            "escenario_sha256": scenario_hash,
            "kb_sha256": kb_hash,
            "runner_sha256": runner_hash,
            "tokenizer": {
                **FakeGuard.fingerprint,
                "reference": str(model_paths[quant]),
            },
            "modelo_fuente": str(model_paths[quant]),
            "modelo_servido": f"infera-{quant.lower()}",
            "expected_server_pgid": 123,
            "gpu": {"uuid": "GPU-fake"},
            "expected_tasks": 29,
            "max_model_len": 8192,
            "max_tokens": 10,
            "temperatura": 0.0,
            "semilla": 42,
            "post_warmup_settle_s": 30.0,
            "calentamiento_solicitado": 5,
            "request_timeout_s": 600.0,
            "calentamiento_completado": 5,
            "warmup_records": [
                {
                    "index": warmup_index + 1,
                    "prompt_tokens": 100,
                    "completion_tokens": 1,
                    "finish_reason": "stop",
                }
                for warmup_index in range(5)
            ],
            "iniciado_utc": f"2026-07-27T00:00:{position:02}+00:00",
            "telemetry_probe": {"complete": True, "missing": []},
            "reposo": {
                "disponible": True,
                "segundos": 30.0,
                "muestras": 300,
                "muestras_esperadas": 300,
                "fraccion_muestras": 1.0,
                "potencia_reposo_media_w": 30.0,
                "potencia_reposo_mediana_w": 30.0,
                "process_query_available": True,
                "process_query_fraction": 1.0,
                "foreign_compute_pids": [],
                "trace": synthetic_trace(
                    897.0,
                    power_w=30.0,
                    count=300,
                ),
            },
            "energia_total_j": cumulative,
            "energia_mecanismo_j": mechanism,
            "tareas_medidas": 29,
            "intervenciones": 0 if arm == "completo" else 1,
            "raw_sha256": analyzer.sha256_file(path),
        }
        Path(str(path) + ".manifiesto.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
    return raw_dir


class AnalyzerIntegrationTests(unittest.TestCase):
    def test_accepts_only_complete_18_session_campaign(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_dir = write_synthetic_campaign(Path(temporary))
            sessions, manifests = analyzer.cargar(raw_dir, 29, 8192, 4)
            self.assertEqual(len(sessions), 18)
            self.assertEqual(len(manifests), 18)
            session_rows = analyzer.por_sesion(sessions)
            discard_rows = [
                row for row in session_rows if row["brazo"] == "descarte"
            ]
            self.assertTrue(
                all(row["tokens_entrada_de_peticiones"] > 0 for row in discard_rows)
            )
            output = Path(temporary) / "analysis"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "analiza_tres_brazos.py",
                        "--crudos",
                        str(raw_dir),
                        "--salida",
                        str(output),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(analyzer.main(), 0)
            self.assertTrue((output / "informe.md").is_file())
            self.assertTrue((output / "manifiesto_analisis.json").is_file())

    def test_reanalysis_accepts_complete_or_failed_campaign_without_gpu(self):
        for status in ("complete", "failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                raw_dir = write_synthetic_campaign(root)
                campaign_path = root / "manifiesto_campana.json"
                campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
                campaign["status"] = status
                campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
                sessions, manifests = analyzer.cargar(raw_dir, 29, 8192, 4)
                self.assertEqual(len(sessions), 18)
                self.assertEqual(len(manifests), 18)

    def test_reanalysis_records_analyzer_code_drift_instead_of_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = write_synthetic_campaign(root)
            preflight_path = root / "preflight.json"
            preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
            preflight["code_sha256"]["analiza_tres_brazos.py"] = "0" * 64
            preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
            campaign_path = root / "manifiesto_campana.json"
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            campaign["preflight_sha256"] = analyzer.sha256_file(preflight_path)
            campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
            output = root / "analysis_with_drift"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "analiza_tres_brazos.py",
                        "--crudos",
                        str(raw_dir),
                        "--salida",
                        str(output),
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(analyzer.main(), 0)
            manifest = json.loads(
                (output / "manifiesto_analisis.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                manifest["code_drift"][0]["name"],
                "analiza_tres_brazos.py",
            )

    def test_reanalysis_script_recovers_failed_campaign_without_vllm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_synthetic_campaign(root)
            campaign_path = root / "manifiesto_campana.json"
            campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
            campaign["status"] = "failed"
            campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
            output = root / "recovered_analysis"
            subprocess.run(
                [
                    str(INFERA_ROOT / "reanaliza_campana_tres_brazos.sh"),
                    str(root),
                    str(output),
                ],
                cwd=INFERA_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            final = json.loads(campaign_path.read_text(encoding="utf-8"))
            self.assertEqual(final["status"], "complete")
            self.assertTrue((output / "informe.md").is_file())

    def test_rejects_foreign_gpu_process_in_raw_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = write_synthetic_campaign(root)
            raw_path = raw_dir / "run_AWQ_completo_rep1.jsonl"
            rows = analyzer.read_jsonl(raw_path)
            rows[0]["nvml_trace"][3]["compute_pids"].append(999)
            rows[0]["nvml_trace"][3]["compute_process_groups"]["999"] = 999
            rows[0]["nvml_trace"][3]["foreign_compute_pids"] = [999]
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = Path(str(raw_path) + ".manifiesto.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["raw_sha256"] = analyzer.sha256_file(raw_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "procesos GPU ajenos"):
                analyzer.cargar(raw_dir, 29, 8192, 4)

    def test_rejects_energy_that_does_not_reintegrate_from_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = write_synthetic_campaign(root)
            raw_path = raw_dir / "run_AWQ_completo_rep1.jsonl"
            rows = analyzer.read_jsonl(raw_path)
            rows[0]["nvml_trace"][5]["power_w"] = 400.0
            raw_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest_path = Path(str(raw_path) + ".manifiesto.json")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["raw_sha256"] = analyzer.sha256_file(raw_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no se reproduce"):
                analyzer.cargar(raw_dir, 29, 8192, 4)

    def test_rejects_preflight_modified_after_campaign_start(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = write_synthetic_campaign(root)
            with (root / "preflight.json").open("a", encoding="utf-8") as fh:
                fh.write("\n")
            with self.assertRaisesRegex(ValueError, "preflight cambió"):
                analyzer.cargar(raw_dir, 29, 8192, 4)

    def test_rejects_truncated_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            path.write_text('{"a": 1}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "truncamiento"):
                analyzer.read_jsonl(path)


class CampaignManifestTests(unittest.TestCase):
    def test_records_preflight_failure_even_without_preflight_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifiesto_campana.json"
            with (
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "escribe_manifiesto_campana.py",
                        "--manifest",
                        str(manifest),
                        "--preflight",
                        str(root / "preflight.json"),
                        "--raw-dir",
                        str(root / "raw"),
                        "--analysis-dir",
                        str(root / "analysis"),
                        "--status",
                        "preflight_failed",
                        "--exit-code",
                        "2",
                    ],
                ),
                redirect_stdout(StringIO()),
            ):
                self.assertEqual(campaign_manifest.main(), 0)
            document = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(document["status"], "preflight_failed")
            self.assertFalse(document["preflight_exists"])


if __name__ == "__main__":
    unittest.main()
