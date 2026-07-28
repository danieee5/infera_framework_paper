"""
gpu_power_monitor.py
GPU energy and VRAM monitor via NVML hardware counters.

MEASUREMENT PROTOCOL USED IN THIS REPOSITORY:
  - 500 ms pre/post-inference buffer
  - 100 ms sampling interval / 10 Hz (MELODI default)
  - Trapezoidal integration of power-time profile → Joules
  - VRAM: used / free / total + peak tracking per inference

WHY 500 ms BUFFER:
  The value follows the measurement setup adopted for INFERA. It adds a
  margin around the HTTP request so the sampled window is not limited to the
  exact client-side call. It is not a universal guarantee: the required
  margin can change with hardware, driver, model and serving stack.

TRAPEZOIDAL INTEGRATION:
  energy_J = Σ (P[i] + P[i-1]) / 2 × Δt[i]
  This is the standard numerical method for computing area under
  a power-time curve when samples are evenly spaced at ~100 ms.
"""

import os
import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUSample:
    timestamp: float       # monotonic time (time.monotonic())
    timestamp_unix_s: float
    power_w: float         # GPU power draw in Watts
    vram_used_mb: float    # VRAM used in MiB
    vram_free_mb: float    # VRAM free in MiB
    vram_total_mb: float   # VRAM total capacity in MiB
    temperature_c: Optional[float]
    graphics_clock_mhz: Optional[int]
    sm_clock_mhz: Optional[int]
    memory_clock_mhz: Optional[int]
    gpu_utilization_pct: Optional[int]
    memory_utilization_pct: Optional[int]
    throttle_reasons_mask: Optional[int]
    throttle_reasons_active: list[str]
    performance_state: Optional[int]
    process_query_available: bool
    compute_pids: Optional[list[int]]
    compute_process_groups: Optional[dict[str, int]]
    foreign_compute_pids: list[int]

    def as_dict(self) -> dict[str, Any]:
        """Representación persistente con precisión suficiente para reintegrar."""
        return {
            "timestamp_monotonic_s": round(self.timestamp, 9),
            "timestamp_unix_s": round(self.timestamp_unix_s, 6),
            "power_w": round(self.power_w, 3),
            "vram_used_mb": round(self.vram_used_mb, 3),
            "vram_free_mb": round(self.vram_free_mb, 3),
            "vram_total_mb": round(self.vram_total_mb, 3),
            "temperature_c": self.temperature_c,
            "graphics_clock_mhz": self.graphics_clock_mhz,
            "sm_clock_mhz": self.sm_clock_mhz,
            "memory_clock_mhz": self.memory_clock_mhz,
            "gpu_utilization_pct": self.gpu_utilization_pct,
            "memory_utilization_pct": self.memory_utilization_pct,
            "throttle_reasons_mask": self.throttle_reasons_mask,
            "throttle_reasons_active": self.throttle_reasons_active,
            "performance_state": self.performance_state,
            "process_query_available": self.process_query_available,
            "compute_pids": self.compute_pids,
            "compute_process_groups": self.compute_process_groups,
            "foreign_compute_pids": self.foreign_compute_pids,
        }


@dataclass
class InferenceEnergyResult:
    energy_j: float           # Joules — trapezoidal integration over full window
    duration_s: float         # Elapsed time from first to last sample
    avg_power_w: float        # Mean power over sampling window
    peak_power_w: float       # Maximum instantaneous power observed
    vram_used_mb_start: float # VRAM at first sample (pre-inference baseline)
    vram_used_mb_peak: float  # Maximum VRAM observed during window
    vram_total_mb: float      # Total GPU VRAM capacity
    sample_count: int         # Total NVML samples collected
    nvml_available: bool      # False if pynvml not installed or init failed
    # Potencia mediana observada en el margen previo a la petición. Sirve como
    # referencia de reposo local para calcular una energía dinámica.
    baseline_power_w: float = 0.0
    # Energía por encima de esa referencia. Es un análisis de sensibilidad:
    # descuenta el consumo que la GPU habría tenido sin la petición.
    energy_above_baseline_j: float = 0.0
    # La traza es la evidencia primaria. Los agregados anteriores se pueden
    # recalcular a partir de estas muestras sin volver a encender la GPU.
    samples: list[dict[str, Any]] = field(default_factory=list)


class GPUPowerMonitor:
    """
    Measures GPU energy consumption and VRAM usage during a single inference call.

    The monitor runs a background thread that samples NVML at 100 ms intervals.
    start_monitoring() waits 500 ms (pre-buffer) before returning.
    stop_monitoring() waits 500 ms (post-buffer) before stopping the thread,
    then computes energy via trapezoidal integration and returns the result.

    Usage:
        monitor = GPUPowerMonitor(device_index=0)
        monitor.start_monitoring()           # includes 500 ms pre-buffer
        result = call_vllm(...)              # your inference call here
        energy = monitor.stop_monitoring()   # includes 500 ms post-buffer
        print(f"{energy.energy_j:.4f} J | peak VRAM: {energy.vram_used_mb_peak:.0f} MB")
        monitor.cleanup()                    # call once when done with all runs
    """

    # Values used by the executed INFERA experiment. Changing them creates a
    # different measurement configuration and must be documented.
    BUFFER_MS = 500
    SAMPLING_MS = 100   # 10 Hz — MELODI default sampling rate

    def __init__(
        self,
        device_index: int = 0,
        allowed_process_group: Optional[int] = None,
    ):
        self.device_index = device_index
        self.allowed_process_group = allowed_process_group
        self._samples: list[GPUSample] = []
        self._monitoring = False
        self._thread: Optional[threading.Thread] = None
        self._handle = None
        self._init_ok = False

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._init_ok = True
            except Exception as e:
                print(f"WARNING: NVML init failed: {e}")
                print("  Energy and VRAM measurements will be zeroed.")

    @staticmethod
    def _optional(callable_value, default=None):
        try:
            return callable_value()
        except Exception:
            return default

    def _clock(self, clock_type_name: str) -> Optional[int]:
        clock_type = getattr(pynvml, clock_type_name, None)
        if clock_type is None:
            return None
        return self._optional(
            lambda: int(pynvml.nvmlDeviceGetClockInfo(self._handle, clock_type))
        )

    def _throttle_reasons(self) -> tuple[Optional[int], list[str]]:
        getters = [
            getattr(
                pynvml,
                "nvmlDeviceGetCurrentClocksEventReasons",
                None,
            ),
            getattr(
                pynvml,
                "nvmlDeviceGetCurrentClocksThrottleReasons",
                None,
            ),
        ]
        mask = None
        for getter in getters:
            if getter is not None:
                mask = self._optional(lambda: int(getter(self._handle)))
            if mask is not None:
                break
        if mask is None:
            return None, []
        active = []
        prefixes = ("nvmlClocksEventReason", "nvmlClocksThrottleReason")
        excluded = ("All", "None")
        for name in dir(pynvml):
            if not name.startswith(prefixes) or name.endswith(excluded):
                continue
            value = getattr(pynvml, name)
            if isinstance(value, int) and value and mask & value:
                active.append(name)
        return mask, sorted(set(active))

    def _compute_processes(
        self,
    ) -> tuple[bool, Optional[list[int]], Optional[dict[str, int]], list[int]]:
        getter = getattr(pynvml, "nvmlDeviceGetComputeRunningProcesses", None)
        if getter is None:
            return False, None, None, []
        try:
            pids = sorted({int(process.pid) for process in getter(self._handle)})
        except Exception:
            return False, None, None, []
        process_groups = {}
        foreign = []
        for pid in pids:
            try:
                process_group = int(os.getpgid(pid))
            except (OSError, ProcessLookupError):
                # El proceso pudo terminar entre NVML y getpgid. Se conserva
                # el PID; la siguiente muestra o el control externo lo aclara.
                process_group = -1
            process_groups[str(pid)] = process_group
            if (
                self.allowed_process_group is not None
                and process_group != self.allowed_process_group
            ):
                foreign.append(pid)
        return True, pids, process_groups, foreign

    def _sample_gpu(self) -> Optional[GPUSample]:
        """Take one NVML sample. Core power/memory errors drop the sample."""
        if self._handle is None:
            return None
        try:
            timestamp = time.monotonic()
            timestamp_unix_s = time.time()
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            utilization = self._optional(
                lambda: pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            )
            throttle_mask, throttle_active = self._throttle_reasons()
            (
                process_query_available,
                compute_pids,
                process_groups,
                foreign_pids,
            ) = self._compute_processes()
            return GPUSample(
                timestamp=timestamp,
                timestamp_unix_s=timestamp_unix_s,
                power_w=power_mw / 1000.0,
                vram_used_mb=mem.used / (1024 ** 2),
                vram_free_mb=mem.free / (1024 ** 2),
                vram_total_mb=mem.total / (1024 ** 2),
                temperature_c=self._optional(
                    lambda: float(
                        pynvml.nvmlDeviceGetTemperature(
                            self._handle,
                            pynvml.NVML_TEMPERATURE_GPU,
                        )
                    )
                ),
                graphics_clock_mhz=self._clock("NVML_CLOCK_GRAPHICS"),
                sm_clock_mhz=self._clock("NVML_CLOCK_SM"),
                memory_clock_mhz=self._clock("NVML_CLOCK_MEM"),
                gpu_utilization_pct=(
                    int(utilization.gpu) if utilization is not None else None
                ),
                memory_utilization_pct=(
                    int(utilization.memory)
                    if utilization is not None
                    else None
                ),
                throttle_reasons_mask=throttle_mask,
                throttle_reasons_active=throttle_active,
                performance_state=self._optional(
                    lambda: int(
                        pynvml.nvmlDeviceGetPerformanceState(self._handle)
                    )
                ),
                process_query_available=process_query_available,
                compute_pids=compute_pids,
                compute_process_groups=process_groups,
                foreign_compute_pids=foreign_pids,
            )
        except Exception:
            return None

    @property
    def available(self) -> bool:
        """True cuando NVML se inicializó para el dispositivo solicitado."""
        return self._init_ok

    def device_metadata(self) -> dict:
        """Identidad estable del dispositivo usado por CUDA/NVML."""
        if not self._init_ok or self._handle is None:
            return {"disponible": False, "device_index": self.device_index}

        def text(value):
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)

        memory = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        metadata = {
            "disponible": True,
            "device_index": self.device_index,
            "name": text(pynvml.nvmlDeviceGetName(self._handle)),
            "uuid": text(pynvml.nvmlDeviceGetUUID(self._handle)),
            "driver_version": text(pynvml.nvmlSystemGetDriverVersion()),
            "nvml_version": text(pynvml.nvmlSystemGetNVMLVersion()),
            "memory_total_bytes": int(memory.total),
        }
        try:
            metadata["power_limit_w"] = round(
                pynvml.nvmlDeviceGetPowerManagementLimit(self._handle) / 1000.0,
                3,
            )
        except Exception:
            metadata["power_limit_w"] = None
        return metadata

    def telemetry_probe(self, require_complete: bool = False) -> dict[str, Any]:
        """Comprueba antes de medir que NVML expone toda la telemetría."""
        sample = self._sample_gpu()
        if sample is None:
            result = {
                "complete": False,
                "missing": ["core_power_or_memory"],
                "sample": None,
            }
        else:
            required = (
                "temperature_c",
                "graphics_clock_mhz",
                "sm_clock_mhz",
                "memory_clock_mhz",
                "gpu_utilization_pct",
                "memory_utilization_pct",
                "throttle_reasons_mask",
                "performance_state",
            )
            missing = [
                field_name
                for field_name in required
                if getattr(sample, field_name) is None
            ]
            if not sample.process_query_available:
                missing.append("compute_process_query")
            result = {
                "complete": not missing,
                "missing": missing,
                "sample": sample.as_dict(),
            }
        if require_complete and not result["complete"]:
            raise RuntimeError(
                "NVML no expone la telemetría requerida: "
                + ", ".join(result["missing"])
            )
        return result

    def _monitor_loop(self):
        """Background thread: sample GPU every SAMPLING_MS ms until stopped."""
        interval = self.SAMPLING_MS / 1000.0
        next_sample = time.monotonic()
        while self._monitoring:
            s = self._sample_gpu()
            if s is not None:
                self._samples.append(s)
            next_sample += interval
            remaining = next_sample - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            else:
                next_sample = time.monotonic()

    def start_monitoring(self):
        """
        Start the background sampling thread.
        Blocks for BUFFER_MS (500 ms) before returning. This is the margin
        adopted by the executed INFERA configuration.
        Call this IMMEDIATELY BEFORE the inference call.
        """
        self._samples = []
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        time.sleep(self.BUFFER_MS / 1000.0)  # pre-inference buffer

    def stop_monitoring(
        self,
        baseline_power_w: Optional[float] = None,
    ) -> InferenceEnergyResult:
        """
        Block for BUFFER_MS (500 ms) post-inference, then stop the thread
        and compute the energy result.
        Call this IMMEDIATELY AFTER the inference call returns.
        Returns an InferenceEnergyResult with all computed metrics.
        """
        time.sleep(self.BUFFER_MS / 1000.0)  # post-inference buffer
        self._monitoring = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        # Fallback result when NVML is not available
        if not NVML_AVAILABLE or not self._init_ok or len(self._samples) < 2:
            return InferenceEnergyResult(
                energy_j=0.0,
                duration_s=0.0,
                avg_power_w=0.0,
                peak_power_w=0.0,
                vram_used_mb_start=0.0,
                vram_used_mb_peak=0.0,
                vram_total_mb=0.0,
                sample_count=len(self._samples),
                nvml_available=self._init_ok,
                samples=[sample.as_dict() for sample in self._samples],
            )

        timestamps = [s.timestamp for s in self._samples]
        powers     = [s.power_w   for s in self._samples]
        vrams      = [s.vram_used_mb for s in self._samples]

        # Trapezoidal integration: energy_J = Σ (P[i]+P[i-1])/2 × Δt[i]
        energy_j = sum(
            (powers[i] + powers[i - 1]) / 2.0 * (timestamps[i] - timestamps[i - 1])
            for i in range(1, len(timestamps))
        )

        # La campaña de tres brazos pasa la mediana de los 30 s de reposo de la
        # sesión. El prebuffer local queda solo como fallback para otros usos.
        if baseline_power_w is None:
            n_pre = max(1, int(self.BUFFER_MS / self.SAMPLING_MS) - 1)
            baseline_w = (
                statistics.median(powers[:n_pre])
                if len(powers) > n_pre
                else powers[0]
            )
        else:
            baseline_w = float(baseline_power_w)
            if baseline_w <= 0:
                raise ValueError("baseline_power_w debe ser positivo")
        energy_above = sum(
            max(0.0, (powers[i] + powers[i - 1]) / 2.0 - baseline_w)
            * (timestamps[i] - timestamps[i - 1])
            for i in range(1, len(timestamps))
        )

        return InferenceEnergyResult(
            energy_j=energy_j,
            duration_s=timestamps[-1] - timestamps[0],
            avg_power_w=sum(powers) / len(powers),
            peak_power_w=max(powers),
            vram_used_mb_start=vrams[0],
            vram_used_mb_peak=max(vrams),
            vram_total_mb=self._samples[0].vram_total_mb,
            sample_count=len(self._samples),
            nvml_available=True,
            baseline_power_w=baseline_w,
            energy_above_baseline_j=energy_above,
            samples=[sample.as_dict() for sample in self._samples],
        )

    def medir_reposo(self, segundos: float = 30.0) -> dict:
        """Mide la potencia de la GPU en reposo, sin peticiones en curso.

        Se ejecuta una vez al inicio de cada sesión y queda registrada en el
        manifiesto. Permite informar una sensibilidad con línea base declarada
        en lugar de suponerla después.
        """
        if segundos <= 0:
            raise ValueError("segundos debe ser positivo")
        if not self._init_ok:
            return {"disponible": False}
        muestras: list[GPUSample] = []
        inicio = time.monotonic()
        fin = inicio + segundos
        interval = self.SAMPLING_MS / 1000.0
        next_sample = inicio
        while time.monotonic() < fin:
            s = self._sample_gpu()
            if s is not None:
                muestras.append(s)
            next_sample += interval
            remaining = min(fin, next_sample) - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            elif next_sample < time.monotonic():
                next_sample = time.monotonic()
        if not muestras:
            return {"disponible": False}
        esperado = max(1, int(segundos * 1000.0 / self.SAMPLING_MS))
        powers = [sample.power_w for sample in muestras]
        temperatures = [
            sample.temperature_c
            for sample in muestras
            if sample.temperature_c is not None
        ]
        media = statistics.mean(powers)
        sd = statistics.stdev(powers) if len(powers) > 1 else 0.0
        return {
            "disponible": True,
            "segundos": round(segundos, 2),
            "muestras": len(muestras),
            "muestras_esperadas": esperado,
            "fraccion_muestras": round(len(muestras) / esperado, 4),
            "potencia_reposo_media_w": round(media, 3),
            "potencia_reposo_mediana_w": round(statistics.median(powers), 3),
            "potencia_reposo_sd_w": round(sd, 3),
            "potencia_reposo_cv_pct": round(100.0 * sd / media, 3) if media else None,
            "potencia_reposo_min_w": round(min(powers), 3),
            "potencia_reposo_max_w": round(max(powers), 3),
            "temperatura_media_c": (
                round(statistics.mean(temperatures), 3)
                if temperatures else None
            ),
            "temperatura_min_c": min(temperatures) if temperatures else None,
            "temperatura_max_c": max(temperatures) if temperatures else None,
            "process_query_available": all(
                sample.process_query_available for sample in muestras
            ),
            "process_query_fraction": round(
                sum(sample.process_query_available for sample in muestras)
                / len(muestras),
                4,
            ),
            "foreign_compute_pids": sorted({
                pid
                for sample in muestras
                for pid in sample.foreign_compute_pids
            }),
            "trace": [sample.as_dict() for sample in muestras],
        }

    def cleanup(self):
        """Release NVML resources. Call once when done with all benchmark runs."""
        if NVML_AVAILABLE and self._init_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
