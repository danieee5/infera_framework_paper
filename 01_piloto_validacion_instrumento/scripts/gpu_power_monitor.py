"""
gpu_power_monitor.py
GPU energy and VRAM monitor via NVML hardware counters.

MEASUREMENT PROTOCOL:
  - 500 ms pre/post-inference buffer (Husom et al. 2026, MELODI Table 1 — 100% CCR)
  - 100 ms sampling interval / 10 Hz (MELODI default)
  - Trapezoidal integration of power-time profile → Joules
  - VRAM: used / free / total + peak tracking per inference

WHY 500 ms BUFFER:
  Husom et al. (2026) tested multiple buffer values experimentally:
  - 200 ms → 0% capture completeness rate (DISCARDED)
  - 400 ms → 48% CCR (INSUFFICIENT)
  - 500 ms → 100% CCR (ADOPTED)
  This is the minimum buffer that guarantees full capture of the
  power profile on both sides of the inference call.

TRAPEZOIDAL INTEGRATION:
  energy_J = Σ (P[i] + P[i-1]) / 2 × Δt[i]
  This is the standard numerical method for computing area under
  a power-time curve when samples are evenly spaced at ~100 ms.
"""

import threading
import time
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class GPUSample:
    timestamp: float       # wall-clock time (time.time())
    power_w: float         # GPU power draw in Watts
    vram_used_mb: float    # VRAM used in MiB
    vram_free_mb: float    # VRAM free in MiB
    vram_total_mb: float   # VRAM total capacity in MiB


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

    # DO NOT reduce these values — 500 ms is empirically validated minimum (MELODI Table 1)
    BUFFER_MS = 500
    SAMPLING_MS = 100   # 10 Hz — MELODI default sampling rate

    def __init__(self, device_index: int = 0):
        self.device_index = device_index
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

    def _sample_gpu(self) -> Optional[GPUSample]:
        """Take one NVML sample. Returns None on any error."""
        if self._handle is None:
            return None
        try:
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            return GPUSample(
                timestamp=time.time(),
                power_w=power_mw / 1000.0,
                vram_used_mb=mem.used / (1024 ** 2),
                vram_free_mb=mem.free / (1024 ** 2),
                vram_total_mb=mem.total / (1024 ** 2),
            )
        except Exception:
            return None

    def _monitor_loop(self):
        """Background thread: sample GPU every SAMPLING_MS ms until stopped."""
        while self._monitoring:
            s = self._sample_gpu()
            if s is not None:
                self._samples.append(s)
            time.sleep(self.SAMPLING_MS / 1000.0)

    def start_monitoring(self):
        """
        Start the background sampling thread.
        Blocks for BUFFER_MS (500 ms) before returning — this is the
        pre-inference buffer required for 100% capture completeness rate.
        Call this IMMEDIATELY BEFORE the inference call.
        """
        self._samples = []
        self._monitoring = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        time.sleep(self.BUFFER_MS / 1000.0)  # pre-inference buffer

    def stop_monitoring(self) -> InferenceEnergyResult:
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
            )

        timestamps = [s.timestamp for s in self._samples]
        powers     = [s.power_w   for s in self._samples]
        vrams      = [s.vram_used_mb for s in self._samples]

        # Trapezoidal integration: energy_J = Σ (P[i]+P[i-1])/2 × Δt[i]
        energy_j = sum(
            (powers[i] + powers[i - 1]) / 2.0 * (timestamps[i] - timestamps[i - 1])
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
        )

    def cleanup(self):
        """Release NVML resources. Call once when done with all benchmark runs."""
        if NVML_AVAILABLE and self._init_ok:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
