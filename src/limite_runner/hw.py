"""Hardware detection and backend/quantization auto-selection.

Detection runs at startup (``limite doctor``) and drives the default choice of
backend and quantization; every decision can be overridden from config.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass, field

from .config import DEFAULT_QUANT, LITERT_KV_GIB, RunConfig

log = logging.getLogger(__name__)


@dataclass
class Hardware:
    os_name: str = ""
    os_release: str = ""
    arch: str = ""
    cpu_count: int = 0
    ram_total_gib: float = 0.0
    ram_available_gib: float = 0.0
    gpu_name: str | None = None
    vram_total_gib: float | None = None
    cuda_version: str | None = None
    metal_available: bool = False
    wsl: bool = False
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "os": f"{self.os_name} {self.os_release}",
            "arch": self.arch,
            "cpu_count": self.cpu_count,
            "ram_total_gib": round(self.ram_total_gib, 2),
            "ram_available_gib": round(self.ram_available_gib, 2),
            "gpu": self.gpu_name,
            "vram_total_gib": self.vram_total_gib,
            "cuda": self.cuda_version,
            "metal": self.metal_available,
            "wsl": self.wsl,
            "notes": self.notes,
        }


def detect_hardware() -> Hardware:
    hw = Hardware(
        os_name=platform.system(),
        os_release=platform.release(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 0,
    )
    try:
        import psutil

        vm = psutil.virtual_memory()
        hw.ram_total_gib = vm.total / 2**30
        hw.ram_available_gib = vm.available / 2**30
    except Exception as exc:  # pragma: no cover - psutil always bundled
        hw.notes.append(f"psutil unavailable: {exc}")

    if hw.os_name == "Darwin":
        hw.metal_available = True  # Apple Silicon / AMD: Metal is always present
    if hw.os_name == "Linux":
        hw.wsl = "microsoft" in platform.release().lower() or bool(
            os.environ.get("WSL_DISTRO_NAME")
        )

    # NVIDIA via nvidia-smi (works on Windows, Linux and WSL2)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if out.returncode == 0 and out.stdout.strip():
                line = out.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in line.split(",")]
                hw.gpu_name = parts[0] or None
                if len(parts) > 1 and parts[1].isdigit():
                    hw.vram_total_gib = int(parts[1]) / 1024
                if len(parts) > 2:
                    hw.notes.append(f"NVIDIA driver {parts[2]}")
        except Exception as exc:
            hw.notes.append(f"nvidia-smi failed: {exc}")
    if shutil.which("nvcc"):
        try:
            out = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=10
            )
            for line in out.stdout.splitlines():
                if "release" in line:
                    hw.cuda_version = line.split("release")[-1].strip().split(",")[0]
                    break
        except Exception:
            pass
    return hw


def estimate_ram_needed(backend: str, ctx: int) -> float:
    """Rough RAM/VRAM requirement (GiB) before loading a model."""
    if backend == "litert":
        # bundle ~0.8 GiB + KV cache + runtime overhead
        return 0.8 + LITERT_KV_GIB.get(ctx, 1.5) + 1.0
    if backend == "llamacpp":
        return 1.3 + 0.15  # Q8_0 weights + KV at 16K + overhead (scaled below)
    return 2.5  # transformers BF16 weights + activations


def check_memory_budget(hw: Hardware, backend: str, ctx: int) -> str | None:
    """Return an error message if the machine cannot host ``backend`` at ``ctx``."""
    need = estimate_ram_needed(backend, ctx)
    if backend == "llamacpp":
        need = 1.3 + (ctx / 16384) * 0.15 + 0.5
    pool = hw.vram_total_gib if (backend == "llamacpp" and hw.gpu_name) else hw.ram_available_gib
    if pool is not None and pool < need:
        return (
            f"insufficient {'VRAM' if backend == 'llamacpp' and hw.gpu_name else 'RAM'} "
            f"for backend={backend} ctx={ctx}: need ~{need:.1f} GiB, available {pool:.1f} GiB. "
            f"LiteRT KV cache alone is {LITERT_KV_GIB.get(ctx, '?')} GiB at ctx={ctx}."
        )
    return None


def choose_backend(hw: Hardware, cfg: RunConfig) -> str:
    """Pick the best backend for this machine unless explicitly overridden."""
    if cfg.backend:
        return cfg.backend
    # Preference: llamacpp (primary) if a build/model is plausible, else litert
    # on CPU-only machines, else transformers when GPU is present but no build.
    if hw.gpu_name or hw.metal_available:
        return "llamacpp"
    return "litert"


def choose_quant(hw: Hardware, cfg: RunConfig) -> str:
    if cfg.quant:
        return cfg.quant
    return DEFAULT_QUANT
