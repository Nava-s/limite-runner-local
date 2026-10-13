"""Hardware detection and auto-selection unit tests."""

from limite_runner.config import RunConfig
from limite_runner.hw import (
    Hardware,
    check_memory_budget,
    choose_backend,
    choose_quant,
    detect_hardware,
)


def test_detect_hardware_runs():
    hw = detect_hardware()
    assert hw.os_name in ("Windows", "Linux", "Darwin")
    assert hw.arch != ""
    assert hw.cpu_count >= 1
    assert hw.ram_total_gib > 0


def test_memory_budget_refuses_impossible_litert_ctx():
    hw = Hardware(ram_available_gib=2.0)
    err = check_memory_budget(hw, "litert", 65536)
    assert err is not None and "insufficient RAM" in err


def test_memory_budget_allows_sane_machine():
    hw = Hardware(ram_available_gib=32.0)
    assert check_memory_budget(hw, "litert", 16384) is None


def test_auto_backend_cpu_only_is_litert():
    hw = Hardware(gpu_name=None, metal_available=False)
    assert choose_backend(hw, RunConfig()) == "litert"


def test_auto_backend_gpu_is_llamacpp():
    hw = Hardware(gpu_name="NVIDIA GeForce RTX 5090 Laptop GPU")
    assert choose_backend(hw, RunConfig()) == "llamacpp"


def test_explicit_backend_override_wins():
    hw = Hardware(gpu_name="NVIDIA GeForce RTX 5090 Laptop GPU")
    assert choose_backend(hw, RunConfig(backend="transformers")) == "transformers"


def test_default_quant_is_q8_0():
    assert choose_quant(Hardware(), RunConfig()) == "Q8_0"
