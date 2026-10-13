"""Unit tests for the validation numerics (no model needed)."""

import numpy as np

from limite_runner.validate import (
    VALIDATION_PROMPTS,
    mean_kl,
    report_table,
    top1_agreement,
)
from limite_runner.validate import ValidationReport


def test_twenty_fixed_prompts_cover_math_and_non_math():
    assert len(VALIDATION_PROMPTS) == 20
    joined = " ".join(VALIDATION_PROMPTS).lower()
    assert "aime" in joined
    assert "universe" in joined or "quantum" in joined


def test_kl_identical_distributions_is_zero():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(4, 64))
    assert mean_kl(logits, logits) == 0.0


def test_kl_grows_with_perturbation():
    rng = np.random.default_rng(1)
    ref = rng.normal(size=(8, 128))
    small = ref + rng.normal(scale=0.01, size=ref.shape)
    big = ref + rng.normal(scale=1.0, size=ref.shape)
    assert 0.0 < mean_kl(ref, small) < mean_kl(ref, big)


def test_top1_agreement_identical_is_one():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(8, 128))
    assert top1_agreement(logits, logits) == 1.0


def test_report_table_marks_pass_fail():
    rep = ValidationReport(
        backend="llamacpp", quant="Q8_0", ctx=16384, n_prompts=20,
        n_tokens=16, kl_threshold=0.05, top1_threshold=0.90,
        kl_overall=0.007, top1_overall=0.97, finite_everywhere=True,
        stop_tokens_ok=True, passed=True, seconds=1.0, prompts=[],
    )
    assert "PASS" in report_table(rep)
    rep.passed = False
    assert "FAIL" in report_table(rep)


def test_report_table_marks_partial_pass():
    # A PASS with components that could not be checked must not read as a
    # complete verification.
    rep = ValidationReport(
        backend="litert", quant=None, ctx=16384, n_prompts=20,
        n_tokens=16, kl_threshold=0.05, top1_threshold=0.90,
        kl_overall=None, top1_overall=0.95, finite_everywhere=True,
        stop_tokens_ok=True, passed=True, seconds=1.0, prompts=[],
        stop_tokens_status="unverifiable", partial=True,
    )
    table = report_table(rep)
    assert "PASS (partial" in table
    assert "KL n/a" in table
    assert "stop-token check unverifiable" in table


def test_report_table_partial_lists_only_the_missing_parts():
    rep = ValidationReport(
        backend="llamacpp", quant="Q8_0", ctx=16384, n_prompts=20,
        n_tokens=16, kl_threshold=0.05, top1_threshold=0.90,
        kl_overall=0.007, top1_overall=0.97, finite_everywhere=True,
        stop_tokens_ok=True, passed=True, seconds=1.0, prompts=[],
        stop_tokens_status="unverifiable", partial=True,
    )
    table = report_table(rep)
    assert "PASS (partial: stop-token check unverifiable)" in table
    assert "KL n/a" not in table.replace("KL mean=n/a", "")
