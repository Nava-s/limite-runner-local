"""Cache-coherence rule under teacher forcing (near-tie flips are numerical)."""

import numpy as np

from limite_runner.validate import CACHE_FLIP_MAX_GAP, cache_coherent_forced


def _logits(top_vals: dict[int, float], vocab: int = 10) -> np.ndarray:
    row = np.full(vocab, -10.0)
    for i, v in top_vals.items():
        row[i] = v
    return row


def test_identical_argmax_is_coherent():
    L = np.stack([_logits({3: 2.0, 1: 1.0})] * 4)
    assert cache_coherent_forced(L, [3, 3, 3, 3], [3, 3, 3, 3])


def test_small_gap_flip_to_second_is_accepted():
    L = np.stack([_logits({3: 2.0, 1: 2.0 - CACHE_FLIP_MAX_GAP / 2})])
    assert cache_coherent_forced(L, [3], [1])


def test_wide_gap_flip_is_rejected():
    L = np.stack([_logits({3: 2.0, 1: 1.0})])
    assert not cache_coherent_forced(L, [3], [1])


def test_flip_to_a_token_outside_top2_is_rejected():
    L = np.stack([_logits({3: 2.0, 1: 1.99, 5: 0.0})])
    assert not cache_coherent_forced(L, [3], [5])
