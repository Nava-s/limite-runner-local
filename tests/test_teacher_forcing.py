"""Teacher-forcing input file: raw int32 little-endian, read by limite-logits --forced."""

import numpy as np

from limite_runner.validate import write_forced_tokens


def test_forced_tokens_roundtrip_int32_le(tmp_path):
    ids = [1, 151665, 7, 2147483647]
    p = write_forced_tokens(tmp_path / "ids.bin", ids)
    assert p.stat().st_size == 4 * len(ids)
    assert np.fromfile(p, dtype="<i4").tolist() == ids
