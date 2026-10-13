"""Unit tests for the tri-state stop-token check (no model needed)."""

from limite_runner.backends.base import Backend, BackendInfo, GenerationResult, SamplingParams
from limite_runner.config import EOS_TOKEN_ID, TOKEN_ENDOFTEXT, TOKEN_IM_END
from limite_runner.validate import check_stop_tokens


class _Tokenizer:
    """Minimal stand-in: maps special tokens to fixed ids."""

    def __init__(self, mapping: dict[str, list[int]]) -> None:
        self.mapping = mapping

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(self.mapping[text])


class _Stub(Backend):
    name = "stub"

    def __init__(
        self,
        tokenizer=None,
        eos: list[list[int]] | None = None,
        eos_error: Exception | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        if eos is not None or eos_error is not None:
            def _eos_ids() -> list[list[int]]:
                if eos_error is not None:
                    raise eos_error
                return list(eos or [])

            # Set only when the backend exposes engine EOS ids at all:
            # hasattr() is how the check tells the two situations apart.
            self.eos_ids = _eos_ids

    def load(self, **kwargs) -> None: ...
    def generate(self, prompt: str, params: SamplingParams) -> GenerationResult: ...
    def stream(self, prompt: str, params: SamplingParams): ...
    def info(self) -> BackendInfo: ...


GOOD_TOK = _Tokenizer(
    {
        TOKEN_ENDOFTEXT: [EOS_TOKEN_ID],
        TOKEN_IM_END: [EOS_TOKEN_ID + 2],
    }
)
BAD_TOK = _Tokenizer(
    {
        TOKEN_ENDOFTEXT: [999],
        TOKEN_IM_END: [],
    }
)


def test_verified_when_both_sides_check_out():
    res = check_stop_tokens(_Stub(tokenizer=GOOD_TOK), _Stub(tokenizer=GOOD_TOK))
    assert res.status == "verified"
    assert res.ok is True
    assert any(str(EOS_TOKEN_ID) in e for e in res.evidence)


def test_failed_when_endoftext_maps_to_wrong_id():
    res = check_stop_tokens(_Stub(tokenizer=GOOD_TOK), _Stub(tokenizer=BAD_TOK))
    assert res.status == "failed"
    assert res.ok is False


def test_failed_when_engine_declares_no_eos():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK), _Stub(eos=[])  # engine with no EOS ids
    )
    assert res.status == "failed"


def test_unverifiable_when_neither_tokenizer_nor_eos_ids():
    # Regression: the old bool check returned True here without verifying
    # anything (this was the llamacpp candidate path).
    res = check_stop_tokens(_Stub(tokenizer=GOOD_TOK), _Stub())
    assert res.status == "unverifiable"
    assert res.ok is True  # unverifiable is reported, not treated as failure
    assert any("not checkable" in e for e in res.evidence)


def test_unverifiable_when_eos_check_raises():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK),
        _Stub(eos_error=RuntimeError("boom")),
    )
    assert res.status == "unverifiable"


def test_stop_token_emission_evidence_from_free_tokens():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK),
        _Stub(tokenizer=GOOD_TOK),
        free_tokens=[10, 20, EOS_TOKEN_ID, 30],
    )
    assert res.status == "verified"
    # An emission is recorded, but explicitly not claimed as a halt proof.
    assert any(
        "emission observed" in e and "step 2" in e and "not by itself a proof" in e
        for e in res.evidence
    )


def test_actual_stop_probe_reports_engine_halt():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK),
        _Stub(tokenizer=GOOD_TOK),
        actual_stop="stop",
    )
    assert res.status == "verified"
    assert any(
        "actual stop" in e and "finish_reason=stop" in e for e in res.evidence
    )


def test_actual_stop_length_is_informational_not_a_proof():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK),
        _Stub(tokenizer=GOOD_TOK),
        actual_stop="length",
    )
    assert res.status == "verified"  # the cap is ours, not a stop failure
    assert any("finish_reason=length" in e and "no halt observed" in e for e in res.evidence)


def test_missing_actual_stop_probe_is_not_claimed():
    res = check_stop_tokens(_Stub(tokenizer=GOOD_TOK), _Stub(tokenizer=GOOD_TOK))
    assert not any("actual stop" in e for e in res.evidence)


def test_missing_stop_in_free_tokens_is_informational_only():
    res = check_stop_tokens(
        _Stub(tokenizer=GOOD_TOK),
        _Stub(tokenizer=GOOD_TOK),
        free_tokens=[10, 20],
    )
    assert res.status == "verified"
    assert any("informational" in e for e in res.evidence)


def test_evidence_listed_for_both_sides():
    res = check_stop_tokens(_Stub(tokenizer=GOOD_TOK), _Stub(eos=[[7, 8]]))
    assert res.status == "verified"
    assert any(e.startswith("reference:") for e in res.evidence)
    assert any(e.startswith("candidate:") for e in res.evidence)
