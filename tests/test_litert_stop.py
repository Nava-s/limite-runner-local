"""Unit tests for the LiteRT incremental stop-string filter and streaming.

Regression coverage for the streaming bug: the stop path re-emitted the
whole accumulated text (reproduced as ``abcdef`` -> ``abcdefabcdef``) and a
stop marker split across chunks leaked its prefix into the output. Also
covers ``stream_full``: the real finish reason (length vs stop) instead of a
hardcoded "stop".
"""

from types import SimpleNamespace

from limite_runner.backends.base import SamplingParams
from limite_runner.backends.litert import LiteRTBackend, _StopFilter

ENDOFTEXT = "<|endoftext|>"


def drain(filt: _StopFilter, pieces: list[str]) -> list[str]:
    out: list[str] = []
    for p in pieces:
        delta, done = filt.feed(p)
        if delta:
            out.append(delta)
        if done:
            return out
    tail = filt.flush()
    if tail:
        out.append(tail)
    return out


def test_no_reemission_of_accumulated_text_at_stop():
    filt = _StopFilter((ENDOFTEXT,))
    assert drain(filt, ["abc", f"def{ENDOFTEXT}tail"]) == ["abc", "def"]


def test_no_reemission_when_stop_in_single_chunk():
    filt = _StopFilter((ENDOFTEXT,))
    assert drain(filt, [f"abcdef{ENDOFTEXT}"]) == ["abcdef"]


def test_stop_split_across_chunks_does_not_leak_prefix():
    filt = _StopFilter(("STOP",))
    assert drain(filt, ["AB", "ST", "OP", "more"]) == ["AB"]


def test_text_looking_like_stop_prefix_is_released_when_not_a_stop():
    filt = _StopFilter(("STOP",))
    assert drain(filt, ["STO", "NE"]) == ["STONE"]


def test_stop_complete_in_later_chunk_after_hold():
    filt = _StopFilter(("STOP",))
    # "TO" is held (prefix of STOP? no: "STOP".startswith("TO") is False) --
    # but "STO" is held; when "P" arrives the stop completes and nothing of
    # it may leak.
    assert drain(filt, ["STO", "P and done"]) == []


def test_flush_emits_held_tail_when_generation_ends_without_stop():
    filt = _StopFilter((ENDOFTEXT,))
    delta, done = filt.feed("abc<")
    assert (delta, done) == ("abc", False)  # "<" held: could start ENDOFTEXT
    assert filt.flush() == "<"  # no stop ever came: release the held text


def test_earliest_stop_wins_across_multiple_stops():
    filt = _StopFilter(("<|im_end|>", ENDOFTEXT))
    pieces = ["a<|im_end|>b"]
    assert drain(filt, pieces) == ["a"]


def test_text_between_two_arrivals_before_stop_is_emitted_once():
    filt = _StopFilter((ENDOFTEXT,))
    out = drain(filt, ["one ", "two ", f"three{ENDOFTEXT}"])
    assert out == ["one ", "two ", "three"]
    assert "".join(out).count("one") == 1


def test_empty_stop_strings_are_ignored():
    filt = _StopFilter(("", ENDOFTEXT))
    assert drain(filt, ["abc", f"{ENDOFTEXT}"]) == ["abc"]


def test_no_stops_passes_everything_through():
    filt = _StopFilter(())
    assert drain(filt, ["a", "b", "c"]) == ["a", "b", "c"]


# -- stream_full: real finish reason -------------------------------------


class FakeSession:
    def __init__(self, texts, decode_count=None, bench_error=False):
        self.texts = texts
        self.decode_count = decode_count
        self.bench_error = bench_error
        self.closed = False

    def run_prefill(self, contents):
        pass

    def run_decode_async(self):
        for t in self.texts:
            yield SimpleNamespace(texts=[t])

    def get_benchmark_info(self):
        if self.bench_error:
            raise RuntimeError("counters unavailable")
        return SimpleNamespace(last_decode_token_count=self.decode_count)

    def close(self):
        self.closed = True


class FakeEngine:
    def __init__(self, session, token_ids=None):
        self.session = session
        self.token_ids = token_ids

    def create_session(self, **kwargs):
        return self.session

    def tokenize(self, text):
        return list(self.token_ids if self.token_ids is not None else text)


def _backend(session, token_ids=None) -> LiteRTBackend:
    b = LiteRTBackend(bundle_path="unused")
    b.load = lambda: None  # type: ignore[method-assign]
    b._engine = FakeEngine(session, token_ids)
    return b


def test_stream_full_reports_length_when_cap_reached():
    # Engine decoded exactly max_output_tokens tokens: the cap ended it.
    s = FakeSession(["hello world"], decode_count=10)
    chunks = list(
        _backend(s).stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
        )
    )
    assert "".join(c.text for c in chunks) == "hello world"
    assert chunks[-1].finish_reason == "length"


def test_stream_full_reports_stop_on_early_engine_halt():
    s = FakeSession(["hello"], decode_count=3)
    chunks = list(
        _backend(s).stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
        )
    )
    assert chunks[-1].finish_reason == "stop"


def test_stream_full_reports_stop_when_stop_string_hits():
    s = FakeSession([f"abc{ENDOFTEXT}"], decode_count=10)
    chunks = list(
        _backend(s).stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
        )
    )
    assert "".join(c.text for c in chunks) == "abc"
    assert chunks[-1].finish_reason == "stop"


def test_stream_full_falls_back_to_tokenize_when_counters_missing():
    s = FakeSession(["hello world"], bench_error=True)
    # Re-tokenizing the emitted text yields >= max_tokens ids: cap hit.
    b = _backend(s, token_ids=list(range(10)))
    chunks = list(
        b.stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
        )
    )
    assert chunks[-1].finish_reason == "length"


def test_stream_full_closes_the_session():
    s = FakeSession(["x"], decode_count=1)
    list(
        _backend(s).stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=5)
        )
    )
    assert s.closed


def test_generate_propagates_finish_reason():
    s = FakeSession(["hello world"], decode_count=10)
    res = _backend(s).generate(
        "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=10)
    )
    assert res.finish_reason == "length"
    assert res.text == "hello world"


def test_stream_full_reports_unverified_when_nothing_was_observed():
    # Neither the session counters nor the tokenizer can tell how the
    # generation ended: the end condition is unknown and must NOT be
    # fabricated as "stop" (an unobserved end is never a concluded one).
    s = FakeSession(["hello"], bench_error=True)
    b = _backend(s)

    def boom(text):
        raise RuntimeError("no tokenizer")

    b._engine.tokenize = boom
    chunks = list(
        b.stream_full(
            "p", SamplingParams(temperature=0.0, top_p=1.0, max_tokens=5)
        )
    )
    assert chunks[-1].finish_reason == "unverified"
