"""Unit tests for llama-server log draining and startup cleanup (no model).

Regression coverage: llama-server's combined stdout/stderr pipe was never
read while the server ran (a full pipe buffer blocks the subprocess), and a
failed startup (exit or readiness timeout) leaked the subprocess and pipe.
"""

import subprocess
import sys
import threading

import pytest

import limite_runner.backends.llamacpp as mod
from limite_runner.backends.llamacpp import LOG_TAIL_LINES, LlamaCppBackend


def test_drain_logs_consumes_output_and_keeps_tail():
    b = LlamaCppBackend()
    script = "; ".join(f"print('line {i}')" for i in range(500))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    t = threading.Thread(target=b._drain_logs, args=(proc.stdout,))
    t.start()
    proc.wait(timeout=60)
    t.join(timeout=30)
    assert not t.is_alive()
    # The pipe was drained continuously: the ring buffer keeps only the tail.
    assert len(b._log_lines) == LOG_TAIL_LINES
    assert b._log_tail().endswith("line 499")


def test_load_failure_cleans_up_subprocess(monkeypatch, tmp_path):
    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    server = tmp_path / "fake-server"
    server.write_text("dummy")  # exists so find_llama_server accepts it

    b = LlamaCppBackend(model_path=model, server_bin=server)
    real_popen = mod.subprocess.Popen

    def fake_popen(cmd, **kwargs):
        # Stand-in for a llama-server that dies immediately during load.
        return real_popen(
            [sys.executable, "-c", "print('boom'); sys.exit(3)"],
            stdout=kwargs.get("stdout"),
            stderr=kwargs.get("stderr"),
            text=True,
        )

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    with pytest.raises(RuntimeError, match="exited during load"):
        b.load()
    # No subprocess, client or reader thread left behind; the real server
    # output (not a silent empty string) is in the error and log tail.
    assert b._proc is None
    assert b._client is None
    assert "boom" in b._log_tail()


def test_close_joins_the_reader_thread(tmp_path):
    b = LlamaCppBackend()
    proc = subprocess.Popen(
        [sys.executable, "-c", "print('hello')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    b._proc = proc
    b._log_thread = threading.Thread(
        target=b._drain_logs, args=(proc.stdout,), daemon=True
    )
    b._log_thread.start()
    b.close()
    assert b._proc is None
    assert b._log_thread is None
    assert proc.poll() is not None
