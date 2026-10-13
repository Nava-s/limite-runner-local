"""CLI tests: setup --no-build must not prompt (Docker entrypoint path).

Regression: `limite setup --no-build` still asked "Apply the third-party
Limite patch ... and build?" even though --no-build neither applies nor
builds anything; without stdin (container startup) that aborts the setup.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

import limite_runner.cli as cli
from limite_runner.cli import app


def _patch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "detect_hardware", lambda: None)
    monkeypatch.setattr(cli, "choose_backend", lambda hw, cfg: "llamacpp")
    monkeypatch.setattr(cli, "choose_quant", lambda hw, cfg: "Q8_0")
    monkeypatch.setattr(cli, "check_memory_budget", lambda hw, name, ctx: None)
    monkeypatch.setattr(
        cli, "setup_artifacts", lambda ddir, name, q, ctx: {"files": []}
    )
    monkeypatch.setattr(cli, "show_patch_diff", lambda ddir, d: "patch diff")


def test_setup_no_build_skips_confirmation(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    diff_calls = []
    monkeypatch.setattr(
        cli, "show_patch_diff", lambda ddir, d: diff_calls.append(1) or "diff"
    )
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["setup", "--backend", "llamacpp", "--no-build", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "--no-build" in result.output
    assert "skipped" in result.output
    # No patch diff in --no-build: it must not need git in minimal runtimes.
    assert diff_calls == []


def test_setup_build_shows_patch_diff_and_asks(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    monkeypatch.setattr(cli, "show_patch_diff", lambda ddir, d: "patch diff")
    runner = CliRunner()
    result = runner.invoke(
        app, ["setup", "--backend", "llamacpp", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "patch diff" in result.output
    assert "Aborted" in result.output


def test_setup_build_still_asks_confirmation(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path)
    runner = CliRunner()
    # No stdin: the confirmation must still gate the actual build.
    result = runner.invoke(
        app, ["setup", "--backend", "llamacpp", "--data-dir", str(tmp_path)]
    )
    assert result.exit_code != 0
    assert "Aborted" in result.output


def test_show_patch_diff_without_git(monkeypatch, tmp_path):
    # The runtime container ships no git: the informational diff must not
    # abort `setup` (reproduced by the clean Docker run).
    import limite_runner.artifacts as artifacts

    patch_dir = tmp_path / "conv" / "runtime"
    patch_dir.mkdir(parents=True)
    (patch_dir / "limite.patch").write_text("diff --git a b\n", encoding="utf-8")
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError("git")),
    )
    out = artifacts.show_patch_diff(tmp_path, tmp_path)
    assert "THIRD-PARTY CODE" in out
    assert "git not available" in out


# -- configuration precedence, status reporting, missing requirements -----

def test_env_int_and_finish_note():
    import os

    from limite_runner.cli import _env_int, _finish_note

    assert _env_int("LIMITE_MAX_TOKENS", 2048) == 2048
    os.environ["LIMITE_MAX_TOKENS"] = "77"
    try:
        assert _env_int("LIMITE_MAX_TOKENS", 2048) == 77
    finally:
        del os.environ["LIMITE_MAX_TOKENS"]
    # A truncated answer must never read as concluded.
    note = _finish_note("length")
    assert "truncated" in note and "incomplete" in note and "length" in note
    note = _finish_note("stop")
    assert "concluded" in note


def test_build_script_is_resolvable_and_ships_the_companion_source():
    from limite_runner.cli import _build_script

    script = _build_script()
    assert script.exists()
    assert script.name == "build_llamacpp.sh"
    # The companion source ships next to it (wheel layout) or in tools/.
    text = script.read_text(encoding="utf-8")
    assert "limite-logits.cpp" in text


def test_load_backend_honors_limite_model_with_spaces(monkeypatch, tmp_path):
    # LIMITE_MODEL overrides the data-dir-derived GGUF path entirely; paths
    # with spaces must survive.
    import os

    import limite_runner.cli as cli
    from limite_runner.config import RunConfig

    model = tmp_path / "modelli con spazi" / "mio modello.gguf"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"x")
    monkeypatch.setenv("LIMITE_MODEL", str(model))
    captured = {}

    class Stub:
        def load(self):
            return None

    def fake_make_backend(name, cfg, **kwargs):
        captured.update(kwargs)
        return Stub()

    monkeypatch.setattr(cli, "make_backend", fake_make_backend)
    cli._load_backend("llamacpp", RunConfig())
    assert captured["model_path"] == model


def test_missing_model_error_is_actionable():
    from limite_runner.backends.llamacpp import LlamaCppBackend

    b = LlamaCppBackend(model_path=Path("does not exist.gguf"))
    with pytest.raises(RuntimeError) as exc:
        b.load()
    msg = str(exc.value)
    assert "GGUF model not found" in msg
    assert "limite setup" in msg and "LIMITE_MODEL" in msg


def test_missing_server_error_is_actionable(monkeypatch, tmp_path):
    import limite_runner.backends.llamacpp as mod

    model = tmp_path / "fake.gguf"
    model.write_bytes(b"x")
    monkeypatch.setattr(mod, "find_llama_server", lambda explicit=None: None)
    b = mod.LlamaCppBackend(model_path=model)
    with pytest.raises(RuntimeError) as exc:
        b.load()
    msg = str(exc.value)
    assert "llama-server" in msg
    assert "LIMITE_LLAMA_SERVER" in msg and "build_llamacpp.sh" in msg


def test_missing_template_error_is_actionable(tmp_path):
    from limite_runner.config import RunConfig

    with pytest.raises(RuntimeError) as exc:
        cli._render_or_raw(RunConfig(data_dir=tmp_path), "What is 2 + 2?", False)
    msg = str(exc.value)
    assert "chat template not found" in msg
    assert "limite setup" in msg


def test_missing_model_error_surfaces_as_cli_error(monkeypatch, tmp_path):
    # Full CLI path: a missing model must print one actionable ERROR line and
    # exit 1, never a traceback.
    import shutil

    spaced = tmp_path / "data dir con spazi"
    (spaced / "conv" / "upstream").mkdir(parents=True)
    shutil.copy(
        Path(__file__).parent / "data" / "chat_template.jinja",
        spaced / "conv" / "upstream" / "chat_template.jinja",
    )
    monkeypatch.setattr(cli, "detect_hardware", lambda: None)
    monkeypatch.setattr(cli, "choose_backend", lambda hw, cfg: "llamacpp")
    monkeypatch.setattr(cli, "choose_quant", lambda hw, cfg: "Q8_0")
    monkeypatch.setattr(cli, "check_memory_budget", lambda hw, name, ctx: None)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            "What is 2 + 2?",
            "--backend",
            "llamacpp",
            "--data-dir",
            str(spaced),
        ],
    )
    assert result.exit_code == 1
    assert "ERROR" in result.output
    assert "GGUF model not found" in result.output
    assert "limite setup" in result.output


def test_cfg_honors_limite_quant_and_limite_ctx(monkeypatch):
    # Direct CLI use must respect the documented LIMITE_QUANT / LIMITE_CTX
    # (the launcher only translates them into flags for its own invocation).
    monkeypatch.setenv("LIMITE_QUANT", "Q5_K_M")
    monkeypatch.setenv("LIMITE_CTX", "32768")
    cfg = cli._cfg(None, None, None, None, None)
    assert cfg.quant == "Q5_K_M"
    assert cfg.ctx == 32768


def test_cfg_flags_beat_env(monkeypatch):
    monkeypatch.setenv("LIMITE_QUANT", "Q5_K_M")
    monkeypatch.setenv("LIMITE_CTX", "32768")
    cfg = cli._cfg(None, "Q8_0", 16384, None, None)
    assert cfg.quant == "Q8_0"
    assert cfg.ctx == 16384


def test_cfg_env_defaults_without_env(monkeypatch):
    monkeypatch.delenv("LIMITE_QUANT", raising=False)
    monkeypatch.delenv("LIMITE_CTX", raising=False)
    cfg = cli._cfg(None, None, None, None, None)
    assert cfg.quant is None  # auto-choice later
    assert cfg.ctx == 16384


def test_cfg_rejects_invalid_ctx(monkeypatch):
    monkeypatch.setenv("LIMITE_CTX", "123")
    with pytest.raises(Exception):
        cli._cfg(None, None, None, None, None)


def test_finish_note_unverified_is_not_concluded():
    note = cli._finish_note("unverified")
    assert "NOT verified" in note
    assert "concluded" not in note.replace("NOT verified as complete", "")


def test_setup_build_receives_patch_and_data_dir(monkeypatch, tmp_path):
    # Regression: the build subprocess did not get --patch nor LIMITE_DATA_DIR
    # from --data-dir, so on a clean install it searched for the patch in the
    # wrong place. It must receive both, pointing at the setup data dir.
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        import subprocess as sp

        return sp.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli, "detect_hardware", lambda: None)
    monkeypatch.setattr(cli, "choose_backend", lambda hw, cfg: "llamacpp")
    monkeypatch.setattr(cli, "choose_quant", lambda hw, cfg: "Q8_0")
    monkeypatch.setattr(cli, "check_memory_budget", lambda hw, name, ctx: None)
    monkeypatch.setattr(
        cli, "setup_artifacts", lambda ddir, name, q, ctx: {"files": []}
    )
    monkeypatch.setattr(cli, "show_patch_diff", lambda ddir, d: "diff")

    patch_dir = tmp_path / "conv" / "runtime"
    patch_dir.mkdir(parents=True)
    (patch_dir / "limite.patch").write_text("diff --git a b\n", encoding="utf-8")

    import subprocess

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("typer.confirm", lambda *a, **k: True)

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["setup", "--backend", "llamacpp", "--data-dir", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    cmd = captured["cmd"]
    assert "--patch" in cmd
    assert cmd[cmd.index("--patch") + 1] == str(tmp_path / "conv" / "runtime" / "limite.patch")
    assert captured["env"]["LIMITE_DATA_DIR"] == str(tmp_path)
