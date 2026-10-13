"""RunConfig data-dir resolution: precedence, env override, default."""

from pathlib import Path

from limite_runner.config import RunConfig


def test_explicit_data_dir_wins_over_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LIMITE_DATA_DIR", str(tmp_path / "from-env"))
    cfg = RunConfig(data_dir=tmp_path / "explicit")
    assert cfg.resolved_data_dir() == tmp_path / "explicit"


def test_env_override_beats_platformdirs(monkeypatch, tmp_path):
    monkeypatch.setenv("LIMITE_DATA_DIR", str(tmp_path / "from-env"))
    cfg = RunConfig()
    assert cfg.resolved_data_dir() == tmp_path / "from-env"


def test_default_is_absolute_path_without_env(monkeypatch):
    monkeypatch.delenv("LIMITE_DATA_DIR", raising=False)
    d = RunConfig().resolved_data_dir()
    assert isinstance(d, Path)
    assert d.is_absolute()
