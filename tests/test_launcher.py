"""Launcher tests: portability contract of Avvia-Limite-GPU.cmd + limite-serve.sh.

The launcher is the supported Windows entry point: it must carry no personal
paths, must work from any working directory (paths with spaces included) and
must explain what is missing instead of downloading or rebuilding anything.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CMD = REPO / "Avvia-Limite-GPU.cmd"
SH = REPO / "scripts" / "limite-serve.sh"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_launcher_files_exist():
    assert CMD.exists()
    assert SH.exists()


def test_launcher_has_no_personal_paths():
    # Regression guard: the first draft hard-coded /home/<user>/... paths of
    # the reference PC. Nothing personal may come back -- including in the
    # optional experiments and their report files (they are publishable).
    targets = [CMD, SH]
    scripts = REPO / "scripts"
    if scripts.exists():
        targets.extend(p for p in scripts.rglob("*") if p.is_file())
    extra = REPO / "limite-launcher.env.example"
    if extra.exists():
        targets.append(extra)
    exp = REPO / "experiments"
    if exp.exists():
        targets.extend(p for p in exp.rglob("*") if p.is_file())
    for path in targets:
        text = _read(path).lower()
        # Placeholder home paths are fine (templates, anonimized reports):
        # strip them before looking for a real /home/<something> path.
        text = text.replace("/home/<user>/", "~placeholder~/").replace(
            "/home/<utente>/", "~placeholder~/"
        )
        for needle in (
            "antonio",
            "proietti",
            "personal notebook",
            "/home/",
            "c:\\users",
            "appdata",
        ):
            if path.name == "limite-launcher.env.example" and needle == "/home/":
                continue
            assert needle not in text, f"{path.name} leaks {needle!r}"


def test_launcher_defaults_are_documented():
    cmd = _read(CMD)
    sh = _read(SH)
    for text in (cmd, sh):
        assert "2048" in text  # default generation cap
        assert "127.0.0.1" in text
        assert "8080" in text
    assert "LIMITE_WSL_DISTRO" in cmd
    assert "LIMITE_DATA_DIR" in sh and "LIMITE_LLAMA_SERVER" in sh
    assert "LIMITE_GPU" in sh and "LIMITE_MAX_TOKENS" in sh
    # Config precedence is stated where users look for it.
    assert "limite-launcher.env" in cmd


def test_launcher_uses_its_own_directory_not_the_cwd():
    # Starting the launcher from any directory must work: everything is
    # resolved relative to the script location.
    cmd = _read(CMD)
    assert "%~dp0" in cmd
    assert "scripts\\limite-serve.sh" in cmd or "scripts/limite-serve.sh" in cmd


def test_env_example_ships_and_is_commented():
    example = REPO / "limite-launcher.env.example"
    assert example.exists()
    text = _read(example)
    for key in (
        "LIMITE_WSL_DISTRO",
        "LIMITE_DATA_DIR",
        "LIMITE_LLAMA_SERVER",
        "LIMITE_MODEL",
        "LIMITE_HOST",
        "LIMITE_PORT",
        "LIMITE_GPU",
        "LIMITE_MAX_TOKENS",
    ):
        assert key in text
    # Every sample line is commented out: copying it changes nothing.
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            pytest.fail(f"limite-launcher.env.example has an active line: {line!r}")


def test_gitignore_excludes_user_launcher_config():
    text = _read(REPO / ".gitignore")
    assert "limite-launcher.env" in text


# -- live preflight (Windows + WSL only) ---------------------------------

def _wsl_available() -> bool:
    return sys.platform == "win32" and shutil.which("wsl") is not None


@pytest.mark.skipif(not _wsl_available(), reason="needs Windows + WSL2")
def test_check_mode_reports_missing_distribution():
    env = dict(os.environ, LIMITE_WSL_DISTRO="limite-no-such-distro")
    r = subprocess.run(
        ["cmd", "/c", str(CMD), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "limite-no-such-distro" in out
    assert "ERRORE" in out


@pytest.mark.skipif(not _wsl_available(), reason="needs Windows + WSL2")
def test_check_mode_reports_missing_data_dir_with_spaces_in_path(tmp_path):
    # A data dir that does not exist (with spaces in the path) must produce a
    # clear error quoting the path -- and proves spaces survive the launcher.
    missing = tmp_path / "data dir con spazi"
    env = dict(os.environ, LIMITE_DATA_DIR=str(missing))
    r = subprocess.run(
        ["cmd", "/c", str(CMD), "--check"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert r.returncode != 0
    out = r.stdout + r.stderr
    assert "data dir con spazi" in out
    assert "ERROR" in out or "ERRORE" in out


# -- behavioural probe tests (Windows PowerShell, same as the launcher) ---
#
# The launcher decides "reuse / start / refuse" from the probe exit codes:
# these tests run the REAL launcher probe path (Avvia-Limite-GPU.cmd --probe,
# which calls scripts/health-probe.ps1 under powershell.exe) against local
# servers that reproduce each scenario: 200, 503, 404, timeout, refused.

import http.server
import threading
import time


def _probe(port: int, host: str = "127.0.0.1") -> int:
    env = dict(os.environ, LIMITE_HOST=host, LIMITE_PORT=str(port))
    r = subprocess.run(
        ["cmd", "/c", str(CMD), "--probe"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    return r.returncode


class _ScenarioHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        scenario = self.server.scenario
        if scenario == "timeout":
            # Accepts the connection but never answers within the probe
            # timeout: a timeout must NOT be read as "port is free".
            time.sleep(15)
            return
        if scenario == "ok":
            body = b'{"status": "ok", "ready": true, "service": "limite-runner"}'
            code = 200
        elif scenario == "not_ready":
            body = b'{"status": "not_ready", "ready": false, "service": "limite-runner"}'
            code = 503
        elif scenario == "bare_ok":
            # The old native llama-server: {"status":"ok"} with NO identity.
            body = b'{"status": "ok"}'
            code = 200
        else:  # 404 or any non-runner page
            body = b"<html><body>404 page not found</body></html>"
            code = 404
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve(scenario: str):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScenarioHandler)
    server.scenario = scenario
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_200_runner_is_reused():
    server = _serve("ok")
    try:
        assert _probe(server.server_address[1]) == 0
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_503_runner_is_not_ready_not_other_service():
    # Regression: HTTP 503 is an answer, and the body still carries the
    # identity -> the runner is present but not ready (2), never "free".
    server = _serve("not_ready")
    try:
        assert _probe(server.server_address[1]) == 2
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_200_without_identity_is_other_service():
    # The original bug: any {"status":"ok"} was treated as the runner. The
    # native llama-server answers exactly this and must be refused (3).
    server = _serve("bare_ok")
    try:
        assert _probe(server.server_address[1]) == 3
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_404_is_other_service():
    server = _serve("404")
    try:
        assert _probe(server.server_address[1]) == 3
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_timeout_is_uncertain_not_free():
    # A timeout does not prove the port is free: the launcher must not start
    # automatically on 4.
    server = _serve("timeout")
    try:
        assert _probe(server.server_address[1]) == 4
    finally:
        server.shutdown()


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_connection_refused_means_free():
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free_port = s.getsockname()[1]
    assert _probe(free_port) == 1


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_dns_failure_is_uncertain_not_free():
    assert _probe(8080, host="nonexistent-limite-probe.invalid") == 4


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell")
@pytest.mark.parametrize(
    "socket_code, expected",
    [(None, 4), (11001, 4), (10061, 1), (10060, 4)],
    ids=["generic-connect-failure", "host-not-found", "nested-refused", "timed-out"],
)
def test_probe_connect_failure_requires_specific_socket_error(socket_code, expected):
    # Exercise the actual script with deterministic transport exceptions.
    # Keep an extra wrapper to verify traversal of the entire inner chain.
    ps1 = str(REPO / "scripts" / "health-probe.ps1").replace("'", "''")
    inner = "$null" if socket_code is None else (
        "[System.Exception]::new('wrapper', "
        f"[System.Net.Sockets.SocketException]::new({socket_code}))"
    )
    script = (
        "function Invoke-WebRequest { "
        "throw [System.Net.WebException]::new('transport', "
        + inner
        + ", [System.Net.WebExceptionStatus]::ConnectFailure, $null) }; "
        + f"& '{ps1}' -Url 'http://127.0.0.1:8080/health'; exit $LASTEXITCODE"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="needs Windows PowerShell + cmd")
def test_probe_uses_windows_powershell_like_the_launcher():
    # Contract: the probe runs under Windows PowerShell (powershell.exe), not
    # pwsh, and stays 5.1-compatible (no -SkipHttpErrorCheck).
    ps1 = REPO / "scripts" / "health-probe.ps1"
    assert ps1.exists()
    cmd = _read(CMD)
    assert "powershell.exe" in cmd
    assert "health-probe.ps1" in cmd
    text = _read(ps1)
    assert "pwsh" not in text
    assert "SkipHttpErrorCheck" not in text
    for code in ("exit 0", "exit 1", "exit 2", "exit 3", "exit 4"):
        assert code in text
