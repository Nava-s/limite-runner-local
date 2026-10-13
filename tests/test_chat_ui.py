"""Chat UI tests: the bundled page must obey the Limite interface contract.

The page is the only browser interface we support: it sends only the latest
user question through the runner's preprocessing, carries no tools/MCP or
system-prompt customization, has a real generation cap (default 2048) and
never presents a truncated answer as concluded.
"""

from fastapi.testclient import TestClient

from limite_runner.backends.base import Backend, BackendInfo, GenerationResult
from limite_runner.config import DEFAULT_MAX_TOKENS
from limite_runner.server import create_app

from .test_template import TEMPLATE


class FakeBackend(Backend):
    name = "fake"

    def load(self, **kwargs) -> None:
        return None

    def generate(self, prompt, params) -> GenerationResult:
        return GenerationResult(text="ok")

    def stream(self, prompt, params):
        yield "ok"

    def info(self) -> BackendInfo:
        return BackendInfo(name=self.name, available=True, detail="fake")


def _client(**kwargs) -> TestClient:
    return TestClient(create_app(FakeBackend(), template_path=str(TEMPLATE), **kwargs))


def test_chat_page_is_served_at_root():
    r = _client().get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_chat_page_sends_only_the_last_question():
    # The request builder must keep exactly one user message: history stays
    # visible in the page but is never part of the payload.
    html = _client().get("/simple").text
    payload = html.split("body: JSON.stringify(", 1)[1]
    assert 'messages: [{ role: "user", content: question }]' in payload
    # No system message and no tools/functions anywhere in the payload.
    assert '"tools"' not in payload
    assert '"functions"' not in payload
    assert '"system"' not in payload


def test_chat_page_has_no_tool_or_mcp_controls():
    html = _client().get("/simple").text
    # Tools/MCP/system-prompt customization must be disabled, not offered:
    # no controls, and the page says so explicitly.
    for marker in ('id="tools"', 'id="mcp"', 'name="system"', "tool_choice", "add tool"):
        assert marker not in html
    assert "MCP" in html and "disabilitati" in html


def test_chat_page_default_cap_is_2048_and_configurable():
    html = _client().get("/simple").text
    assert 'id="max_tokens"' in html
    assert 'value="%d"' % DEFAULT_MAX_TOKENS in html
    assert DEFAULT_MAX_TOKENS == 2048
    # Configurable: the server default can be overridden at creation time.
    html2 = _client(default_max_tokens=77).get("/simple").text
    assert 'value="77"' in html2
    assert "77" in html2.split("predefinito del server:")[1][:20]


def test_chat_page_reports_all_end_states():
    # Concluded / token cap / interrupted / error must each be distinguishable
    # and a truncated answer must never read as complete.
    html = _client().get("/simple").text
    assert "Risposta conclusa" in html
    assert "Limite di token raggiunto" in html
    assert "Interrotta" in html
    assert "Errore" in html
    assert "troncata" in html  # the cap warning says the answer is truncated
    assert 'finish === "length"' in html
    assert 'finish === "stop"' in html
    # An end nobody observed is reported as unverified, never as concluded.
    assert 'finish === "unverified"' in html
    assert "NON è verificata" in html
    assert "AbortError" in html  # user interruption path


def test_chat_page_can_stop_a_generation():
    html = _client().get("/simple").text
    assert 'id="stop"' in html
    assert "controller.abort()" in html


def test_legacy_native_ui_props_is_answered_with_a_hint():
    # Leftover tabs of the native llama-server UI poll /props forever: the
    # runner must answer clearly instead of a silent generic 404.
    r = _client().get("/props?model=limite-1b-violetto&autoload=false")
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "limite-runner" in detail
    assert "/" in detail  # points at the supported chat


def test_access_log_filter_silences_the_native_ui_poll_loop():
    import logging

    from limite_runner.server.app import _LegacyUiNoiseFilter

    f = _LegacyUiNoiseFilter()

    def record(msg):
        r = logging.LogRecord(
            "uvicorn.access", logging.INFO, __file__, 1, msg, (), None
        )
        return r

    noisy = record(
        '127.0.0.1:51952 - "GET /props?model=limite-1b-violetto&autoload=false '
        'HTTP/1.1" 404 Not Found'
    )
    assert f.filter(noisy) is False  # the flood is dropped
    for path in ("/sw.js", "/build.json", "/tools", "/v1/streams/lookup", "/health"):
        assert f.filter(record(f'127.0.0.1:1 - "GET {path} HTTP/1.1" 404')) is False
    normal = record('127.0.0.1:1 - "POST /v1/chat/completions HTTP/1.1" 200 OK')
    assert f.filter(normal) is True  # real traffic stays visible
    models = record('127.0.0.1:1 - "GET /v1/models HTTP/1.1" 200 OK')
    assert f.filter(models) is True


def test_rich_ui_bundle_and_configuration():
    client = _client(default_max_tokens=77)
    response = client.get('/')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert '_app/' in response.text
    import re
    scripts = re.findall(r'(?:src|href)="([^\"]+\.(?:js|css))"', response.text)
    assert scripts
    for url in scripts:
        assert client.get('/' + url.removeprefix('./')).status_code == 200
    props = client.get('/ui/props').json()
    assert props['default_generation_settings']['params']['max_tokens'] == 77
    assert props['role'] == 'model'
    assert props['modalities'] == dict(vision=False, audio=False, video=False)
    assert client.get('/health').json()['service'] == 'limite-runner'
