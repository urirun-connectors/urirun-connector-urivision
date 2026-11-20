# Author: Tom Sapletta · Part of the ifURI solution.
from __future__ import annotations

import json
import sys
import threading
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from urirun_connector_urivision import (
    CONNECTOR_ID,
    process_compile,
    process_run,
    urirun_bindings,
)

ROUTES = {
    "urivision://host/doctor/query/report",
    "urivision://host/process/query/compile",
    "urivision://host/process/command/run",
}

VURI = (
    "proc   view://local/screenshot/query/decision-card\n"
    "use    llm://gemini/gemini-3.5-flash as vision(reasoning=low,json=strict)\n"
    "read   artifact://screenshots/x/home.png as image/png\n"
    "inspect image for aspect://ui/layout\n"
    'decide "Czy ekran można pokazać klientowi?"\n'
    "emit   artifact://reports/x/home.decision.json as schema://vision/decision-card.v1\n"
)


def test_connector_id():
    assert CONNECTOR_ID == "urivision"


def test_bindings_serializable_and_complete():
    b = urirun_bindings()["bindings"]
    assert set(b) == ROUTES
    json.dumps(urirun_bindings())
    # compile is a read-only query -> in-process; run calls a model -> isolated subprocess
    assert b["urivision://host/process/query/compile"]["adapter"] == "local-function"
    assert b["urivision://host/process/command/run"]["adapter"] == "local-function-subprocess"


def test_lazy_import_no_urivision_at_module_top():
    import importlib
    import sys

    for m in [m for m in list(sys.modules) if m == "urivision" or m.startswith("urivision.")]:
        del sys.modules[m]
    importlib.reload(importlib.import_module("urirun_connector_urivision.core"))
    assert "urivision" not in sys.modules, "connector import pulled urivision eagerly"


def test_compile_requires_source():
    assert process_compile(source="", path="")["ok"] is False


def test_compile_dry_run_no_model():
    # The cheap gate: compile the process to its would-be model call WITHOUT invoking a model.
    r = process_compile(source=VURI)
    assert r["ok"] is True and r.get("compiled_ok") is True
    assert "call" in r  # the compiled vision-LLM call


def test_run_missing_source_returns_envelope():
    assert process_run(source="", path="")["ok"] is False  # never raises


def test_urivision_run_oqllm_http_integration(monkeypatch, tmp_path):
    """Verify end-to-end process_run through a deterministic local OpenAI/OQLLM HTTP endpoint."""
    expected_card = {
        "decision": "ok",
        "confidence": 0.95,
        "summary": "Layout clean and ready for customer view.",
        "findings": [
            {
                "aspect": "ui/layout",
                "severity": "info",
                "evidence": "Header and primary controls are aligned.",
            }
        ],
    }
    captured_requests: list[dict] = []

    class _OqllmHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            captured_requests.append(
                {"path": self.path, "auth": self.headers.get("Authorization"), "body": body}
            )
            payload = json.dumps(
                {
                    "id": "chatcmpl-test-urivision",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(expected_card),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), _OqllmHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def _completion(*, model, messages, api_base, api_key, **_kwargs):
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}/chat/completions",
            data=json.dumps({"model": model, "messages": messages}).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=5.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
        )

    fake_litellm = types.ModuleType("litellm")
    fake_litellm.completion = _completion
    fake_litellm.supports_vision = lambda *a, **k: True
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    img_file = tmp_path / "home.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x01")
    api_base = f"http://127.0.0.1:{port}/v1"

    monkeypatch.setenv("URIVISION_ROOT_SCREENSHOTS", str(tmp_path))
    monkeypatch.setenv("URIVISION_ROOT_REPORTS", str(tmp_path))
    monkeypatch.setenv("URIVISION_LLM_API_BASE", api_base)
    monkeypatch.setenv("URIVISION_LLM_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", api_base)
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    vuri = (
        "proc   view://local/screenshot/query/decision-card\n"
        "use    llm://openai/oqlos/system-one-router as vision(reasoning=low,json=strict)\n"
        "read   artifact://screenshots/home.png as image/png\n"
        "inspect image for aspect://ui/layout\n"
        'decide "Czy ekran mozna pokazac klientowi?"\n'
        "emit   artifact://reports/home.decision.json as schema://vision/decision-card.v1\n"
    )

    try:
        r = process_run(source=vuri)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)

    assert r["ok"] is True
    assert r["decision"] == "ok"
    assert r["card"] == expected_card
    assert len(captured_requests) == 1
    assert captured_requests[0]["path"] == "/v1/chat/completions"
    assert captured_requests[0]["auth"] == "Bearer dummy"
    report_path = tmp_path / "home.decision.json"
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == expected_card
