# Author: Tom Sapletta · Part of the ifURI solution.
from __future__ import annotations

import json
import pytest

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


def test_urivision_run_oqllm_docker_live(monkeypatch, tmp_path):
    """Verify urivision execution through local OQLLM Docker instance."""
    import urllib.request
    try:
        with urllib.request.urlopen("http://127.0.0.1:11437/health", timeout=1.0) as resp:
            if resp.status != 200:
                pytest.skip("OQLLM Docker not healthy")
    except Exception:
        pytest.skip("OQLLM Docker not reachable on 11437")

    import litellm
    monkeypatch.setattr(litellm, "supports_vision", lambda *a, **k: True)

    img_file = tmp_path / "home.png"
    img_file.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\x01")

    monkeypatch.setenv("URIVISION_ROOT_SCREENSHOTS", str(tmp_path))
    monkeypatch.setenv("URIVISION_ROOT_REPORTS", str(tmp_path))
    monkeypatch.setenv("URIVISION_LLM_API_BASE", "http://127.0.0.1:11437/v1")
    monkeypatch.setenv("URIVISION_LLM_API_KEY", "dummy")
    monkeypatch.setenv("OPENAI_API_BASE", "http://127.0.0.1:11437/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")

    vuri = (
        "proc   view://local/screenshot/query/decision-card\n"
        "use    llm://openai/oqlos/system-one-router as vision(reasoning=low,json=strict)\n"
        "read   artifact://screenshots/home.png as image/png\n"
        "inspect image for aspect://ui/layout\n"
        'decide "Czy ekran mozna pokazac klientowi?"\n'
        "emit   artifact://reports/home.decision.json as schema://vision/decision-card.v1\n"
    )

    r = process_run(source=vuri)
    assert r["ok"] is True
    assert "card" in r
