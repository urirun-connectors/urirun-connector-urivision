# Author: Tom Sapletta · Part of the ifURI solution.
"""urirun-connector-urivision — native URI surface over urivision / VURI.

urivision compiles a ``.vuri`` process (image + model + aspects + decision question + result
schema) into a vision-LLM call that returns a DecisionCard: a decision with evidence, not a
description. It is the DECISION layer of the perception stack (vdisplay = environment,
vql = visual structure, urivision = decision). It shipped as a standalone DSL + ``run_file`` +
CLI — URI-flavoured in its OWN DSL but NOT a served urirun connector, so a node/flow/readiness
kernel could not compose it. This connector serves it as ``urivision://`` routes.

Built to URI_NATIVE_CONNECTOR_CHECKLIST: lazy imports, handlers never raise (urirun envelope),
the compile/dry-run query is in-process, the model-calling run is isolated, heavy deps optional
(the LLM runner needs ``urivision[litellm]`` — the connector degrades to ok:false without it).
"""
from __future__ import annotations

from typing import Any

import urirun

from . import _urirun_compat

CONNECTOR_ID = "urivision"
conn = _urirun_compat.connector(CONNECTOR_ID, scheme="urivision")


def _ok(**kw: Any) -> dict[str, Any]:
    return urirun.ok(connector=CONNECTOR_ID, **kw)


def _fail(msg: str, action: str, **extra: Any) -> dict[str, Any]:
    return urirun.fail(msg, connector=CONNECTOR_ID, action=action, **extra)


def _source(source: str, path: str) -> tuple[str | None, dict | None]:
    """Resolve the .vuri text from inline ``source`` or a file ``path``. (text, error_dict)."""
    if source:
        return source, None
    if path:
        try:
            from pathlib import Path
            return Path(path).expanduser().read_text(encoding="utf-8"), None
        except OSError as exc:
            return None, _fail(str(exc), "urivision-run")
    return None, _fail("source (inline .vuri) or path is required", "urivision-run")


@conn.handler("process/query/compile", isolated=False,
              meta={"label": "Compile a .vuri process to the vision-LLM call WITHOUT running it (dry-run)"})
def process_compile(source: str = "", path: str = "") -> dict[str, Any]:
    """Parse + validate + compile a ``.vuri`` process and return the model call it WOULD make —
    no model invoked. The cheap gate: catch a malformed process here before spending a call."""
    text, err = _source(source, path)
    if err:
        return err
    try:
        from urivision import run_source
        r = run_source(text, dry_run=True)
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc), "urivision-compile")
    ok = bool(r.get("ok", True))
    return _ok(action="urivision-compile", compiled_ok=ok, **{
        k: v for k, v in r.items() if k not in ("ok", "connector", "action")})


@conn.handler("process/command/run", isolated=True,
              meta={"label": "Run a .vuri process → DecisionCard (decision + evidence via vision LLM)"})
def process_run(source: str = "", path: str = "", roots: dict | None = None) -> dict[str, Any]:
    """Execute a ``.vuri`` process end-to-end: compile → vision-LLM call → DecisionCard. Isolated
    (external model call, can be slow). Needs ``urivision[litellm]`` + model creds — absent, it
    degrades to ok:false with the reason (never crashes the node). ``roots`` maps artifact://
    root names to paths for image/report resolution."""
    text, err = _source(source, path)
    if err:
        return err
    try:
        from urivision import run_source
        r = run_source(text, roots=roots or None, dry_run=False)
    except Exception as exc:  # noqa: BLE001
        return _fail(str(exc), "urivision-run", hint="install urivision[litellm] + set model creds")
    if not r.get("ok", True):
        return _fail(r.get("error") or "run failed", "urivision-run",
                     **{k: v for k, v in r.items() if k not in ("ok", "error", "connector")})
    return _ok(action="urivision-run", **{k: v for k, v in r.items() if k not in ("ok", "connector")})


def urirun_bindings() -> dict[str, Any]:
    """Serializable v2 bindings (entry point: urirun.bindings)."""
    return conn.bindings()

@conn.handler("urivision://host/doctor/query/report", isolated=True, meta={"label": "Connector readiness report"})
def doctor() -> dict[str, Any]:
    """Return a safe, read-only connector readiness report for CI smoke tests."""
    return {
        "ok": True,
        "connector": CONNECTOR_ID,
        "version": _connector_version(),
        "status": "ready",
    }


def _connector_version() -> str:
    try:
        from importlib.metadata import version

        return version("urirun-connector-urivision")
    except Exception:
        return "0.1.0"


def connector_manifest() -> dict[str, Any]:
    """Manifest prose + a GENERATED per-URI capability list (URI_COMMAND_STANDARD.md §6)."""
    m = _urirun_compat.load_manifest(__package__) or {}
    try:
        from urirun_connectors_toolkit.connector_sdk import manifest_routes
        m["routes"] = manifest_routes(urirun_bindings())
    except Exception:  # noqa: BLE001 - enrichment; never break the manifest
        pass
    return m


def main(argv: list[str] | None = None) -> int:
    return conn.cli(argv, manifest_prose=_urirun_compat.load_manifest(__package__))


if __name__ == "__main__":
    raise SystemExit(main())
