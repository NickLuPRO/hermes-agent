"""Status distinguishes current context occupancy from cumulative token usage."""

from types import SimpleNamespace

import pytest

from tui_gateway import server


def status_output(monkeypatch, session):
    monkeypatch.setitem(server._sessions, "status-context", session)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    response = server.handle_request({
        "id": "status", "method": "session.status",
        "params": {"session_id": "status-context"},
    })
    assert "result" in response, response
    return response["result"]["output"]


@pytest.mark.parametrize("source", ["live", "restored", "mirrored"])
@pytest.mark.parametrize("estimated", [False, True])
def test_status_uses_conversation_context_snapshot(monkeypatch, source, estimated):
    usage = {"total": 1049498, "context_used": 2500, "context_max": 10000,
             "context_estimated": estimated}
    agent = SimpleNamespace(
        session_total_tokens=usage["total"],
        context_compressor=SimpleNamespace(
            last_prompt_tokens=usage["context_used"],
            last_real_prompt_tokens=0 if estimated else usage["context_used"],
            context_length=usage["context_max"],
        ),
    )
    session = {"agent": agent, "running": False}
    if source != "live":
        session["_metadata_mirror"] = {"usage": usage}
        session["agent"] = None if source == "restored" else SimpleNamespace(session_total_tokens=1)
        session["_compute_host_active"] = source == "mirrored"

    output = status_output(monkeypatch, session)

    mark = "~" if estimated else ""
    assert f"Context usage: {mark}2,500 / 10,000 tokens ({mark}25.0%)" in output
    assert "Tokens: 1,049,498" in output


@pytest.mark.parametrize(("usage", "expected"), [
    ({"context_used": 0, "context_max": 10000}, "0 / 10,000 tokens (0.0%)"),
    ({}, "unavailable"),
    ({"context_max": 10000}, "unavailable"),
    ({"context_used": None, "context_max": 10000}, "unavailable"),
    ({"context_used": 2500, "context_max": 0}, "2,500 tokens (limit unavailable)"),
    ({"context_used": 2500}, "2,500 tokens (limit unavailable)"),
    ({"context_used": 0, "context_max": 0}, "0 tokens (limit unavailable)"),
])
def test_status_context_handles_zero_and_missing_data(monkeypatch, usage, expected):
    output = status_output(monkeypatch, {
        "agent": None, "_metadata_mirror": {"usage": {"total": 1049498, **usage}},
    })

    assert f"Context usage: {expected}" in output
    assert "Tokens: 1,049,498" in output
    assert "nan" not in output.lower()
    assert "inf%" not in output.lower()
