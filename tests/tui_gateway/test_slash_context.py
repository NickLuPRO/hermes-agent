"""/context must inspect the conversation, never the agent-less slash worker."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def conversation(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("context-session", source="tui")
    monkeypatch.setattr(server, "_db", db)
    worker = Mock()
    worker.run.return_value = "(._.) No active agent -- send a message first."
    session = {
        "session_key": "context-session",
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "agent": None,
        "cwd": str(tmp_path),
        "slash_worker": worker,
    }
    monkeypatch.setitem(server._sessions, "context-sid", session)
    yield session, db, worker
    db.close()


def context_output():
    response = server.handle_request({
        "id": "context", "method": "slash.exec",
        "params": {"command": "/context", "session_id": "context-sid"},
    })
    assert "result" in response, response
    return response["result"]["output"]


@pytest.mark.parametrize("estimated", [False, True])
def test_context_reports_local_live_details(conversation, tmp_path, estimated):
    session, db, worker = conversation
    for role in ("system", "user", "assistant", "tool"):
        db.append_message("context-session", role, f"{role} content")
    session["history"] = [{"role": "user", "content": "stale history"}]
    compressor = SimpleNamespace(
        last_prompt_tokens=2500, last_real_prompt_tokens=0 if estimated else 2500,
        context_length=10000, compression_count=2,
    )
    session["agent"] = SimpleNamespace(
        model="conversation-model", context_compressor=compressor,
        _cached_system_prompt="unchanged prompt", platform="tui",
    )
    (tmp_path / "AGENTS.md").write_text("Use focused regression tests.\n")

    output = context_output()

    assert "Conversation: 4 messages" in output
    assert "user: 1, assistant: 1, tool: 1, system: 1" in output
    assert "Model: conversation-model" in output
    assert "Provider: auto" in output
    mark = "~" if estimated else ""
    assert f"Context usage: {mark}2,500 / 10,000 tokens ({mark}25.0%)" in output
    assert "Compressions: 2" in output
    assert "Context files" in output
    assert "✓ AGENTS.md" in output
    assert session["agent"]._cached_system_prompt == "unchanged prompt"
    worker.run.assert_not_called()


@pytest.mark.parametrize("source", ["db", "history", "empty"])
def test_context_without_local_agent_uses_session_transcript(conversation, source):
    session, db, worker = conversation
    if source == "db":
        db.append_message("context-session", "user", "restored question")
    elif source == "history":
        session["history"] = [{"role": "user", "content": "live question"}]

    output = context_output()

    expected = "Conversation is empty (no messages yet)." if source == "empty" else "Conversation: 1 messages"
    assert expected in output
    assert f"user: {0 if source == 'empty' else 1}" in output
    assert "No active agent" not in output
    worker.run.assert_not_called()
