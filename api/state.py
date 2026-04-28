"""In-memory session store for per-user conversation history."""

from qna import ConversationMemory

_sessions: dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    _sessions.pop(session_id, None)
