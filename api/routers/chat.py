"""Chat endpoints — multi-turn RAG Q&A with per-session conversation memory."""

from fastapi import APIRouter, HTTPException, Request
from api.schemas import ChatRequest, ChatResponse, HistoryResponse, HistoryTurn
from api.state import get_memory, clear_memory
from qna import rewrite_query_with_context
from config import DEPARTMENTS

router = APIRouter(prefix="/chat", tags=["chat"])


def get_rag_model(request: Request, dept_id: str):
    models = request.app.state.rag_models
    if dept_id not in models:
        raise HTTPException(status_code=404, detail=f"Department '{dept_id}' not found.")
    return models[dept_id]


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    """Send a message and get a RAG-powered answer.

    Pass the same `session_id` across calls to maintain conversation context.
    The `dept_id` controls which department corpus is queried.
    """
    model = get_rag_model(req, request.dept_id)
    memory = get_memory(request.session_id)
    original_query = request.message

    standalone_query = rewrite_query_with_context(original_query, memory)
    rewritten = standalone_query if standalone_query != original_query else None

    try:
        response = model.generate_content(standalone_query)
        answer = response.text
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Model error: {exc}")

    memory.add_turn(original_query, answer)

    return ChatResponse(
        session_id=request.session_id,
        answer=answer,
        rewritten_query=rewritten,
    )


@router.get("/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str):
    """Return the full conversation history for a session."""
    memory = get_memory(session_id)
    turns = [HistoryTurn(user=t["user"], assistant=t["assistant"])
             for t in memory.get_history()]
    return HistoryResponse(session_id=session_id, history=turns)


@router.delete("/{session_id}")
async def reset_session(session_id: str):
    """Clear the conversation history for a session and start fresh."""
    clear_memory(session_id)
    return {"session_id": session_id, "message": "Session cleared."}
