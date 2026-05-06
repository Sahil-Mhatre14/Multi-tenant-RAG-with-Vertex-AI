"""Admin endpoints — department provisioning."""

import re
from fastapi import APIRouter, HTTPException, Request
from vertexai.preview import rag

from api.schemas import CreateDepartmentRequest, CreateDepartmentResponse, UpdateDepartmentRequest
from config import DEPARTMENTS, save_departments
from qna import create_rag_model

router = APIRouter(prefix="/admin", tags=["admin"])

_SLUG_RE = re.compile(r'^[a-z][a-z0-9-]*$')


@router.post("/departments", response_model=CreateDepartmentResponse, status_code=201)
async def create_department(body: CreateDepartmentRequest, req: Request):
    """Create a new department: provisions a Vertex AI RAG corpus, persists it to
    GCS, and hot-reloads the in-memory model registry — no server restart needed.
    """
    if not _SLUG_RE.match(body.dept_id):
        raise HTTPException(
            status_code=400,
            detail="dept_id must be lowercase letters, digits, and hyphens, and must start with a letter.",
        )

    if body.dept_id in DEPARTMENTS:
        raise HTTPException(
            status_code=409,
            detail=f"Department '{body.dept_id}' already exists.",
        )

    if not body.display_name.strip():
        raise HTTPException(status_code=400, detail="display_name cannot be empty.")

    try:
        corpus = rag.create_corpus(display_name=body.display_name.strip())
        corpus_name = corpus.name
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to create Vertex AI corpus: {exc}")

    fallback = (body.fallback_message or "").strip() or \
        f"Please contact the {body.display_name.strip()} department or visit sjsu.edu for help."

    DEPARTMENTS[body.dept_id] = {
        "display_name": body.display_name.strip(),
        "corpus_name": corpus_name,
        "fallback_message": fallback,
    }

    try:
        save_departments()
    except Exception as exc:
        # Roll back the in-memory entry so state stays consistent with GCS
        DEPARTMENTS.pop(body.dept_id, None)
        raise HTTPException(status_code=502, detail=f"Corpus created but failed to persist config: {exc}")

    req.app.state.rag_models[body.dept_id] = create_rag_model(
        corpus_name=corpus_name,
        display_name=body.display_name.strip(),
        fallback_message=fallback,
    )

    return CreateDepartmentResponse(
        dept_id=body.dept_id,
        display_name=body.display_name.strip(),
        corpus_name=corpus_name,
    )


@router.patch("/departments/{dept_id}", response_model=CreateDepartmentResponse)
async def update_department(dept_id: str, body: UpdateDepartmentRequest, req: Request):
    """Update a department's display name and/or fallback message.

    Persists the change to GCS and hot-reloads the department's RAG model so
    the new system instruction takes effect immediately.
    """
    if dept_id not in DEPARTMENTS:
        raise HTTPException(status_code=404, detail=f"Department '{dept_id}' not found.")

    dept = DEPARTMENTS[dept_id]

    if body.display_name is not None:
        dept["display_name"] = body.display_name.strip()

    if body.fallback_message is not None:
        dept["fallback_message"] = body.fallback_message.strip()

    try:
        save_departments()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to persist config: {exc}")

    req.app.state.rag_models[dept_id] = create_rag_model(
        corpus_name=dept["corpus_name"],
        display_name=dept["display_name"],
        fallback_message=dept.get("fallback_message", "Please contact the relevant department or visit sjsu.edu for help."),
    )

    return CreateDepartmentResponse(
        dept_id=dept_id,
        display_name=dept["display_name"],
        corpus_name=dept["corpus_name"],
    )


@router.delete("/departments/{dept_id}")
async def delete_department(dept_id: str, req: Request):
    """Remove a department from the registry.

    Deletes the entry from GCS and unloads its RAG model. The Vertex AI corpus
    itself is NOT deleted — remove it manually from the GCP console if needed.
    """
    if dept_id not in DEPARTMENTS:
        raise HTTPException(status_code=404, detail=f"Department '{dept_id}' not found.")

    DEPARTMENTS.pop(dept_id)

    try:
        save_departments()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to persist config: {exc}")

    req.app.state.rag_models.pop(dept_id, None)

    return {"dept_id": dept_id, "message": f"Department '{dept_id}' deleted."}
