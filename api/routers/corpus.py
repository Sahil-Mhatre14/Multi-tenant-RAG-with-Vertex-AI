"""Corpus management endpoints — list and delete files per department corpus."""

from fastapi import APIRouter, HTTPException
from api.schemas import ListFilesResponse, CorpusFile, DeleteFileResponse
from config import DEPARTMENTS

from vertexai.preview import rag

router = APIRouter(prefix="/corpus", tags=["corpus"])


def get_corpus_name(dept_id: str) -> str:
    dept = DEPARTMENTS.get(dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{dept_id}' not found.")
    return dept["corpus_name"]


@router.get("/{dept_id}/files", response_model=ListFilesResponse)
async def list_files(dept_id: str):
    """List all files currently indexed in the given department's RAG corpus."""
    corpus_name = get_corpus_name(dept_id)
    try:
        files = list(rag.list_files(corpus_name=corpus_name))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to list files: {exc}")

    corpus_files = []
    for f in files:
        state = (
            f.file_status.state.name
            if hasattr(f, "file_status") and f.file_status
            else "UNKNOWN"
        )
        corpus_files.append(CorpusFile(
            name=f.name,
            display_name=f.display_name or "",
            state=state,
        ))

    return ListFilesResponse(corpus=corpus_name, files=corpus_files)


@router.delete("/{dept_id}/files/{file_id}", response_model=DeleteFileResponse)
async def delete_file(dept_id: str, file_id: str):
    """Delete a file from the department's RAG corpus by its resource ID.

    `file_id` is the last segment of the resource name returned by `list_files`.
    """
    corpus_name = get_corpus_name(dept_id)
    file_name = f"{corpus_name}/ragFiles/{file_id}"
    try:
        rag.delete_file(name=file_name)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Delete failed: {exc}")

    return DeleteFileResponse(
        success=True,
        message=f"File '{file_id}' deleted from corpus.",
    )
