"""Ingestion endpoints — scrape URLs and upload files into a department's RAG corpus."""

import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.schemas import (
    IngestURLRequest, IngestURLResponse,
    BatchIngestRequest, BatchIngestResponse, FailedURL,
)
from scraper import ingest_url_to_corpus, ingest_multiple_urls
from config import DEPARTMENTS

router = APIRouter(prefix="/ingest", tags=["ingest"])


def get_corpus_name(dept_id: str) -> str:
    dept = DEPARTMENTS.get(dept_id)
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{dept_id}' not found.")
    return dept["corpus_name"]


@router.post("/url", response_model=IngestURLResponse)
async def ingest_url(request: IngestURLRequest):
    """Scrape a URL, upload to GCS, and import it into the department's RAG corpus."""
    corpus_name = get_corpus_name(request.dept_id)
    result = ingest_url_to_corpus(
        url=str(request.url),
        corpus_name=corpus_name,
        dept_id=request.dept_id,
    )

    if not result["success"]:
        return IngestURLResponse(success=False, message=result["message"])

    scrape = result.get("scrape_result", {})
    return IngestURLResponse(
        success=True,
        message=result["message"],
        gcs_uri=result.get("gcs_uri"),
        title=scrape.get("title"),
        word_count=scrape.get("word_count"),
    )


@router.post("/urls", response_model=BatchIngestResponse)
async def ingest_urls(request: BatchIngestRequest):
    """Batch-scrape multiple URLs and import them all into the department's RAG corpus."""
    corpus_name = get_corpus_name(request.dept_id)
    urls = [str(u) for u in request.urls]
    result = ingest_multiple_urls(
        urls=urls,
        corpus_name=corpus_name,
        dept_id=request.dept_id,
    )

    failed = [FailedURL(url=f["url"], error=f["error"]) for f in result["failed"]]
    return BatchIngestResponse(
        total=len(urls),
        successful=result["successful"],
        failed=failed,
    )


@router.post("/file", response_model=IngestURLResponse)
async def ingest_file(
    file: UploadFile = File(...),
    dept_id: str = Form("it"),
):
    """Upload a PDF (or text) file and import it into the department's RAG corpus."""
    import vertexai
    from vertexai.preview import rag
    from config import PROJECT_ID, LOCATION

    corpus_name = get_corpus_name(dept_id)
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    allowed_types = {"application/pdf", "text/plain"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Only PDF and plain text are accepted.",
        )

    contents = await file.read()
    suffix = ".pdf" if file.content_type == "application/pdf" else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        rag_file = rag.upload_file(
            corpus_name=corpus_name,
            path=tmp_path,
            display_name=file.filename,
            description=f"Uploaded via API — dept: {dept_id}",
        )
        return IngestURLResponse(
            success=True,
            message=f"File '{file.filename}' uploaded to corpus.",
            gcs_uri=rag_file.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Upload failed: {exc}")
    finally:
        os.unlink(tmp_path)
