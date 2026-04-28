"""Pydantic request/response models for the RAG Engine API."""

from pydantic import BaseModel, HttpUrl
from typing import Optional


# ─── Departments ─────────────────────────────────────────────────────────────

class DepartmentInfo(BaseModel):
    dept_id: str
    display_name: str


class DepartmentsResponse(BaseModel):
    departments: list[DepartmentInfo]


# ─── Chat ────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    dept_id: str = "it"


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    rewritten_query: Optional[str] = None


class HistoryTurn(BaseModel):
    user: str
    assistant: str


class HistoryResponse(BaseModel):
    session_id: str
    history: list[HistoryTurn]


# ─── Ingestion ────────────────────────────────────────────────────────────────

class IngestURLRequest(BaseModel):
    url: HttpUrl
    dept_id: str = "it"


class IngestURLResponse(BaseModel):
    success: bool
    message: str
    gcs_uri: Optional[str] = None
    title: Optional[str] = None
    word_count: Optional[int] = None


class BatchIngestRequest(BaseModel):
    urls: list[HttpUrl]
    dept_id: str = "it"


class FailedURL(BaseModel):
    url: str
    error: str


class BatchIngestResponse(BaseModel):
    total: int
    successful: list[str]
    failed: list[FailedURL]


# ─── Corpus ──────────────────────────────────────────────────────────────────

class CorpusFile(BaseModel):
    name: str
    display_name: str
    state: str


class ListFilesResponse(BaseModel):
    corpus: str
    files: list[CorpusFile]


class DeleteFileResponse(BaseModel):
    success: bool
    message: str
