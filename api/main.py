"""SJSU IT Service Desk — RAG Engine API

Run with:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, ingest, corpus, departments, admin
from config import PROJECT_ID, LOCATION, DEPARTMENTS, load_departments


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load departments from GCS, then initialise one RAG model per corpus."""
    import vertexai
    from qna import create_rag_model

    print(f"Initialising Vertex AI (project={PROJECT_ID}, location={LOCATION})…")
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    print("Loading departments from GCS…")
    load_departments()

    app.state.rag_models = {}
    for dept_id, dept in DEPARTMENTS.items():
        print(f"  Loading RAG model for dept='{dept_id}'…")
        app.state.rag_models[dept_id] = create_rag_model(
            corpus_name=dept["corpus_name"],
            display_name=dept.get("display_name", dept_id),
            fallback_message=dept.get("fallback_message", "Please contact the relevant department or visit sjsu.edu for help."),
        )

    print("All RAG models ready.")
    yield
    app.state.rag_models = {}


app = FastAPI(
    title="SJSU IT RAG Engine",
    description="RAG-powered Q&A and knowledge-base management API for the SJSU IT Service Desk.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_PREFIX = "/api/v1"
app.include_router(chat.router, prefix=API_PREFIX)
app.include_router(ingest.router, prefix=API_PREFIX)
app.include_router(corpus.router, prefix=API_PREFIX)
app.include_router(departments.router, prefix=API_PREFIX)
app.include_router(admin.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok"}
