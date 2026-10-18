from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import uuid

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from src.agent.graph import ask
from src.agent import graph
from src.agent.schema import get_schema
from src.server.demo import get_guard, BodyLimitMiddleware, browser_session

app = FastAPI(title="AI Agent Chatbot")
logging.basicConfig(level=logging.INFO)
app.add_middleware(BodyLimitMiddleware)

# __file__ = .../src/server/app.py

SRC_DIR = Path(__file__).resolve().parents[1]
UI_DIR = SRC_DIR / "ui"
INDEX_HTML = UI_DIR / "index.html"

app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

@app.middleware("http")
async def session_cookie(request: Request, call_next):
    # Attach the cookie even when admission/provider/validation returns an error.
    pending = Response()
    if request.url.path in {"/", "/ask"} and get_guard() is not None:
        request.state.demo_identity = browser_session(request, pending)
    response = await call_next(request)
    if "set-cookie" in pending.headers:
        response.headers.append("set-cookie", pending.headers["set-cookie"])
    return response


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def non_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Question must not be blank")
        return value


class AskResponse(BaseModel):
    question: str
    sql: str
    answer: str
    image_b64: Optional[str] = None
    cache_hit: bool = False
    cache_source: str = "none"
    attempts: int = 0
    metrics: dict = Field(default_factory=dict)
    request_id: str = ""
    ok: bool = True


@app.get("/")
def index():
    return FileResponse(str(INDEX_HTML))


@app.get("/health")
def health():
    return {"status": "ok"}


def demo_admission(request: Request):
    guard = get_guard()
    if guard is None:
        yield
    else:
        with guard.admit(request.state.demo_identity):
            yield


@app.post("/ask", response_model=AskResponse)
def ask_agent(req: AskRequest, admission=Depends(demo_admission)):
    request_id = uuid.uuid4().hex
    try:
        result = ask(req.question)
    except (FileNotFoundError, ValueError):
        raise HTTPException(503, detail={"message": "Check database and provider configuration.", "request_id": request_id}) from None
    except Exception:
        # Never include provider errors, credentials, prompts or SQL in the HTTP error.
        raise HTTPException(502, detail={"message": "The agent could not complete this request. Please retry.", "request_id": request_id}) from None
    return AskResponse(
        question=req.question,
        sql=result.get("sql", ""),
        answer=result.get("answer", ""),
        image_b64=result.get("image_b64"),
        cache_hit=result.get("cache_hit", False),
        cache_source=result.get("cache_source", "none"),
        attempts=result.get("attempts", 0),
        metrics=result.get("metrics", {}),
        request_id=result.get("request_id", request_id),
        ok=result.get("ok", True),
    )


@app.get("/ready")
def ready():
    try:
        if not get_schema(graph.DB_PATH):
            raise ValueError("No tables")
    except Exception:
        raise HTTPException(503, "Build the database before asking questions.") from None
    return {"status": "ready", "provider_checked": False}
