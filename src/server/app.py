from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.agent.graph import ask

app = FastAPI(title="Delivery Cadet AI Agent")

# __file__ = .../src/server/app.py

SRC_DIR = Path(__file__).resolve().parents[1]
UI_DIR = SRC_DIR / "ui"
INDEX_HTML = UI_DIR / "index.html"

app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    question: str
    sql: str
    answer: str
    image_b64: Optional[str] = None


@app.get("/")
def index():
    return FileResponse(str(INDEX_HTML))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_agent(req: AskRequest):
    result = ask(req.question)
    return AskResponse(
        question=req.question,
        sql=result.get("sql", ""),
        answer=result.get("answer", ""),
        image_b64=result.get("image_b64"),
    )
