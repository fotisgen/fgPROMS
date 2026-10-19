"""fgPROMS local web server (FastAPI)."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
EXAMPLES = ROOT / "examples"

app = FastAPI(title="fgPROMS", version="0.1.0")


class RunRequest(BaseModel):
    code: str


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


@app.get("/api/examples")
def list_examples() -> list:
    return sorted(p.name for p in EXAMPLES.glob("*.gPROMS"))


@app.get("/api/examples/{name}")
def get_example(name: str):
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        raise HTTPException(status_code=400, detail="Bad name")
    path = EXAMPLES / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return {"name": name, "code": path.read_text(encoding="utf-8", errors="replace")}


@app.post("/api/run")
def run(req: RunRequest):
    return engine.run_model(req.code)


app.mount("/vendor", StaticFiles(directory=FRONTEND / "vendor"), name="vendor")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
