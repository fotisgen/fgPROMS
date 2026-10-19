"""fgPROMS local web server (FastAPI)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import engine
from .export_csv import run_to_csv

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
EXAMPLES = ROOT / "examples"
BENCHMARKS = ROOT / "benchmarks"
PROJECTS = ROOT / "projects"

app = FastAPI(title="fgPROMS", version="0.2.0")


class RunRequest(BaseModel):
    code: str


class ProjectSaveRequest(BaseModel):
    code: str


class ExportRequest(BaseModel):
    results: Optional[dict] = None
    timeSeries: Optional[dict] = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")


def _valid_name(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9_\- .]+\.gPROMS$", name, re.IGNORECASE))


@app.get("/api/examples")
def list_examples() -> list:
    names = [p.name for p in EXAMPLES.glob("*.gPROMS")]
    names += [p.name for p in BENCHMARKS.glob("*.gPROMS")]
    return sorted(set(names))


@app.get("/api/examples/{name}")
def get_example(name: str):
    if not _valid_name(name):
        raise HTTPException(status_code=400, detail="Bad name")
    for folder in (EXAMPLES, BENCHMARKS):
        path = folder / name
        if path.is_file():
            return {"name": name, "code": path.read_text(encoding="utf-8",
                                                         errors="replace")}
    raise HTTPException(status_code=404, detail="Not found")


@app.post("/api/run")
def run(req: RunRequest):
    return engine.run_model(req.code)


@app.post("/api/export")
def export(req: ExportRequest):
    return {"csv": run_to_csv(req.results, req.timeSeries)}


# ---------------- projects ----------------
@app.get("/api/projects")
def list_projects() -> list:
    PROJECTS.mkdir(exist_ok=True)
    return sorted(p.name for p in PROJECTS.glob("*.gPROMS"))


@app.get("/api/projects/{name}")
def get_project(name: str):
    if not _valid_name(name):
        raise HTTPException(status_code=400, detail="Bad name")
    path = PROJECTS / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return {"name": name, "code": path.read_text(encoding="utf-8", errors="replace")}


@app.post("/api/projects/{name}")
def save_project(name: str, req: ProjectSaveRequest):
    if not _valid_name(name):
        raise HTTPException(status_code=400, detail="Bad name")
    PROJECTS.mkdir(exist_ok=True)
    (PROJECTS / name).write_text(req.code, encoding="utf-8")
    return {"ok": True, "name": name}


@app.delete("/api/projects/{name}")
def delete_project(name: str):
    if not _valid_name(name):
        raise HTTPException(status_code=400, detail="Bad name")
    path = PROJECTS / name
    if path.is_file():
        path.unlink()
    return {"ok": True, "name": name}


app.mount("/vendor", StaticFiles(directory=FRONTEND / "vendor"), name="vendor")
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
