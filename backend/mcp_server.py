"""fgPROMS MCP server — expose the simulation engine to AI assistants and
other programs via the Model Context Protocol (idea: DWSIM's MCP server).

Run with:  .venv\\Scripts\\python backend\\mcp_server.py   (stdio transport)

Tools exposed:
  * run_model(code)      — parse, analyse and solve an fgPROMS model
  * list_examples()      — the built-in examples and benchmarks
  * get_example(name)    — the source of one example
  * list_projects()      — saved projects in the projects/ folder
  * save_project(name, code) / delete_project(name)
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    # running as a plain script (python backend/mcp_server.py)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from backend import engine
    from backend.app import EXAMPLES, BENCHMARKS, PROJECTS
else:
    from . import engine
    from .app import EXAMPLES, BENCHMARKS, PROJECTS

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("fgPROMS")


@mcp.tool()
def run_model(code: str) -> dict:
    """Run an fgPROMS model (gPROMS-style language: MODEL/VARIABLE/EQUATION,
    $ derivatives, SCHEDULE, OPTIMIZE, PROCESS flowsheets, ...) and return
    the full result: messages, model summary, solved values, residuals and
    (for dynamic runs) the time series."""
    return engine.run_model(code)


@mcp.tool()
def list_examples() -> list:
    """List the built-in example and benchmark models."""
    names = [p.name for p in EXAMPLES.glob("*.gPROMS")]
    names += [p.name for p in BENCHMARKS.glob("*.gPROMS")]
    return sorted(set(names))


@mcp.tool()
def get_example(name: str) -> dict:
    """Return the source code of a built-in example/benchmark."""
    for folder in (EXAMPLES, BENCHMARKS):
        path = folder / name
        if path.is_file():
            return {"name": name,
                    "code": path.read_text(encoding="utf-8",
                                           errors="replace")}
    return {"error": f"example {name!r} not found"}


@mcp.tool()
def list_projects() -> list:
    """List the projects saved in the projects/ folder."""
    PROJECTS.mkdir(exist_ok=True)
    return sorted(p.name for p in PROJECTS.glob("*.gPROMS"))


@mcp.tool()
def save_project(name: str, code: str) -> dict:
    """Save a model into the projects/ folder."""
    if not name.endswith(".gPROMS"):
        name += ".gPROMS"
    PROJECTS.mkdir(exist_ok=True)
    (PROJECTS / name).write_text(code, encoding="utf-8")
    return {"ok": True, "name": name}


@mcp.tool()
def delete_project(name: str) -> dict:
    """Delete a project from the projects/ folder."""
    path = PROJECTS / name
    if path.is_file():
        path.unlink()
    return {"ok": True, "name": name}


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
