"""MCP server tests — M30: handler-level + real stdio end-to-end."""
import asyncio
import sys
from pathlib import Path

import pytest

from conftest import load_model

pytest.importorskip("mcp", reason="mcp SDK not installed")

ROOT = Path(__file__).resolve().parent.parent


def test_handler_run_model():
    from backend.mcp_server import run_model as mcp_run
    r = mcp_run(load_model("cstr.gPROMS", "examples"))
    assert r["ok"] is True
    assert abs(r["results"]["values"]["ca"] - 1.0 / 3.0) < 1e-9


def test_handler_examples():
    from backend.mcp_server import list_examples, get_example
    names = list_examples()
    assert "cstr.gPROMS" in names
    assert "phase_envelope.gPROMS" in names
    body = get_example("cstr.gPROMS")
    assert "MODEL CSTR" in body["code"]


def test_handler_projects():
    from backend.mcp_server import save_project, list_projects, delete_project
    name = "mcp_test_project.gPROMS"
    delete_project(name)
    save_project(name, "MODEL M END")
    assert name in list_projects()
    delete_project(name)
    assert name not in list_projects()


def test_mcp_end_to_end_stdio():
    """Launch the real MCP server as a subprocess and talk to it over the
    stdio transport — a genuine end-to-end protocol test."""
    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
    except ImportError as e:
        pytest.skip(f"mcp client not available: {e}")

    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "backend" / "mcp_server.py")],
            cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {t.name for t in tools.tools}
                assert {"run_model", "list_examples", "get_example",
                        "list_projects", "save_project"} <= names

                res = await session.call_tool(
                    "run_model",
                    {"code": load_model("cstr.gPROMS", "examples")})
                text = res.content[0].text
                assert "CSTR" in text or '"success": true' in text.lower()

                res2 = await session.call_tool("list_examples", {})
                text2 = " ".join(c.text for c in res2.content
                                  if hasattr(c, "text"))
                assert "cstr.gPROMS" in text2

    asyncio.run(go())
