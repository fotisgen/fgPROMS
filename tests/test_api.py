"""API tests via FastAPI TestClient (no real server needed)."""
from fastapi.testclient import TestClient

from backend.app import app
from conftest import load_model

client = TestClient(app)


def test_index_serves_app():
    r = client.get("/")
    assert r.status_code == 200
    assert "fgPROMS" in r.text


def test_examples_listed():
    r = client.get("/api/examples")
    assert r.status_code == 200
    names = r.json()
    assert "cstr.gPROMS" in names
    assert "tank.gPROMS" in names
    assert "heat_exchanger.gPROMS" in names


def test_example_content():
    r = client.get("/api/examples/cstr.gPROMS")
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "cstr.gPROMS"
    assert "MODEL CSTR" in body["code"]


def test_example_traversal_rejected():
    r = client.get("/api/examples/..%2F..%2Frun.bat")
    assert r.status_code in (400, 404)


def test_run_steady():
    r = client.post("/api/run", json={"code": load_model("cstr.gPROMS", "examples")})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["results"]["mode"] == "steady"
    assert body["results"]["success"] is True
    assert abs(body["results"]["values"]["ca"] - 1.0 / 3.0) < 1e-9


def test_run_dynamic_returns_series():
    r = client.post("/api/run", json={"code": load_model("tank.gPROMS", "examples")})
    body = r.json()
    assert body["results"]["mode"] == "dynamic"
    assert body["results"]["success"] is True
    assert body["timeSeries"] is not None
    assert len(body["timeSeries"]["t"]) == 101


def test_run_syntax_error():
    r = client.post("/api/run", json={"code": "MODEL X"})
    body = r.json()
    assert body["ok"] is False
    assert body["messages"][0]["code"] == "parse_error"


def test_run_empty_code():
    r = client.post("/api/run", json={"code": ""})
    body = r.json()
    assert body["messages"][0]["code"] == "err_empty"


def test_static_assets_served():
    for path in ("/static/app.js", "/static/style.css", "/static/i18n.js",
                 "/vendor/codemirror.min.js"):
        r = client.get(path)
        assert r.status_code == 200
