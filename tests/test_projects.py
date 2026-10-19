"""Project storage + CSV export API tests."""
from fastapi.testclient import TestClient

from backend.app import app
from conftest import load_model

client = TestClient(app)

NAME = "test_project_xyz.gPROMS"


def _cleanup():
    client.delete(f"/api/projects/{NAME}")


def test_save_and_load_roundtrip():
    _cleanup()
    code = load_model("cstr.gPROMS", "examples")
    r = client.post(f"/api/projects/{NAME}", json={"code": code})
    assert r.status_code == 200 and r.json()["ok"] is True

    names = client.get("/api/projects").json()
    assert NAME in names

    r = client.get(f"/api/projects/{NAME}")
    assert r.status_code == 200
    assert r.json()["code"] == code

    _cleanup()
    assert NAME not in client.get("/api/projects").json()


def test_save_overwrite():
    _cleanup()
    client.post(f"/api/projects/{NAME}", json={"code": "MODEL A END"})
    client.post(f"/api/projects/{NAME}", json={"code": "MODEL B END"})
    r = client.get(f"/api/projects/{NAME}")
    assert r.json()["code"] == "MODEL B END"
    _cleanup()


def test_delete_missing_ok():
    client.delete(f"/api/projects/{NAME}")
    r = client.delete(f"/api/projects/{NAME}")
    assert r.status_code == 200


def test_bad_project_name_rejected():
    r = client.get("/api/projects/..%2F..%2Frun.bat")
    assert r.status_code in (400, 404)
    r = client.post("/api/projects/..%2F..%2Fx.gPROMS", json={"code": "x"})
    assert r.status_code in (400, 404)   # rejected or routed away — never written
    r = client.delete("/api/projects/..%2F..%2Fx.gPROMS")
    assert r.status_code in (400, 404)


def test_export_csv_steady():
    code = load_model("cstr.gPROMS", "examples")
    run = client.post("/api/run", json={"code": code}).json()
    r = client.post("/api/export", json={
        "results": run["results"], "timeSeries": run["timeSeries"]})
    csv_text = r.json()["csv"]
    lines = csv_text.strip().split("\n")
    assert lines[0] == "variable,value"
    assert len(lines) == 4  # header + 3 variables
    assert "fout,1" in csv_text
    assert "ca,0.3333333333" in csv_text
    assert "r,0.1666666667" in csv_text


def test_export_csv_dynamic():
    code = load_model("tank.gPROMS", "examples")
    run = client.post("/api/run", json={"code": code}).json()
    r = client.post("/api/export", json={
        "results": run["results"], "timeSeries": run["timeSeries"]})
    csv_text = r.json()["csv"]
    lines = csv_text.strip().split("\n")
    assert lines[0] == "t,h,c,fout"
    n = len(run["timeSeries"]["t"])
    assert len(lines) == n + 1   # header + one row per time point
    t0, h0 = lines[1].split(",")[:2]
    assert float(t0) == 0.0
    assert abs(float(h0) - 1.0) < 1e-9
    # last row matches the final state (CSV keeps 8 significant digits)
    t_last, h_last = lines[-1].split(",")[:2]
    assert abs(float(h_last) - run["timeSeries"]["variables"]["h"][-1]) < 1e-6


def test_export_empty_results():
    r = client.post("/api/export", json={"results": None, "timeSeries": None})
    assert r.status_code == 200
    assert r.json()["csv"] == "no results\n"


def test_examples_include_benchmarks():
    names = client.get("/api/examples").json()
    assert "exp_decay.gPROMS" in names
    assert "van_der_pol.gPROMS" in names
    assert "rachford_rice.gPROMS" in names
    r = client.get("/api/examples/exp_decay.gPROMS")
    assert r.status_code == 200
    assert "MODEL ExpDecay" in r.json()["code"]
