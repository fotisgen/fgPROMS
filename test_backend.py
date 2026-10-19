"""Backend smoke tests: run every example through the full pipeline."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.engine import run_model  # noqa: E402

EXAMPLES = ["cstr.gPROMS", "heat_exchanger.gPROMS", "tank.gPROMS", "batch_reactor.gPROMS"]


def main():
    ok_all = True
    for name in EXAMPLES:
        code = (Path("examples") / name).read_text(encoding="utf-8")
        r = run_model(code)
        print("=" * 62)
        print(name)
        for m in r["messages"]:
            print(f"   [{m['level']:>5}] {m['code']:<22} {m.get('args', {})}")
        if r.get("results"):
            res = r["results"]
            print(f"   -> mode={res['mode']} success={res['success']} "
                  f"method={res.get('method')}")
            vals = res.get("values") or {}
            for k, v in vals.items():
                print(f"      {k:<10} = {v:.6g}")
            if not res["success"]:
                ok_all = False
        if r.get("timeSeries"):
            ts = r["timeSeries"]
            n = len(ts["t"])
            print(f"   -> time series: {n} points")
            for k, v in ts["variables"].items():
                print(f"      {k:<10}  t0={v[0]:.6g}  tend={v[-1]:.6g}")
        print()
    print("ALL OK" if ok_all else "SOME FAILURES")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
