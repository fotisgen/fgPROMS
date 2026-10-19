import sys
sys.path.insert(0, ".")
from backend.parser.gproms_parser import parse_model
from backend.analyzer.analyzer import analyze
import backend.solver.solver as S
from pathlib import Path

# monkey-patch solve_step to log
orig_solve_dynamic = S.solve_dynamic

m = parse_model(Path("benchmarks/batch_heater.gPROMS").read_text(encoding="utf-8"))
info = analyze(m)

# instrument: wrap the internal solve_step by patching _bdf_coeffs? Simpler:
# add a print inside solve_dynamic via source? Use a quick copy of the loop.
# Instead: run and print maxOrder + series around 10.8-10.91 with t diffs.
res = S.solve_dynamic(m, info)
ts = res["series"]
t = ts["t"]
print("maxOrder:", res["maxOrder"])
prev = None
for i, tt in enumerate(t):
    if 10.7 < tt < 11.05:
        dt = (tt - prev) if prev is not None else 0
        print(f"  t={tt:.6f} (dt={dt:.6f}) T={ts['variables']['t'][i]:.9f}")
        prev = tt
