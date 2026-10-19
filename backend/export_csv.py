"""CSV export of fgPROMS results (testable, backend-side)."""
from __future__ import annotations

import csv
import io
from typing import Optional


def run_to_csv(results: Optional[dict], time_series: Optional[dict]) -> str:
    """Convert a run result into CSV text.

    Dynamic runs: one row per time point (columns: t, then each variable).
    Steady runs: two columns (variable, value).
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")

    if time_series and time_series.get("variables"):
        names = list(time_series["variables"].keys())
        w.writerow(["t"] + names)
        for i, t in enumerate(time_series["t"]):
            row = [f"{t:.8g}"]
            row += [f"{time_series['variables'][n][i]:.8g}" for n in names]
            w.writerow(row)
    elif results and results.get("values"):
        w.writerow(["variable", "value"])
        for k, v in results["values"].items():
            w.writerow([k, f"{v:.10g}"])
    else:
        w.writerow(["no results"])

    return buf.getvalue()
