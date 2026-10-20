"""Excel (.xlsx) export of fgPROMS results via openpyxl (BSD)."""
from __future__ import annotations

import io
from typing import Optional


def run_to_xlsx(results: Optional[dict], time_series: Optional[dict],
                messages: Optional[list] = None) -> bytes:
    """Convert a run result into an .xlsx workbook (bytes).

    * 'Results' sheet: variable / value (steady) — or the whole time
      series grid (dynamic).
    * 'Messages' sheet: solver/analyser messages.
    """
    from openpyxl import Workbook

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"

    if time_series and time_series.get("variables"):
        names = list(time_series["variables"].keys())
        ws.append(["t"] + names)
        for i, t in enumerate(time_series["t"]):
            ws.append([float(t)]
                      + [float(time_series["variables"][n][i])
                         for n in names])
        if results and results.get("values"):
            ws.append([])
            ws.append(["final values:"])
            for k, v in results["values"].items():
                ws.append([k, float(v)])
    elif results and results.get("values"):
        ws.append(["variable", "value"])
        for k, v in results["values"].items():
            ws.append([k, float(v)])
    else:
        ws.append(["no results"])

    ws2 = wb.create_sheet("Messages")
    ws2.append(["level", "code", "details"])
    for m in (messages or []):
        ws2.append([m.get("level", ""), m.get("code", ""),
                    str(m.get("args", ""))])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()
