"""PDF report export of fgPROMS results via reportlab (BSD)."""
from __future__ import annotations

import io
from typing import Optional


def run_to_pdf(results: Optional[dict], time_series: Optional[dict],
               model: Optional[dict] = None, messages: Optional[list] = None,
               title: str = "fgPROMS report") -> bytes:
    """Build a simple PDF report: model summary, parameter values,
    results table and messages."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    if model:
        story.append(Paragraph(f"Model: {model.get('name', '—')}",
                               styles["Heading2"]))
        rows = [["Parameter", "Type", "Value"]]
        for p in model.get("parameters", []):
            rows.append([str(p.get("name", "")), str(p.get("type", "")),
                         f"{p.get('value', ''):.8g}"
                         if isinstance(p.get("value"), (int, float))
                         else str(p.get("value", ""))])
        t = Table(rows, colWidths=[5 * cm, 5 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))

    if results:
        story.append(Paragraph("Results", styles["Heading2"]))
        rows = [["Variable", "Value"]]
        for k, v in (results.get("values") or {}).items():
            rows.append([str(k), f"{v:.10g}"])
        t = Table(rows, colWidths=[8 * cm, 8 * cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4 * cm))
        if results.get("objective"):
            story.append(Paragraph(
                f"Objective ({results['objective'].get('kind', '')}): "
                f"{results['objective'].get('value')}", styles["Normal"]))
        if results.get("maxResidual") is not None:
            story.append(Paragraph(
                f"Max residual: {results['maxResidual']:.3e}",
                styles["Normal"]))

    if messages:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("Messages", styles["Heading2"]))
        for m in messages:
            story.append(Paragraph(
                f"<b>{m.get('level', '')}</b> {m.get('code', '')} — "
                f"{m.get('args', '')}", styles["BodyText"]))

    doc.build(story)
    return buf.getvalue()
