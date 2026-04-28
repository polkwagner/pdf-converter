"""Build tests/fixtures/sample.pdf — a 3-page PDF with a Roman-numeral page label
on page 1 (front matter), a regular page, and a page containing a simple table.

Run from repo root:
    ./venv/bin/python tests/fixtures/build/build_pdf.py
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    PageBreak,
    Table,
    TableStyle,
    Spacer,
)

OUT = Path(__file__).resolve().parent.parent / "sample.pdf"


def build() -> Path:
    doc = SimpleDocTemplate(str(OUT), pagesize=LETTER, title="Sample Fixture")
    styles = getSampleStyleSheet()
    story = []

    # Page 1 — front matter heading; we set page label "i" via a low-level hack below
    story.append(Paragraph("Preface", styles["Title"]))
    story.append(Paragraph(
        "This preface page is intended to test page-label translation: the visible "
        "page label should be 'i' (Roman numeral), even though the page index is 0.",
        styles["BodyText"],
    ))
    story.append(PageBreak())

    # Page 2 — body
    story.append(Paragraph("Chapter 1: Introduction", styles["Heading1"]))
    story.append(Paragraph(
        "This second page contains body content. Position markers should label "
        "this as Page 1 if Roman-numeral front matter is offset correctly.",
        styles["BodyText"],
    ))
    story.append(PageBreak())

    # Page 3 — table
    story.append(Paragraph("Chapter 2: Table Test", styles["Heading1"]))
    story.append(Spacer(1, 12))
    data = [
        ["Column A", "Column B", "Column C"],
        ["alpha", "beta", "gamma"],
        ["delta", "epsilon", "zeta"],
    ]
    t = Table(data)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ]))
    story.append(t)

    doc.build(story)

    # ReportLab does not expose page-label dicts via SimpleDocTemplate. Patch via PyMuPDF.
    import fitz
    pdf = fitz.open(str(OUT))
    pdf.set_page_labels([
        {"startpage": 0, "style": "r", "prefix": "", "firstpagenum": 1},  # i, ii (Roman)
        {"startpage": 1, "style": "D", "prefix": "", "firstpagenum": 1},  # 1, 2 (Decimal)
    ])
    pdf.saveIncr()
    pdf.close()

    return OUT


if __name__ == "__main__":
    p = build()
    size_kb = p.stat().st_size / 1024
    print(f"Wrote {p} ({size_kb:.1f} KB)")
