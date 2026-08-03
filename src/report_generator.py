from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_excel_report(df: pd.DataFrame, output_path: str | Path) -> None:
    df.to_excel(output_path, index=False)


def export_pdf_report(df: pd.DataFrame, output_path: str | Path) -> None:
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("Install reportlab to generate PDF reports.") from exc

    document = SimpleDocTemplate(str(output_path), pagesize=letter)
    style_sheet = getSampleStyleSheet()
    story = [Paragraph("Candidate Recruitment Report", style_sheet["Title"]), Spacer(1, 18)]

    table_data = [["Candidate", "Score", "Skills", "Missing Skills", "Recommendation"]]
    for _, row in df.iterrows():
        table_data.append([
            row["Candidate Name"],
            f"{row['Match Score']}%",
            row["Skills Found"],
            row.get("Missing Skills", ""),
            row.get("Recommendation", ""),
        ])

    table = Table(table_data)
    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ])
    )
    story.append(table)
    document.build(story)
