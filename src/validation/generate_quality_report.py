"""
Render reports/data_quality_report.json into a human-readable
reports/Data_Quality_Report.pdf

Usage:
    python src/validation/generate_quality_report.py
"""
import json
import os
import sys
from datetime import datetime

from fpdf import FPDF
from fpdf.enums import XPos, YPos

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ingestion"))
from common import get_logger, PROJECT_ROOT  # noqa: E402

logger = get_logger("validation.report")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")


class QualityReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "RecoMart - Data Quality Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 6, datetime.utcnow().strftime("Generated %Y-%m-%d %H:%M UTC"), new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")


def build_pdf(report: dict, out_path: str):
    pdf = QualityReportPDF()
    pdf.add_page()

    for ds in report.get("datasets", []):
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 20, 90)
        pdf.cell(0, 8, f"Dataset: {ds.get('dataset')}   |   Status: {ds.get('status')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Helvetica", "", 10)

        pdf.cell(0, 6, f"Row count: {ds.get('row_count', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.cell(0, 6, f"Duplicate rows: {ds.get('duplicate_rows', 'N/A')}   "
                        f"Duplicate PKs: {ds.get('duplicate_primary_keys', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        null_pct = ds.get("null_pct_by_column", {})
        if null_pct:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Missing value % by column:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            for col, pct in null_pct.items():
                pdf.cell(0, 5, f"   - {col}: {pct}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        range_violations = ds.get("range_violations", {})
        if range_violations:
            pdf.set_font("Helvetica", "B", 10)
            pdf.cell(0, 6, "Range-check violations:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            for col, count in range_violations.items():
                pdf.cell(0, 5, f"   - {col}: {count} rows out of range", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        issues = ds.get("issues", [])
        if issues:
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(150, 0, 0)
            pdf.cell(0, 6, "Issues:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 9)
            for issue in issues:
                pdf.multi_cell(0, 5, f"   - {issue}")
            pdf.set_text_color(0, 0, 0)

        pdf.ln(4)

    pdf.output(out_path)


def run():
    json_path = os.path.join(REPORTS_DIR, "data_quality_report.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Run validate_data.py first.")
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    out_path = os.path.join(REPORTS_DIR, "Data_Quality_Report.pdf")
    build_pdf(report, out_path)
    logger.info("Data Quality Report PDF written to %s", out_path)
    return out_path


if __name__ == "__main__":
    run()
