import io
import json
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors  
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


def generate_report_pdf(report: dict) -> bytes:
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        fontSize=18, textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#555555"),
        spaceAfter=12,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#1a1a2e"),
        spaceBefore=16, spaceAfter=6,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=15,
        textColor=colors.HexColor("#333333"),
    )
    disclaimer_style = ParagraphStyle(
        "Disclaimer", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#888888"),
        spaceBefore=20, borderPad=4,
    )

    story = []

    # ── HEADER ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Medical RAG Assistant", title_style))
    story.append(Paragraph("AI-Generated Medical Report Analysis", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dddddd")))
    story.append(Spacer(1, 12))

    # ── PATIENT INFO ─────────────────────────────────────────────────────────
    patient = report.get("patient_info") or {}
    if isinstance(patient, str):
        try:
            patient = json.loads(patient)
        except Exception:
            patient = {}

    created_at = report.get("created_at", "")
    if hasattr(created_at, "strftime"):
        created_at = created_at.strftime("%d %B %Y, %I:%M %p")

    story.append(Paragraph("Patient Information", section_style))

    patient_data = [
        ["Name", patient.get("name") or "—"],
        ["Age", str(patient.get("age") or "—")],
        ["Sex", patient.get("sex") or "—"],
        ["Report File", report.get("filename") or "—"],
        ["Analysis Date", str(created_at) or "—"],
    ]

    patient_table = Table(patient_data, colWidths=[4 * cm, 13 * cm])
    patient_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4ff")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1a1a2e")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
        ("ROWBACKGROUND", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("PADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 12))

    # ── PARAMETERS TABLE ─────────────────────────────────────────────────────
    thought = report.get("thought", "")
    if thought:
        story.append(Paragraph("Parameters Found", section_style))
        params = _parse_thought(thought)
        if params:
            header = ["Parameter", "Value", "Unit", "Reference Range"]
            rows = [header] + [[p["name"], p["value"], p["unit"], p["ref"]] for p in params]
            param_table = Table(rows, colWidths=[5 * cm, 3 * cm, 3 * cm, 6 * cm])
            param_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("ROWBACKGROUND", (1, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(param_table)
            story.append(Spacer(1, 8))

    # ── ABNORMAL VALUES ───────────────────────────────────────────────────────
    observation = report.get("observation", "")
    if observation:
        story.append(Paragraph("Abnormal Values", section_style))
        obs = _parse_observation(observation)
        if obs:
            header = ["Parameter", "Status", "Possible Condition"]
            rows = [header] + [[o["name"], o["status"], o["cond"]] for o in obs]
            obs_table = Table(rows, colWidths=[5 * cm, 3 * cm, 9 * cm])
            obs_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(obs_table)
            story.append(Spacer(1, 8))

    # ── FULL ANALYSIS ─────────────────────────────────────────────────────────
    analysis = report.get("analysis", "")
    if analysis:
        story.append(Paragraph("Full Analysis", section_style))
        # Split by lines and add each as a paragraph
        for line in analysis.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 4))
                continue
            # Convert markdown headers to bold
            if line.startswith("## "):
                story.append(Paragraph(f"<b>{line[3:]}</b>", body_style))
            elif line.startswith("# "):
                story.append(Paragraph(f"<b>{line[2:]}</b>", body_style))
            elif line.startswith("- ") or line.startswith("* "):
                story.append(Paragraph(f"&bull; {line[2:]}", body_style))
            elif line.startswith("**") and line.endswith("**"):
                story.append(Paragraph(f"<b>{line[2:-2]}</b>", body_style))
            else:
                # Escape special XML chars
                line = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(line, body_style))

    # ── DISCLAIMER ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#dddddd")))
    story.append(Paragraph(
        "Disclaimer: This analysis is generated by an AI system for educational purposes only. "
        "It should not be used as a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider.",
        disclaimer_style
    ))

    doc.build(story)
    return buffer.getvalue()


def _parse_thought(text):
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        param = _extract(line, r"Parameter:\s*([^|]+)")
        val   = _extract(line, r"Value:\s*([^|]+)")
        unit  = _extract(line, r"Unit:\s*([^|]+)")
        ref   = _extract(line, r"Reference:\s*(.+)")
        if param:
            results.append({"name": param, "value": val, "unit": unit, "ref": ref})
    return results


def _parse_observation(text):
    results = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        param  = _extract(line, r"Parameter:\s*([^|]+)")
        status = _extract(line, r"Status:\s*([^|]+)")
        cond   = _extract(line, r"Possible condition:\s*(.+)")
        if param:
            results.append({"name": param, "status": status, "cond": cond})
    return results


def _extract(text, pattern):
    import re
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else ""
