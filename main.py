from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from pathlib import Path
import os
import subprocess
import textwrap

out_pdf = Path("/mnt/data/plumbing_cold_call_full_script.pdf")
tmp_prefix = "/mnt/data/plumbing_cold_call_full_script"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    name="TitleCustom",
    parent=styles["Title"],
    fontName="Helvetica-Bold",
    fontSize=20,
    leading=24,
    textColor=colors.HexColor("#17324d"),
    spaceAfter=10,
))
styles.add(ParagraphStyle(
    name="Section",
    parent=styles["Heading2"],
    fontName="Helvetica-Bold",
    fontSize=12.5,
    leading=15,
    textColor=colors.HexColor("#17324d"),
    spaceBefore=8,
    spaceAfter=6,
))
styles.add(ParagraphStyle(
    name="BodyCustom",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.8,
    leading=13,
    spaceAfter=5,
))
styles.add(ParagraphStyle(
    name="Small",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=8.8,
    leading=11,
    textColor=colors.HexColor("#334155"),
    spaceAfter=4,
))
styles.add(ParagraphStyle(
    name="MonoLike",
    parent=styles["BodyText"],
    fontName="Helvetica",
    fontSize=9.6,
    leading=12.5,
    backColor=colors.HexColor("#F8FAFC"),
    borderPadding=6,
    borderColor=colors.HexColor("#CBD5E1"),
    borderWidth=0.6,
    borderRadius=4,
    spaceAfter=6,
))

story = []

def p(text, style="BodyCustom"):
    story.append(Paragraph(text, styles[style]))

def box(rows, col_widths=(45*mm, 125*mm)):
    tbl = Table(rows, colWidths=col_widths, hAlign="LEFT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.2),
        ("LEADING", (0, 0), (-1, -1), 12),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6))

def bullet_list(items):
    for item in items:
        story.append(Paragraph(f"• {item}", styles["BodyCustom"]))

def header_footer(c, doc):
    c.saveState()
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.setLineWidth(0.6)
    c.line(doc.leftMargin, A4[1] - 18*mm, A4[0] - doc.rightMargin, A4[1] - 18*mm)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColor(colors.HexColor("#17324d"))
    c.drawString(doc.leftMargin, A4[1] - 14.5*mm, "Plumbing Cold Call Script")
    c.setFont("Helvetica", 8)
    c.setFillColor(colors.HexColor("#475569"))
    c.drawRightString(A4[0] - doc.rightMargin, A4[1] - 14.5*mm, f"Page {doc.page}")
    c.setStrokeColor(colors.HexColor("#CBD5E1"))
    c.line(doc.leftMargin, 14*mm, A4[0] - doc.rightMargin, 14*mm)
    c.restoreState()

doc = SimpleDocTemplate(
    str(out_pdf),
    pagesize=A4,
    leftMargin=18*mm,
    rightMargin=18*mm,
    topMargin=24*mm,
    bottomMargin=20*mm,
)

story.append(Paragraph("Plumbing Cold Call Script", styles["TitleCustom"]))
p("Use this as a live call guide. Do not read it like a paragraph. Follow the sequence. Ask one question at a time. Keep the pace calm and short.")
story.append(Spacer(1, 5))

story.append(Paragraph("1) Call sequence at a glance", styles["Section"]))
box([
    ["Step", "What to do"],
    ["1. Intro", "Open with a short identity line and ask for 20 seconds."],
    ["2. Discovery", "Ask one question about how they handle after-hours calls."],
    ["3. Response", "Use the answer to move into the right branch."],
    ["4. Micro-pitch", "Explain the outcome first, not the tech."],
    ["5. Close", "Ask for a short walkthrough or demo."],
])

story.append(Paragraph("2) Receptionist mode", styles["Section"]))
p("Use these lines when you reach the front desk, assistant, or anyone screening the call.")
box([
    ["They say", "You say"],
    ["What is this regarding?", "It is about missed after-hours plumbing calls and who handles new leads on your side."],
    ["He is not here.", "No problem. What is the best way to reach the person who handles after-hours calls or new leads?"],
    ["Send an email.", "Happy to. What is the best email, and who should I address it to so it reaches the right person?"],
    ["We already have an answering service.", "That makes sense. We usually help when calls are answered, but jobs still slip through or never get booked."],
    ["We are not interested.", "Fair enough. Is that because after-hours calls are already covered, or because it is not a priority right now?"],
    ["He handles that himself.", "Perfect. What is the best time to catch him for a 20-second question?"],
    ["He does not take cold calls.", "Understood. What is the cleanest way to get a quick yes or no on whether missed after-hours calls are an issue?"],
    ["Call back later.", "Sure. What time is usually best when he is not in the middle of a job?"],
    ["How did you get this number?", "This number is listed for the business. We reach plumbing companies that may be missing after-hours leads."],
    ["We do not need help with that.", "Completely fair. Just to confirm, are all your after-hours calls answered and booked already?"],
    ["Is this sales?", "Yes, it is. I will keep it brief and only continue if it is relevant."],
    ["We are too busy.", "That is exactly why I called. Busy plumbing shops usually lose the most calls after hours."],
])

story.append(Paragraph("3) Owner mode", styles["Section"]))
box([
    ["Step", "Line"],
    ["Opener", "Hi John, this is Husain. Quick one - we help plumbing businesses stop losing after-hours calls and turn them into booked jobs. Did I catch you for 20 seconds?"],
    ["If they ask what this is", "Basically, we help plumbers stop losing jobs when calls come in after hours and nobody picks up."],
    ["Discovery question", "How are after-hours calls handled right now?"],
])

story.append(Paragraph("4) Fast branch replies", styles["Section"]))
box([
    ["Their answer", "Your reply"],
    ["Voicemail", "That is exactly where calls slip. Most urgent callers move to the next plumber."],
    ["Live answering service", "Got it. We usually help when calls are answered but not properly qualified or converted."],
    ["We answer everything", "Even weekends and evenings too?"],
    ["Why are you asking?", "Because we help stop missed after-hours jobs and I am checking if that is relevant for you."],
])

story.append(Paragraph("5) Simple pitch", styles["Section"]))
p("We set up an after-hours front desk system that answers naturally, captures details, qualifies urgency, and sends everything to you instantly.")
p("Only mention AI if they ask. The goal is the outcome: answer, qualify, and book the job.")

story.append(Paragraph("6) Objections and exact replies", styles["Section"]))
box([
    ["Objection", "Reply"],
    ["Not interested", "Fair enough. Is that because after-hours calls are already fully covered, or because it is not a priority right now?"],
    ["Send me information", "Sure. What is the best email, and who should I address it to? Also, what time next week is best for a quick follow-up?"],
    ["We already have someone answering calls", "That helps. We usually come in when calls are answered, but jobs still get missed or never booked."],
    ["We do not need more leads", "Understood. Most plumbers do not need more leads - they need to stop losing the ones they already have."],
    ["We are all set", "Good to hear. I only wanted to check whether missed after-hours calls are still leaking revenue."],
    ["Too expensive", "Fair. The real question is whether one or two saved jobs cover it."],
    ["We do not do this", "That is fine. I am just asking how you handle after-hours plumbing calls today."],
    ["Call back another time", "Sure. What time is best?"],
    ["We have voicemail", "Voicemail is common. The problem is that it does not book the job."],
])

story.append(Paragraph("7) Close for the meeting", styles["Section"]))
box([
    ["Close", "Would it be crazy to show you a 10-minute walkthrough this week?"],
    ["Alternative", "Would it make sense to look at what you are losing after hours?"],
    ["Alternative", "If I showed you how many calls are slipping through, would you be open to a quick walkthrough?"],
])

story.append(Paragraph("8) Rules that keep the call strong", styles["Section"]))
bullet_list([
    "Do not explain too much.",
    "Do not pitch AI first.",
    "Ask one question at a time.",
    "Stay calm when blocked - move to the next clean question.",
    "Use the rhythm: permission -> pain question -> outcome -> close.",
    "Never repeat a long line when they say, 'What did you say?' Reframe it in one short sentence.",
])

story.append(Spacer(1, 3))
p("Practice note: this script should feel like a conversation, not a readout. The cleaner your pauses, the stronger it sounds.", "Small")

doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)

# Render to PNGs for inspection
subprocess.run(["bash", "-lc", f"pdftoppm -png {out_pdf} {tmp_prefix}"], check=True)

print(f"Created: {out_pdf}")
print("PNG files:")
for f in sorted(Path("/mnt/data").glob("plumbing_cold_call_full_script-*.png")):
    print(f.name)
