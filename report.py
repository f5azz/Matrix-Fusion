from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

def generate_report(data, filename="report.pdf"):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    # -------- TITLE --------
    content.append(Paragraph("<b>AI Crop Disease Detection Report</b>", styles["Title"]))
    content.append(Spacer(1, 20))

    # -------- FARMER DETAILS --------
    content.append(Paragraph("<b>Farmer Details</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    farmer_table = Table([
        ["Farmer Name:", "__________________________"],
        ["Farmer ID:", "__________________________"],
        ["Location:", data["Location"]],
        ["Date:", "__________________________"]
    ])

    farmer_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    content.append(farmer_table)
    content.append(Spacer(1, 20))

    # -------- ANALYSIS RESULT --------
    content.append(Paragraph("<b>Analysis Result</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    result_table = Table([
        ["Crop", data["Crop"]],
        ["Disease", data["Disease"]],
        ["Confidence", data["Confidence"]],
        ["Severity", data["Severity"]]
    ])

    result_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey)
    ]))

    content.append(result_table)
    content.append(Spacer(1, 20))

    # -------- RECOMMENDATIONS --------
    content.append(Paragraph("<b>Recommended Actions</b>", styles["Heading2"]))
    content.append(Spacer(1, 10))

    for rec in data.get("Recommendations", []):
        content.append(Paragraph(f"• {rec}", styles["Normal"]))

    content.append(Spacer(1, 20))

    # -------- SIGNATURE --------
    content.append(Paragraph("Signature: ____________________", styles["Normal"]))
    content.append(Spacer(1, 10))
    content.append(Paragraph("Authorized by: AI System", styles["Normal"]))

    doc.build(content)

    return filename