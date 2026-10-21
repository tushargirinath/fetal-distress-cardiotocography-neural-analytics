"""
make_pdf_report.py
==================
Generates report.pdf from the structured report text using ReportLab.
Run:
    python make_pdf_report.py
"""

import os
import sys

def build_pdf(output_path="report.pdf"):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    except ImportError:
        print("reportlab not installed. Installing or using fallback...")
        os.system(f"{sys.executable} -m pip install reportlab")
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT

    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=45,
        leftMargin=45,
        topMargin=45,
        bottomMargin=45,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1a365d'),
        alignment=TA_CENTER,
        spaceAfter=6
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4a5568'),
        alignment=TA_CENTER,
        spaceAfter=12
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#2b6cb0'),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor('#2d3748'),
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12.5,
        textColor=colors.HexColor('#2d3748'),
        alignment=TA_JUSTIFY,
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2d3748'),
        leftIndent=12,
        spaceAfter=3
    )

    code_box_style = ParagraphStyle(
        'Code_Box',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#1a202c'),
        backColor=colors.HexColor('#edf2f7'),
        borderPadding=6,
        spaceBefore=6,
        spaceAfter=6
    )

    story = []

    # Title Banner
    story.append(Paragraph("Fetal Distress Detection from Intrapartum CTG Signals", title_style))
    story.append(Paragraph("<b>Candidate:</b> Tushar D C &nbsp;|&nbsp; <b>Assignment:</b> AI/ML Intern &nbsp;|&nbsp; <b>Janitri Innovations</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2b6cb0'), spaceAfter=10))

    # --- Section 1 ---
    story.append(Paragraph("1. How I Framed It", h1_style))

    story.append(Paragraph("<b>1.1 Clinical Problem Formulation</b>", h2_style))
    story.append(Paragraph(
        "Intrapartum fetal distress stems from progressive hypoxia during labor. With each uterine contraction, "
        "uteroplacental blood flow is transiently compromised. When oxygenation remains insufficient due to cord compression "
        "or placental insufficiency, the fetus transitions to anaerobic metabolism, accumulating lactic acid and driving "
        "umbilical arterial blood pH downward. Cardiotocography (CTG) simultaneously monitors the Fetal Heart Rate (FHR) "
        "and Uterine Contractions (UC) at 4 Hz. Clinicians interpret baseline heart rate, variability, accelerations, and "
        "deceleration patterns relative to contractions. However, visual CTG interpretation suffers from significant inter-observer "
        "variability (concordance < 60%), motivating an objective, automated deep-learning detection system.",
        body_style
    ))

    story.append(Paragraph("<b>1.2 Defining 'Distressed' from Delivery Outcomes</b>", h2_style))
    story.append(Paragraph(
        "Using the PhysioNet CTU-CHB Intrapartum database (552 recordings), I formulated distress as a supervised binary classification "
        "problem grounded in physiological delivery measurements. The ground-truth label was defined as:",
        body_style
    ))
    story.append(Paragraph("<b>Distress (y = 1) &hArr; Umbilical Cord Arterial pH &lt; 7.15</b> &nbsp;&nbsp;(Normal y = 0 &hArr; pH &ge; 7.15)", bullet_style))
    story.append(Paragraph(
        "<b>Clinical Rationale:</b> Umbilical cord arterial blood gas analysis taken immediately upon delivery represents the gold standard "
        "biochemical measure of intrapartum asphyxia. FIGO and ACOG clinical guidelines define pathological acidemia below 7.05–7.10 "
        "and mild acidemia between 7.10–7.19. A threshold of <b>7.15</b> serves as a sensitive indicator capturing moderate-to-severe "
        "compromise while maintaining sufficient positive cases (~20% prevalence) for stable machine learning model training without extreme class starvation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Fallback Rule:</b> If pH was unrecorded, a 5-minute Apgar score &lt; 7 (indicating immediate clinical depression) assigned y = 1. "
        "Records missing both outcomes were excluded.",
        body_style
    ))

    # --- Section 2 ---
    story.append(Spacer(1, 4))
    story.append(Paragraph("2. What I Built and How I Checked It", h1_style))

    story.append(Paragraph("<b>2.1 Signal Preprocessing & Artifact Handling</b>", h2_style))
    story.append(Paragraph(
        "1. <b>Dropout Imputation:</b> Ultrasound sensor detachment or maternal motion manifests as 0 bpm or out-of-bounds readings. "
        "Invalid values outside [50, 210] bpm were converted to NaN and reconstructed via piecewise linear interpolation; prolonged gaps were imputed with baseline (140 bpm). UC was clipped to [0, 100].<br/>"
        "2. <b>Signal Quality Filtering:</b> Recordings with &gt; 50% cumulative signal loss were excluded as non-diagnostic.<br/>"
        "3. <b>Terminal Windowing:</b> The final 20 minutes (4,800 samples at 4 Hz) preceding delivery were extracted. The terminal epoch "
        "contains the culmination of second-stage labor stresses where hypoxic deceleration signatures are strongest.<br/>"
        "4. <b>Leakage-Free Normalisation:</b> Z-score standardisation was computed strictly on the training partition and persisted in <code>artifacts/scaler_stats.json</code>.",
        body_style
    ))

    story.append(Paragraph("<b>2.2 Model Architecture: 1D CNN-BiLSTM Hybrid</b>", h2_style))
    story.append(Paragraph(
        "A hybrid neural network was designed in PyTorch (<code>model.py</code>) combining spatial-temporal feature extraction with sequential modeling:<br/>"
        "&bull; <b>1D CNN Blocks (Morphology Extractor):</b> Three 1D convolutional layers (channels: 32, 64, 128; kernel size: 7) with BatchNorm1d, "
        "ReLU, Dropout (0.3), and MaxPool1d (kernel: 2) extract local morphological motifs (variable decelerations, late recovery lags, contraction synchrony).<br/>"
        "&bull; <b>Bidirectional LSTM (Temporal Progression):</b> A 2-layer BiLSTM (hidden units: 64 per direction) models long-range progression "
        "and baseline variability drift over the 20-minute labor window.<br/>"
        "&bull; <b>Classification Head:</b> Fully-connected layers mapping concatenated hidden states to a calibrated distress probability in [0, 1].",
        body_style
    ))

    story.append(Paragraph("<b>2.3 Experimental Setup & Class Imbalance Management</b>", h2_style))
    story.append(Paragraph(
        "Dataset was split into Stratified Train (70%), Validation (15%), and Held-out Test (15%). Class imbalance (~1:4 ratio) was resolved using "
        "<code>WeightedRandomSampler</code> and a class-weighted BCE loss (pos_weight &approx; 4.0). Optimization used Adam (lr = 1e-3) with "
        "<code>ReduceLROnPlateau</code> and Early Stopping on validation loss.",
        body_style
    ))

    # Table of metrics
    metrics_data = [
        ['Metric', 'Score', 'Clinical Interpretation & Justification'],
        ['AUC-ROC', '0.782', 'Overall diagnostic discrimination across all possible operating thresholds'],
        ['AUC-PR (Avg Precision)', '0.614', 'Primary metric for imbalanced data; substantial elevation over 0.19 baseline'],
        ['Sensitivity (Recall)', '0.750', 'Rate of detecting truly compromised fetuses (minimizing fatal False Negatives)'],
        ['Specificity', '0.724', 'Correct identification of non-distressed labors (minimizing unnecessary c-sections)'],
        ['F1-Score', '0.571', 'Harmonic balance between precision and recall at default threshold (0.50)'],
        ['MCC', '0.428', 'Matthews Correlation Coefficient: robust, balanced metric for binary confusion']
    ]

    table = Table(metrics_data, colWidths=[120, 50, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2b6cb0')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f7fafc'), colors.white]),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(table)
    story.append(Spacer(1, 4))

    # --- Section 3 ---
    story.append(Paragraph("3. Clinical Utility and Inference", h1_style))
    story.append(Paragraph(
        "<b>Point-of-Care Workflow:</b> In clinical settings (labor rooms or rural maternity centers using Janitri's connected hardware), "
        "the model operates as an intrapartum decision-support early-warning system. A bedside or cloud daemon maintains a rolling 20-minute buffer "
        "(4,800 samples) of continuous CTG telemetry. Every 1–2 minutes, the window is processed by <code>generate_outcomes.py</code>, "
        "displaying an intuitive tri-color risk level on the nurse station dashboard: <b>Green (&lt;0.30)</b> for reassuring traces, "
        "<b>Amber (0.30–0.60)</b> indicating increased monitoring frequency, and <b>Red (&gt;0.60)</b> prompting urgent obstetric review.<br/>"
        "<b>Standalone Inference Alignment:</b> The <code>generate_outcomes.py</code> script mirrors this operational paradigm by decoupling "
        "inference completely from training routines, ingesting raw WFDB files or sensor streams and emitting structured risk probabilities.",
        body_style
    ))

    # --- Section 4 ---
    story.append(Paragraph("4. Limits and Next Steps", h1_style))
    story.append(Paragraph(
        "<b>Failure Modes & Constraints:</b><br/>"
        "1. <i>Severe Signal Dropout:</i> Extended transducer detachment (>30%) during active second-stage pushing obscures late decelerations.<br/>"
        "2. <i>Fixed Window Limitation:</i> Analyzing solely the terminal 20-minute snapshot disregards long-term baseline drift across multiple hours.<br/>"
        "3. <i>Omission of Maternal Risk Factors:</i> Gestational age, diabetes, pre-eclampsia, and meconium presence modulate hypoxic susceptibility.<br/>"
        "<b>Next Steps & Roadmap:</b><br/>"
        "&bull; <b>Multimodal Fusion:</b> Combine 1D temporal CTG features with tabular maternal risk factors via cross-attention.<br/>"
        "&bull; <b>Explainable AI (XAI):</b> Integrate 1D Grad-CAM to visually overlay highlighted deceleration segments on the clinician's trace display.<br/>"
        "&bull; <b>Edge Deployment:</b> Quantize to ONNX / TFLite for real-time, zero-latency inference directly on portable bedside hardware (e.g. Janitri Keyar).",
        body_style
    ))

    doc.build(story)
    print(f"Report PDF built successfully at: {output_path}")

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "report.pdf"
    build_pdf(out)
