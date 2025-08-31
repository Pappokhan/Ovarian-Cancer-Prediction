import streamlit as st
import joblib
import numpy as np
import time, os, csv
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# --- Utility functions ---
def now_ms():
    return time.perf_counter() * 1000


def log_latency(times):
    os.makedirs("logs", exist_ok=True)
    path = "logs/latency_log.csv"
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=times.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(times)


# --- Function to generate PDF report using reportlab ---
def generate_pdf_report(input_data, prediction, probability, times):
    biomarkers = [
        "AFP (ng/mL)", "Age (years)", "Albumin (g/L)", "Alkaline Phosphatase (U/L)", "AST (U/L)",
        "CA125 (U/mL)", "CA19-9 (U/mL)", "CEA (ng/mL)", "CO2 Combining Power (mmol/L)", "Globulin (g/L)",
        "HE4 (pmol/L)", "Indirect Bilirubin (µmol/L)", "Lymphocyte Count (10⁹/L)", "Lymphocyte %",
        "Mean Corpuscular Hemoglobin (pg)", "Mean Platelet Volume (fL)", "Neutrophil Count (10⁹/L)",
        "Plateletcrit (%)", "Platelet Count (10⁹/L)", "Total Bilirubin (µmol/L)"
    ]

    # Categorize risk
    if probability < 0.3:
        risk_category = "Low Risk"
        risk_message = "Low risk of Ovarian Cancer. Regular monitoring is recommended."
    elif probability < 0.7:
        risk_category = "Moderate Risk"
        risk_message = "Moderate risk of Ovarian Cancer. Consider further monitoring."
    else:
        risk_category = "High Risk"
        risk_message = "High risk of Ovarian Cancer. Please consult a specialist."

    # Create PDF
    pdf_file = "report.pdf"
    doc = SimpleDocTemplate(pdf_file, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("OvaSignal AI: Ovarian Cancer Risk Prediction Report", styles['Title']))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Biomarker Table
    elements.append(Paragraph("Patient Biomarker Inputs", styles['Heading2']))
    biomarker_data = [["Biomarker", "Value"]] + [[b, f"{v:.2f}"] for b, v in zip(biomarkers, input_data)]
    table = Table(biomarker_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 12))

    # Prediction Result
    elements.append(Paragraph("Prediction Result", styles['Heading2']))
    elements.append(Paragraph(f"<b>Risk Category:</b> {risk_category}", styles['Normal']))
    elements.append(Paragraph(f"<b>Probability of Ovarian Cancer:</b> {probability * 100:.2f}%", styles['Normal']))
    elements.append(
        Paragraph(f"<b>Probability of No Ovarian Cancer:</b> {(1 - probability) * 100:.2f}%", styles['Normal']))
    elements.append(Paragraph(f"<b>Recommendation:</b> {risk_message}", styles['Normal']))
    elements.append(Spacer(1, 12))

    # Build PDF
    doc.build(elements)
    return pdf_file


# --- Load trained model ---
model = joblib.load("initial_svm_model.pkl")


# --- Prediction function with timing ---
def predict(input_data):
    t0 = now_ms()
    X = np.array([input_data])
    t1 = now_ms()
    prediction = model.predict(X)
    t2 = now_ms()
    probability = model.predict_proba(X)
    t3 = now_ms()

    times = {
        "preprocess_ms": round(t1 - t0, 2),
        "inference_ms": round(t2 - t1, 2),
        "probability_ms": round(t3 - t2, 2),
        "total_ms": round(t3 - t0, 2)
    }
    return prediction[0], probability[0][1], times


# --- Streamlit UI ---
st.set_page_config(page_title="OvaSignal AI: Ovarian Cancer Predictor", layout="wide")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .block-container { text-align: center; }
    h1, h2, h3, h4, h5, h6, p { text-align: center; }
    .stButton>button { display: block; margin: auto; }
    .stDownloadButton { display: flex; justify-content: center; }
    .stDownloadButton>button { 
        display: inline-block; 
        margin: auto; 
        padding: 0.5rem 1rem; 
        font-size: 1rem; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("OvaSignal AI: Ovarian Cancer Risk Prediction")

st.markdown("""
OvaSignal AI is an **AI-powered clinical decision support tool** that predicts the risk of  
**Ovarian Cancer** using blood biomarkers and baseline health parameters.
""")

# --- Input fields with 4 columns ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    AFP = st.number_input("AFP (ng/mL)", 0.0, 500.0, 5.0, 0.1, format="%.1f")
    Age = st.number_input("Age (years)", 18, 100, 40, 1)
    ALB = st.number_input("Albumin (g/L)", 10.0, 60.0, 40.0, 0.1, format="%.1f")
    ALP = st.number_input("Alkaline Phosphatase (U/L)", 20.0, 400.0, 100.0, 1.0, format="%.1f")
    AST = st.number_input("AST (U/L)", 5.0, 200.0, 25.0, 1.0, format="%.1f")

with col2:
    CA125 = st.number_input("CA125 (U/mL)", 0.0, 2000.0, 35.0, 1.0, format="%.1f")
    CA199 = st.number_input("CA19-9 (U/mL)", 0.0, 2000.0, 20.0, 1.0, format="%.1f")
    CEA = st.number_input("CEA (ng/mL)", 0.0, 200.0, 2.5, 0.1, format="%.1f")
    CO2CP = st.number_input("CO₂ Combining Power (mmol/L)", 10.0, 40.0, 22.0, 0.1, format="%.1f")
    GLO = st.number_input("Globulin (g/L)", 10.0, 50.0, 30.0, 0.1, format="%.1f")

with col3:
    HE4 = st.number_input("HE4 (pmol/L)", 0.0, 1000.0, 60.0, 1.0, format="%.1f")
    IBIL = st.number_input("Indirect Bilirubin (µmol/L)", 0.0, 50.0, 10.0, 0.1, format="%.1f")
    LYM_num = st.number_input("Lymphocyte Count (10⁹/L)", 0.0, 10.0, 2.0, 0.1, format="%.1f")
    LYM_percent = st.number_input("Lymphocyte %", 0.0, 100.0, 30.0, 0.1, format="%.1f")
    MCH = st.number_input("Mean Corpuscular Hemoglobin (pg)", 20.0, 40.0, 30.0, 0.1, format="%.1f")

with col4:
    MPV = st.number_input("Mean Platelet Volume (fL)", 5.0, 15.0, 10.0, 0.1, format="%.1f")
    NEU = st.number_input("Neutrophil Count (10⁹/L)", 0.0, 20.0, 4.0, 0.1, format="%.1f")
    PCT = st.number_input("Plateletcrit (%)", 0.0, 1.0, 0.2, 0.01, format="%.2f")
    PLT = st.number_input("Platelet Count (10⁹/L)", 50.0, 1000.0, 250.0, 1.0, format="%.1f")
    TBIL = st.number_input("Total Bilirubin (µmol/L)", 0.0, 50.0, 15.0, 0.1, format="%.1f")

# --- Collect input data ---
input_data = [
    AFP, Age, ALB, ALP, AST, CA125, CA199, CEA, CO2CP,
    GLO, HE4, IBIL, LYM_num, LYM_percent, MCH, MPV, NEU, PCT, PLT, TBIL
]

# --- Prediction Button ---
if st.button("Predict"):
    prediction, probability, times = predict(input_data)
    log_latency(times)

    st.subheader("Prediction Result")

    # Smart categorization
    if probability < 0.3:
        st.success(f"✅ **Low Risk** of Ovarian Cancer ({probability * 100:.1f}%).")
    elif probability < 0.7:
        st.warning(f"⚠️ **Moderate Risk** of Ovarian Cancer ({probability * 100:.1f}%). Consider further monitoring.")
    else:
        st.error(f"🚨 **High Risk** of Ovarian Cancer ({probability * 100:.1f}%). Please consult a specialist.")

    # Probability progress bars
    st.subheader("Prediction Probabilities")
    st.write("**Ovarian Cancer Risk**")
    st.progress(float(probability))
    st.write(f"{probability * 100:.2f}%")

    st.write("**No Ovarian Cancer**")
    st.progress(float(1 - probability))
    st.write(f"{(1 - probability) * 100:.2f}%")

    # --- Generate and provide download link for PDF ---
    st.subheader("Download Report")
    try:
        pdf_file = generate_pdf_report(input_data, prediction, probability, times)
        with open(pdf_file, "rb") as f:
            st.download_button(
                label="Download Prediction Report (PDF)",
                data=f,
                file_name="OvaAI_Prediction_Report.pdf",
                mime="application/pdf"
            )
        # Clean up temporary file
        if os.path.exists(pdf_file):
            os.remove(pdf_file)
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")