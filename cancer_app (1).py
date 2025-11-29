import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import base64

st.set_page_config(page_title="AeroLung AI • Lung Cancer Risk Prediction", page_icon="Lungs", layout="centered", initial_sidebar_state="collapsed")

st.markdown("""
<style>
    .main {background: linear-gradient(to bottom, #f0f9ff, #e0f2fe);}
    .header-box {background: linear-gradient(135deg, #1e40af, #2563eb); padding: 2.5rem; border-radius: 20px; text-align: center; color: white; box-shadow: 0 15px 35px rgba(37,99,235,0.3); margin-bottom: 2rem;}
    .feature-card {background: white; padding: 1.8rem; border-radius: 16px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); border: 1px solid #e0e0e0; height: 100%;}
    .result-box {background: white; padding: 2rem; border-radius: 20px; text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.12); border-left: 10px solid #dc2626; margin: 2rem 0;}
    .stButton > button {background: #1e40af; color: white; font-weight: bold; border-radius: 12px; padding: 0.8rem 2rem; font-size: 1.1rem; border: none; box-shadow: 0 4px 15px rgba(30,64,175,0.4);}
    .stButton > button:hover {background: #1d4ed8;}
    h1 {font-size: 3rem !important; margin: 0;}
    h3 {font-size: 1.4rem; opacity: 0.9; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('LungCancer_Stacking_100.pkl')
model = load_model()

st.markdown("""
<div class="header-box">
    <h1>AeroLung AI</h1>
    <h3>Lung Cancer Risk Prediction System</h3>
    <p>Clinical-Grade Stacking Ensemble • Calibrated Probability Output</p>
</div>
""", unsafe_allow_html=True)

st.markdown("### Patient Information & Risk Factors")
col1, col2 = st.columns(2)
with col1:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.subheader("Lifestyle & Symptoms")
    smoking = st.selectbox("Smoking Status", ["Never", "Former", "Current"], index=2)
    breathing_issue = st.selectbox("Chronic Breathing Difficulty", ["No", "Yes"], index=1)
    throat_discomfort = st.selectbox("Frequent Throat Irritation", ["No", "Yes"], index=1)
    pollution = st.selectbox("High Pollution Exposure", ["No", "Yes"], index=1)
    family_cancer = st.selectbox("Family History of Lung Cancer", ["No", "Yes"], index=1)
    family_smoking = st.selectbox("Family History of Heavy Smoking", ["No", "Yes"], index=True)
    st.markdown("</div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.subheader("Health & Demographics")
    age = st.slider("Age", 18, 100, 68)
    energy = st.slider("Energy Level (1=Low → 10=High)", 1, 10, 4)
    immunity = st.slider("Immune Health (1=Poor → 10=Excellent)", 1, 10, 3)
    spo2 = st.slider("Resting SpO₂ (%)", 85, 100, 91)
    st.markdown("</div>", unsafe_allow_html=True)

if st.button("Calculate Risk", type="primary", use_container_width=True):
    with st.spinner("Calculating..."):
        features = np.array([[
            ["Never", "Former", "Current"].index(smoking),
            ["No", "Yes"].index(breathing_issue),
            ["No", "Yes"].index(throat_discomfort),
            ["No", "Yes"].index(family_smoking),
            energy, immunity, spo2,
            ["No", "Yes"].index(family_cancer),
            ["No", "Yes"].index(pollution),
            age
        ]])
        prob = model.predict_proba(features)[0][1]
        risk_pct = round(prob * 100, 1)

        if prob < 0.3:
            level, color, advice = "Low Risk", "#16a34a", "Continue healthy habits and annual check-up."
        elif prob < 0.7:
            level, color, advice = "Moderate Risk", "#ca8a04", "Low-dose CT scan + pulmonologist visit recommended."
        else:
            level, color, advice = "High Risk", "#dc2626", "Urgent: Chest CT and oncology referral required."

        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=risk_pct,
            title={'text': "Lung Cancer Risk", 'font': {'size': 20}},
            number={'suffix': "%", 'font': {'size': 50}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 30], 'color': '#dcfce7'},
                    {'range': [30, 70], 'color': '#fef9c3'},
                    {'range': [70,100], 'color': '#fecaca'}
                ],
                'threshold': {'line': {'color': "red", 'width': 6}, 'value': 70}
            }))
        fig.update_layout(height=400, margin=dict(t=60, b=20, l=30, r=30))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:{color}; margin:0;">{level}</h2>
            <h3>Probability: <strong>{prob:.1%}</strong></h3>
            <p><strong>Recommendation:</strong><br>{advice}</p>
        </div>
        """, unsafe_allow_html=True)

        st.session_state.result = {
            "prob": prob, "level": level, "color": color, "advice": advice,
            "fig": fig,
            "inputs": {"Age": age, "Smoking": smoking, "Breathing Issue": breathing_issue,
                       "Throat Irritation": throat_discomfort, "Pollution Exposure": pollution,
                       "Family Lung Cancer": family_cancer, "Family Smoking": family_smoking,
                       "Energy Level": energy, "Immune Health": immunity, "SpO₂ (%)": spo2}
        }

def create_beautiful_pdf():
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=1.5*cm, rightMargin=1.5*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story = []

    # Header
    story.append(Paragraph("AeroLung AI", ParagraphStyle(name="Header", fontSize=28, textColor=colors.HexColor("#1e40af"), spaceAfter=10, alignment=1)))
    story.append(Paragraph("Lung Cancer Risk Report", ParagraphStyle(name="Sub", fontSize=16, textColor=colors.HexColor("#2563eb"), alignment=1)))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y • %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 0.8*cm))

    # Patient Data Table
    data = [["Risk Factor", "Patient Value"]]
    for k, v in st.session_state.result["inputs"].items():
        data.append([k, str(v)])
    table = Table(data, colWidths=[8*cm, 6*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e40af")),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BACKGROUND', (0,1), (-1,-1), colors.HexColor("#f0f9ff")),
        ('GRID', (0,0), (-1,-1), 0.8, colors.grey),
        ('PADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.8*cm))

    # Result & Gauge
    col = st.columns([1, 1.2])
    with col[0]:
        story.append(Paragraph(f"<b>Risk Level:</b> <font color='{st.session_state.result['color']}'>{st.session_state.result['level']}</font>", styles["Normal"]))
        story.append(Paragraph(f"<b>Probability:</b> {st.session_state.result['prob']:.1%}", styles["Normal"]))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("<b>Clinical Recommendation:</b>", styles["Normal"]))
        story.append(Paragraph(st.session_state.result["advice"], ParagraphStyle(name="Advice", fontSize=11, leading=14)))

    img_data = st.session_state.result["fig"].to_image(format="png", width=700, height=400)
    story.append(RLImage(BytesIO(img_data), width=12*cm, height=7*cm))

    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph("Disclaimer: This tool is for screening only • Not a substitute for medical diagnosis • Consult a physician.",
                           ParagraphStyle(name="Disc", fontSize=9, textColor=colors.grey, alignment=1)))

    doc.build(story)
    return buffer.getvalue()

if "result" in st.session_state:
    pdf = create_beautiful_pdf()
    b64 = base64.b64encode(pdf).decode()
    st.markdown(f'''
    <a href="data:application/pdf;base64,{b64}" download="AeroLung_Report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf">
        <button style="background:#dc2626; color:white; padding:1rem 3rem; border:none; border-radius:12px; font-size:1.2rem; cursor:pointer; width:100%;">
            Download PDF Report (Click)
        </button>
    </a>
    ''', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center; color:#6b7280; font-size:0.95rem;'>"
            "This AI tool is for screening purposes only • Not a substitute for professional medical diagnosis</p>",
            unsafe_allow_html=True)
