import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="AeroLung AI • Lung Cancer Risk Prediction",
    page_icon="Lungs",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========================= CUSTOM CSS (Exact Match) =========================
st.markdown("""
<style>
    .main {background: linear-gradient(to bottom, #f0f9ff, #e0f2fe);}
    .header-box {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 35px rgba(37,99,235,0.3);
        margin-bottom: 2rem;
    }
    .feature-card {
        background: white;
        padding: 1.8rem;
        border-radius: 16px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        height: 100%;
    }
    .result-box {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.12);
        border-left: 10px solid #dc2626;
        margin: 2rem 0;
    }
    .stButton > button {
        background: #1e40af;
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        border: none;
        box-shadow: 0 4px 15px rgba(30,64,175,0.4);
    }
    .stButton > button:hover {
        background: #1d4ed8;
    }
    h1 {font-size: 3rem !important; margin: 0;}
    h3 {font-size: 1.4rem; opacity: 0.9; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ========================= LOAD MODEL =========================
@st.cache_resource
def load_model():
    return joblib.load('LungCancer_Stacking_100.pkl')

model = load_model()

# ========================= HEADER =========================
st.markdown("""
<div class="header-box">
    <h1>AeroLung AI</h1>
    <h3>Lung Cancer Risk Prediction System</h3>
    <p>Clinical-Grade Stacking Ensemble • Calibrated Probability Output</p>
</div>
""", unsafe_allow_html=True)

# ========================= INPUT FORM =========================
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
    family_smoking = st.selectbox("Family History of Heavy Smoking", ["No", "Yes"], index=1)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='feature-card'>", unsafe_allow_html=True)
    st.subheader("Health & Demographics")
    age = st.slider("Age", 18, 100, 68, help="Higher age increases risk")
    energy = st.slider("Energy Level (1=Very Low → 10=High)", 1, 10, 4)
    immunity = st.slider("Stress & Immune Health (1=Poor → 10=Excellent)", 1, 10, 3)
    spo2 = st.slider("Resting SpO₂ (%)", 85, 100, 91)
    st.markdown("</div>", unsafe_allow_html=True)

# ========================= CALCULATE BUTTON =========================
if st.button("Calculate Risk", type="primary", use_container_width=True):
    with st.spinner("Calculating clinical risk score..."):
        # Feature vector (exact order your model expects)
        features = np.array([[
            ["Never", "Former", "Current"].index(smoking),
            ["No", "Yes"].index(breathing_issue),
            ["No", "Yes"].index(throat_discomfort),
            ["No", "Yes"].index(family_smoking),
            energy,
            immunity,
            spo2,
            ["No", "Yes"].index(family_cancer),
            ["No", "Yes"].index(pollution),
            age
        ]])

        prob = model.predict_proba(features)[0][1]
        risk_pct = round(prob * 100, 1)

        # Risk classification
        if prob < 0.3:
            level, color, advice = "Low Risk", "#16a34a", "Continue healthy lifestyle and annual screening."
        elif prob < 0.7:
            level, color, advice = "Moderate Risk", "#ca8a04", "Recommend low-dose CT scan and pulmonologist consultation."
        else:
            level, color, advice = "High Risk", "#dc2626", "Urgent: Schedule chest CT and oncology referral immediately."

        # ========================= GAUGE CHART (Exact Match) =========================
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "<b>Lung Cancer Risk Probability</b>", 'font': {'size': 24}},
            number={'suffix': "%", 'font': {'size': 48, 'color': '#1e3a8a'}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#dcfce7'},
                    {'range': [30, 70], 'color': '#fef9c3'},
                    {'range': [70, 100], 'color': '#fecaca'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 8},
                    'thickness': 0.8,
                    'value': 70
                }
            }
        ))
        fig.update_layout(
            height=500,
            margin=dict(l=50, r=50, t=100, b=50),
            font=dict(family="Arial", size=16)
        )
        st.plotly_chart(fig, use_container_width=True)

        # ========================= RESULT BOX (Exact Match) =========================
        st.markdown(f"""
        <div class="result-box">
            <h2 style="color:{color}; margin:0; font-size:2.8rem;">{level}</h2>
            <h3 style="margin:10px 0; font-size:2rem;">Probability: <strong>{prob:.1%}</strong></h3>
            <p style="font-size:1.3rem; color:#374151; margin:20px 0;">
                <strong>Clinical Recommendation:</strong><br>
                {advice}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Save result
        st.session_state.result = {
            "probability": prob,
            "level": level,
            "advice": advice,
            "timestamp": datetime.now().strftime("%B %d, %Y • %H:%M")
        }

# ========================= FOOTER DISCLAIMER =========================
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#6b7280; font-size:0.95rem;'>"
    "This AI tool is for screening purposes only • Not a substitute for professional medical diagnosis • "
    f"Report generated: {datetime.now().strftime('%B %d, %Y')}</p>",
    unsafe_allow_html=True
)
