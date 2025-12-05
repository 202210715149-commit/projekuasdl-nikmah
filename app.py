import streamlit as st
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# LOAD MODEL & SCALER
# ---------------------------------------------------------
logreg = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🧠",
    layout="centered"
)

# ================================
# CUSTOM CSS (PREMIUM UI)
# ================================
st.markdown("""
<style>

body {
    background: linear-gradient(to right, #eef2f3, #dfe9f3);
}

.header {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    padding: 10px;
    background: linear-gradient(90deg, #4b79a1, #283e51);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.glass-card {
    background: rgba(255,255,255,0.5);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1);
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
}

.footer {
    text-align:center;
    margin-top:50px;
    font-size:13px;
    color:#666;
}

</style>
""", unsafe_allow_html=True)

# ============ ABOUT STROKE SECTION (PREMIUM CARD) ============
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

st.markdown("""
### ⚠️ Apa Itu Stroke?
Stroke adalah kondisi darurat medis ketika aliran darah ke otak terganggu, menyebabkan sel-sel otak mati dalam hitungan menit.  
Ini dapat menyebabkan **kelumpuhan, gangguan berbicara, kehilangan ingatan**, hingga kematian jika tidak ditangani segera.

### 🧩 Jenis Stroke:
- **Ischemic Stroke (85% kasus):** Penyumbatan pembuluh darah.
- **Hemorrhagic Stroke:** Pecahnya pembuluh darah di otak.

### 🔥 Faktor Risiko Tinggi:
- Tekanan darah tinggi (hypertension)
- Penyakit jantung
- Kadar gula darah tinggi
- Merokok
- Usia lanjut
- Obesitas / BMI tinggi  
- Gaya hidup tidak aktif

### 🚨 Gejala Umum (FAST):
- **F**ace drooping (wajah menurun)
- **A**rm weakness (lemah tangan)
- **S**peech difficulty (sulit bicara)
- **T**ime to call emergency services

### 🛡 Pencegahan Stroke:
- Mengontrol tekanan darah  
- Mengurangi konsumsi gula dan garam  
- Tidak merokok  
- Berolahraga teratur  
- Menjaga berat badan sehat  
""")

st.markdown("</div>", unsafe_allow_html=True)

st.write("")
st.write("")

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown("<h1 style='text-align:center;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Masukkan data pasien untuk memprediksi risiko stroke.</p>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

# ================================
# SIDEBAR MENU
# ================================
st.sidebar.title("⚙️ Settings")
st.sidebar.info("Isi data pasien lalu klik **Predict Stroke Risk**")
st.sidebar.markdown("---")
st.sidebar.write("Developed by **Nikmah Azizah**")

# ================================
# FORM - PREMIUM GLASS CARD
# ================================
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("🧓 Age", 1, 120, 25)
    hypertension = st.selectbox("💓 Hypertension", [0, 1])
    heart_disease = st.selectbox("❤️ Heart Disease", [0, 1])
    bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 25.0)

with col2:
    avg_glucose_level = st.number_input("🩸 Average Glucose Level", 40.0, 300.0, 100.0)
    gender = st.selectbox("🚻 Gender", ["Female", "Male", "Other"])
    ever_married = st.selectbox("💍 Ever Married", ["No", "Yes"])
    Residence_type = st.selectbox("🏠 Residence Type", ["Urban", "Rural"])
    work_type = st.selectbox("🧑‍💼 Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
    smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes"])

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# ONE HOT ENCODING (SAME AS TRAINING)
# ---------------------------------------------------------
gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0
ever_married_yes = 1 if ever_married == "Yes" else 0
Residence_urban = 1 if Residence_type == "Urban" else 0

work_Never = 1 if work_type == "Never_worked" else 0
work_Private = 1 if work_type == "Private" else 0
work_Self = 1 if work_type == "Self-employed" else 0
work_children = 1 if work_type == "Children" else 0

smoke_former = 1 if smoking_status == "formerly smoked" else 0
smoke_never = 1 if smoking_status == "never smoked" else 0
smoke_smokes = 1 if smoking_status == "smokes" else 0

# ---------------------------------------------------------
# FINAL INPUT ARRAY (16 FEATURES)
# ---------------------------------------------------------
input_data = np.array([[
    age,
    hypertension,
    heart_disease,
    avg_glucose_level,
    bmi,
    gender_male,
    gender_other,
    ever_married_yes,
    work_Never,
    work_Private,
    work_Self,
    work_children,
    Residence_urban,
    smoke_former,
    smoke_never,
    smoke_smokes
]])

input_scaled = scaler.transform(input_data)

# ================================
# PREDICTION BUTTON + LOADING
# ================================
predict = st.button("🔍 Predict Stroke Risk")

if predict:
    with st.spinner("Menganalisis data pasien..."):
        time.sleep(1.2)

    prediction = logreg.predict(input_scaled)[0]
    proba = logreg.predict_proba(input_scaled)[0][1]

    # Probability Chart
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.barh(["Stroke Probability"], [proba], color="#d9534f" if prediction == 1 else "#5cb85c")
    ax.set_xlim([0, 1])
    st.pyplot(fig)

    # Result Message
    if prediction == 1:
        st.markdown(f"""
        <div class='result-box' style='background-color:#ffdddd; color:#b30000;'>
        ⚠️ <b>High Stroke Risk</b><br>Probability: {proba:.2f}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-box' style='background-color:#ddffdd; color:#006600;'>
        🟢 <b>Low Stroke Risk</b><br>Probability: {proba:.2f}
        </div>
        """, unsafe_allow_html=True)

# ================================
# FOOTER
# ================================
st.markdown("""
<div class='footer'>
Created by <b>Nikmah Azizah</b> • Deep Learning Project • ubharajaya  
</div>
""", unsafe_allow_html=True)

st.write("---")
st.caption("✨ Dikembangkan dengan Logistic Regression untuk Deployment | ANN digunakan untuk penelitian model.")
