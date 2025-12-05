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

import streamlit as st

st.set_page_config(page_title="Stroke Information", page_icon="🧠", layout="wide")

# --- CSS untuk Premium UI ---
st.markdown("""
<style>
.content-card {
    background: rgba(255,255,255,0.6);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
}
.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    color: #4b79a1;
}
</style>
""", unsafe_allow_html=True)


# ================================
# SIDEBAR MENU
# ================================
st.sidebar.markdown("<div class='sidebar-title'>📘 Informasi Stroke</div>", unsafe_allow_html=True)

menu = st.radio(
        "",
        [
            "Penjelasan Stroke",
            "Jenis Stroke",
            "Faktor Risiko Tinggi",
            "Gejala FAST",
            "Pencegahan Stroke",
            "Stroke Risk Prediction"
        ]
    )

st.markdown("---")
st.markdown("### ⚙️ Settings")
st.info("Isi data pasien lalu klik **Predict Stroke Risk**")
st.markdown("---")
st.caption("Developed by **Nikmah Azizah**")


# ======================================
# HALAMAN DINAMIS BERDASARKAN MENU
# ======================================
st.markdown("<h1 style='text-align:center;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;opacity:0.7;'>Masukkan data pasien untuk memprediksi risiko stroke.</p>", unsafe_allow_html=True)

st.write("")
st.write("")


# -----------------------------------------------------
# 1. PENJELASAN STROKE
# -----------------------------------------------------
if menu == "Penjelasan Stroke":
    st.subheader("🧠 Apa Itu Stroke?")
    st.write("""
Stroke adalah kondisi ketika aliran darah ke otak terhenti sehingga sel-sel otak mulai mati dalam hitungan menit.
Penyebabnya bisa berupa penyumbatan pembuluh darah atau pecahnya pembuluh darah di otak.
    """)


# -----------------------------------------------------
# 2. JENIS STROKE
# -----------------------------------------------------
elif menu == "Jenis Stroke":
    st.subheader("📌 Jenis Stroke")
    st.write("""
### 1. Ischemic Stroke  
Terjadi karena penyumbatan pembuluh darah otak.

### 2. Hemorrhagic Stroke  
Terjadi karena pecahnya pembuluh darah otak.

### 3. TIA (Transient Ischemic Attack)  
“Mini Stroke” yang menjadi tanda bahaya stroke lebih besar.
    """)


# -----------------------------------------------------
# 3. FAKTOR RISIKO
# -----------------------------------------------------
elif menu == "Faktor Risiko Tinggi":
    st.subheader("🔥 Faktor Risiko Tinggi")
    st.write("""
- Hipertensi  
- Penyakit jantung  
- Gula darah tinggi  
- BMI tinggi  
- Merokok  
- Usia lanjut  
- Riwayat keluarga stroke  
    """)


# -----------------------------------------------------
# 4. GEJALA FAST
# -----------------------------------------------------
elif menu == "Gejala FAST":
    st.subheader("🚨 Gejala Umum Stroke (FAST)")
    st.write("""
**F — Face:** Wajah menurun pada satu sisi  
**A — Arm:** Lengan lemah atau mati rasa  
**S — Speech:** Sulit bicara  
**T — Time:** Segera cari bantuan medis  
    """)


# -----------------------------------------------------
# 5. PENCEGAHAN
# -----------------------------------------------------
elif menu == "Pencegahan Stroke":
    st.subheader("🛡 Pencegahan Stroke")
    st.write("""
- Menurunkan tekanan darah  
- Mengontrol kolesterol  
- Menghindari rokok  
- Menjaga berat badan sehat  
- Olahraga teratur  
- Pola makan sehat  
    """)


# -----------------------------------------------------
# 6. FORM PREDIKSI (INI YANG KAMU MAU)
# -----------------------------------------------------
elif menu == "Stroke Risk Prediction":

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🥚 Age", 1, 120)
        hypertension = st.selectbox("❤️ Hypertension", [0, 1])
        heart = st.selectbox("💔 Heart Disease", [0, 1])
        bmi = st.number_input("🐝 BMI", 10.0, 60.0)

    with col2:
        glucose = st.number_input("🩸 Average Glucose Level", 40.0, 300.0)
        gender = st.selectbox("🚻 Gender", ["Female", "Male", "Other"])
        married = st.selectbox("💍 Ever Married", ["No", "Yes"])
        residence = st.selectbox("🏠 Residence Type", ["Urban", "Rural"])
        work = st.selectbox("👔 Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        smoke = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    # ENCODING
    gender_m = 1 if gender == "Male" else 0
    gender_o = 1 if gender == "Other" else 0
    married_yes = 1 if married == "Yes" else 0
    urban = 1 if residence == "Urban" else 0

    work_never = 1 if work == "Never_worked" else 0
    work_private = 1 if work == "Private" else 0
    work_self = 1 if work == "Self-employed" else 0
    work_child = 1 if work == "Children" else 0

    sm_former = 1 if smoke == "formerly smoked" else 0
    sm_never = 1 if smoke == "never smoked" else 0
    sm_smokes = 1 if smoke == "smokes" else 0

    X = np.array([[
        age, hypertension, heart, glucose, bmi,
        gender_m, gender_o, married_yes,
        work_never, work_private, work_self, work_child,
        urban, sm_former, sm_never, sm_smokes
    ]])

    X_scaled = scaler.transform(X)

    if st.button("🚀 Predict Stroke Risk"):
        pred = logreg.predict(X_scaled)[0]
        prob = logreg.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Risiko Stroke Tinggi (Probabilitas: {prob:.2f})")
        else:
            st.success(f"🟢 Risiko Stroke Rendah (Probabilitas: {prob:.2f})")


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
