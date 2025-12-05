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

# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# SIDEBAR MENU
# ---------------------------------------------------------
st.sidebar.markdown("<div class='sidebar-title'>📘 Informasi Stroke</div>", unsafe_allow_html=True)

menu = st.sidebar.radio(
    "Pilih topik penjelasan:",
    [
        "🧠 Penjelasan Stroke",
        "📌 Jenis Stroke",
        "🔥 Faktor Risiko Tinggi",
        "🚨 Gejala Umum (FAST)",
        "🛡 Pencegahan Stroke",
        "🧠 Stroke Risk Prediction"
    ],
    label_visibility="collapsed"
)

# ---------------------------------------------------------
# KONTEN DINAMIS
# ---------------------------------------------------------
st.markdown("<div class='content-card'>", unsafe_allow_html=True)

# 1. Penjelasan Stroke
if menu == "🧠 Penjelasan Stroke":
    st.markdown("""
    ## 🧠 Apa Itu Stroke?
    Stroke adalah kondisi ketika aliran darah ke otak terhenti...
    """)

# 2. Jenis Stroke
elif menu == "📌 Jenis Stroke":
    st.markdown("""
    ## 📌 Jenis-Jenis Stroke
    - Ischemic Stroke  
    - Hemorrhagic Stroke  
    - TIA (Mini Stroke)
    """)

# 3. Faktor Risiko Tinggi
elif menu == "🔥 Faktor Risiko Tinggi":
    st.markdown("""
    ## 🔥 Faktor Risiko Tinggi Stroke
    - Hipertensi  
    - Diabetes  
    - Kolesterol tinggi  
    - Obesitas  
    """)    

# 4. Gejala FAST
elif menu == "🚨 Gejala Umum (FAST)":
    st.markdown("""
    ## 🚨 Gejala Umum Stroke (FAST)
    F - Face drooping  
    A - Arm weakness  
    S - Speech difficulty  
    T - Time to call emergency  
    """)    

# 5. Pencegahan Stroke
elif menu == "🛡 Pencegahan Stroke":
    st.markdown("""
    ## 🛡 Pencegahan Stroke
    - Jaga berat badan  
    - Kurangi rokok  
    - Kontrol tekanan darah  
    """)    

# ---------------------------------------------------------
# 6. FORM INPUT HANYA MUNCUL DI MENU PREDIKSI
# ---------------------------------------------------------
elif menu == "🧠 Stroke Risk Prediction":

    st.markdown("<h1 style='text-align:center;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)
    st.write("Masukkan data pasien untuk memprediksi risiko stroke.")

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("🧓 Age", 1, 120, 25)
        hypertension = st.selectbox("💓 Hypertension", [0, 1])
        heart_disease = st.selectbox("❤️ Heart Disease", [0, 1])
        bmi = st.number_input("⚖️ BMI", 10.0, 60.0, 25.0)

    with col2:
        avg_glucose_level = st.number_input("🩸 Glucose Level", 40.0, 300.0, 100.0)
        gender = st.selectbox("🚻 Gender", ["Female", "Male", "Other"])
        ever_married = st.selectbox("💍 Ever Married", ["No", "Yes"])
        Residence_type = st.selectbox("🏠 Residence", ["Urban", "Rural"])
        work_type = st.selectbox("🧑‍💼 Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
        smoking_status = st.selectbox("🚬 Smoking Status", ["never smoked", "formerly smoked", "smokes"])

    st.markdown("</div>", unsafe_allow_html=True)

    # ------------------- One Hot Encoding -------------------
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

    input_data = np.array([[
        age, hypertension, heart_disease, avg_glucose_level, bmi,
        gender_male, gender_other, ever_married_yes,
        work_Never, work_Private, work_Self, work_children,
        Residence_urban, smoke_former, smoke_never, smoke_smokes
    ]])

    input_scaled = scaler.transform(input_data)

    if st.button("🔍 Predict Stroke Risk"):
        prediction = logreg.predict(input_scaled)[0]
        proba = logreg.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ High Stroke Risk (Probabilitas: {proba:.2f})")
        else:
            st.success(f"🟢 Low Stroke Risk (Probabilitas: {proba:.2f})")

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("""
<div class='footer'>
Created by <b>Nikmah Azizah</b> • Deep Learning Project • ubharajaya  
</div>
""", unsafe_allow_html=True)
