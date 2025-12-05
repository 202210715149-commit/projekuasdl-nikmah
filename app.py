import streamlit as st
import numpy as np
import pickle

st.set_page_config(page_title="Stroke Prediction", page_icon="🧠", layout="wide")

# =========================
# SESSION STATE FOR NAVIGATION
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

def go_home():
    st.session_state.page = "home"

def go_predict():
    st.session_state.page = "predict"


# =========================
# LOAD MODEL
# =========================
logreg = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


# =========================
# HOME PAGE
# =========================
if st.session_state.page == "home":

    st.markdown("<h1 style='text-align:center;font-size:48px;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;font-size:18px;opacity:0.7;'>Masukkan data pasien untuk memprediksi risiko stroke.</p>", unsafe_allow_html=True)

    st.write("")
    st.write("")

    # Tombol besar menuju halaman prediksi
    st.markdown(
        """
        <div style='display:flex;justify-content:center;'>
            <button style="
                background: linear-gradient(90deg, #4b79a1, #283e51);
                padding: 15px 40px;
                border-radius: 30px;
                font-size: 22px;
                border:none;
                color:white;
                cursor:pointer;"
                onclick="document.querySelector('button[data-baseweb=\'button\']').click()">
                🚀 Mulai Prediksi
            </button>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.button("hidden button", on_click=go_predict, key="predict_button", help="", type="secondary", disabled=False)

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    # Informasi / Edukasi Stroke
    st.markdown("## ℹ️ Apa Itu Stroke?")
    st.write("""
    Stroke adalah kondisi darurat medis akibat terhentinya aliran darah ke otak, 
    menyebabkan kerusakan jaringan otak dalam hitungan menit.
    """)

    st.markdown("## 🔥 Faktor Risiko Utama")
    st.write("""
    - Tekanan darah tinggi (Hypertension)
    - Penyakit jantung
    - Diabetes
    - Merokok
    - BMI tinggi
    """)

    st.markdown("## 🚨 Gejala Umum (FAST)")
    st.write("""
    **F — Face Drooping**  
    **A — Arm Weakness**  
    **S — Speech Difficulty**  
    **T — Time to Call Emergency**  
    """)



# =========================
# PREDICT FORM PAGE
# =========================
elif st.session_state.page == "predict":

    st.markdown("<h1 style='text-align:center;'>🧠 Stroke Risk Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Masukkan data pasien untuk memprediksi risiko stroke.</p>", unsafe_allow_html=True)

    # Tombol kembali ke home
    st.button("⬅️ Kembali ke Home", on_click=go_home)

    st.write("")
    st.write("")

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

    if st.button("🚀 Predict"):
        pred = logreg.predict(X_scaled)[0]
        prob = logreg.predict_proba(X_scaled)[0][1]

        if pred == 1:
            st.error(f"⚠️ Risiko Stroke Tinggi (Probabilitas: {prob:.2f})")
        else:
            st.success(f"🟢 Risiko Stroke Rendah (Probabilitas: {prob:.2f})")
