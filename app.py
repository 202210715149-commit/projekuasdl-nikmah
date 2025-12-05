import streamlit as st
import numpy as np
import pickle

# ---------------------------------------------------------
# LOAD MODEL LOGISTIC REGRESSION & SCALER
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
# FORM INPUT
# ---------------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=25)

hypertension = st.selectbox("Hypertension", [0, 1])
heart_disease = st.selectbox("Heart Disease", [0, 1])

avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)

gender = st.selectbox("Gender", ["Female", "Male", "Other"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes"])

# ---------------------------------------------------------
# ONE-HOT ENCODING - SESUAI TRAINING
# ---------------------------------------------------------

# Gender (training punya: Male, Other → Female=0 untuk semuanya)
gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0

# Ever Married
ever_married_yes = 1 if ever_married == "Yes" else 0

# Residence
Residence_urban = 1 if Residence_type == "Urban" else 0

# Work Type (training punya 4 kolom)
work_Never = 1 if work_type == "Never_worked" else 0
work_Private = 1 if work_type == "Private" else 0
work_Self = 1 if work_type == "Self-employed" else 0
work_children = 1 if work_type == "Children" else 0
# Govt_job tidak perlu encoding karena drop_first=True di training

# Smoking Status
smoke_former = 1 if smoking_status == "formerly smoked" else 0
smoke_never = 1 if smoking_status == "never smoked" else 0
smoke_smokes = 1 if smoking_status == "smokes" else 0

# ---------------------------------------------------------
# SUSUN INPUT MODEL (PENTING!!! Sesuai urutan training)
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

# Scaling
input_scaled = scaler.transform(input_data)

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("Predict Stroke Risk"):
    prediction = logreg.predict(input_scaled)[0]
    proba = logreg.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Risiko Stroke Tinggi (Probabilitas: {proba:.2f})")
    else:
        st.success(f"🟢 Risiko Stroke Rendah (Probabilitas: {proba:.2f})")

st.write("---")
st.caption("Model Logistic Regression | Training menggunakan ANN & Logistic Regression")
