import streamlit as st
import numpy as np
import pickle

# ---------------------------------------------------------
# LOAD MODEL LOGISTIC REGRESSION & SCALER
# ---------------------------------------------------------
logreg = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------
st.title("🧠 Stroke Risk Prediction (Logistic Regression Deployment)")
st.write("Masukkan data pasien untuk memprediksi risiko stroke.")

# ---------------------------------------------------------
# FORM INPUT
# ---------------------------------------------------------
age = st.number_input("Age", min_value=1, max_value=120, value=25)

hypertension = st.selectbox("Hypertension (Tekanan Darah Tinggi)", [0, 1])
heart_disease = st.selectbox("Heart Disease (Penyakit Jantung)", [0, 1])

avg_glucose_level = st.number_input("Average Glucose Level", min_value=40.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=60.0, value=25.0)

gender = st.selectbox("Gender", ["Female", "Male"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
Residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
smoking_status = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])


# ---------------------------------------------------------
# ONE-HOT ENCODING SESUAI TRAINING
# ---------------------------------------------------------
gender_male = 1 if gender == "Male" else 0
ever_married_yes = 1 if ever_married == "Yes" else 0
Residence_urban = 1 if Residence_type == "Urban" else 0

work_Private = 1 if work_type == "Private" else 0
work_Self = 1 if work_type == "Self-employed" else 0
work_Govt = 1 if work_type == "Govt_job" else 0
work_children = 1 if work_type == "Children" else 0

smoke_former = 1 if smoking_status == "formerly smoked" else 0
smoke_never = 1 if smoking_status == "never smoked" else 0
smoke_smokes = 1 if smoking_status == "smokes" else 0


# ---------------------------------------------------------
# SUSUN INPUT MODEL
# ---------------------------------------------------------
input_data = np.array([[
    age,
    hypertension,
    heart_disease,
    avg_glucose_level,
    bmi,
    gender_male,
    ever_married_yes,
    work_Private,
    work_Self,
    work_Govt,
    work_children,
    Residence_urban,
    smoke_former,
    smoke_never,
    smoke_smokes
]])

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
st.caption("Model: Logistic Regression | ANN digunakan hanya untuk penelitian, bukan deploy.")
