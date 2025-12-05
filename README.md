# 🧠 Stroke Prediction Using Artificial Neural Network & Logistic Regression

Project ini bertujuan untuk memprediksi risiko stroke berdasarkan data kesehatan pasien menggunakan dua algoritma:

- **Artificial Neural Network (ANN)** → Model utama (Deep Learning)
- **Logistic Regression** → Model pembanding (Machine Learning)

Dataset yang digunakan berasal dari Kaggle:

🔗 https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset  
📅 Tahun Dataset: **2021**  
📊 Jumlah Data: **5110 baris**  
🔢 Jumlah Variabel: **12 kolom**

---

## 📌 Features (Input Model)

Berikut fitur yang digunakan untuk prediksi (selain target `stroke`):

1. `age`
2. `hypertension`
3. `heart_disease`
4. `avg_glucose_level`
5. `bmi`
6. `gender`
7. `ever_married`
8. `work_type`
9. `Residence_type`
10. `smoking_status`

Semua fitur kategori telah dilakukan **One-Hot Encoding**, dan fitur numerik dinormalisasi menggunakan **StandardScaler**.

---

## 📁 Project Structure

