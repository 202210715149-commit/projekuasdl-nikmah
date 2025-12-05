import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# PAGE CONFIG
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🧠",
    layout="centered"
)

# LOAD CUSTOM CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# DARK MODE SWITCH
dark_mode = st.checkbox("🌙 Dark Mode", value=False)

if dark_mode:
    st.markdown("<body data-theme='dark'>", unsafe_allow_html=True)
else:
    st.markdown("<body data-theme='light'>", unsafe_allow_html=True)

# HEADER WITH PREMIUM GRADIENT
st.markdown("""
<h1 class='header' style='text-align:center;'>🧠 Stroke Prediction Dashboard</h1>
<p style='text-align:center; font-size:17px; opacity:0.8;'>
AI-powered system to estimate stroke risk based on medical indicators.
</p>
""", unsafe_allow_html=True)

st.write("")

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

# =========== MAIN CALL TO ACTION BUTTON ============
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
if st.button("🚀 Mulai Prediksi Risiko Stroke", use_container_width=True):
    switch_page("Predict_Stroke")
st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.markdown("""
<br><br>
<p class='footer'>
Created by <b>Nikmah Azizah</b> • Deep Learning Project • UB Harajaya  
</p>
""", unsafe_allow_html=True)
