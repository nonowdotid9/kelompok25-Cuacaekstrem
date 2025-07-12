import streamlit as st
import pandas as pd
import joblib

# ========================
# Fungsi Bantu
# ========================

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

@st.cache_data
def load_encoders():
    le_kejadi = joblib.load("encoder_kejadian.pkl")
    le_prov = joblib.load("encoder_provinsi.pkl")
    le_label = joblib.load("encoder_label.pkl")
    return le_kejadi, le_prov, le_label

# ========================
# Halaman Streamlit
# ========================

st.set_page_config(page_title="Prediksi Keparahan Cuaca Ekstrem", layout="centered")
st.title("Klasifikasi Tingkat Keparahan Bencana Cuaca Ekstrem")

le_kejadi, le_prov, le_label = load_encoders()

kejadian_input = st.selectbox("Jenis Kejadian", le_kejadi.classes_)
provinsi_input = st.selectbox("Provinsi", le_prov.classes_)

meninggal = st.number_input("Jumlah Meninggal", min_value=0)
hilang = st.number_input("Jumlah Hilang", min_value=0)
terluka = st.number_input("Jumlah Terluka", min_value=0)
rumah_rusak = st.number_input("Jumlah Rumah Rusak", min_value=0)
rumah_terendam = st.number_input("Jumlah Rumah Terendam", min_value=0)
fasum_rusak = st.number_input("Jumlah Fasilitas Umum Rusak", min_value=0)

if st.button("Prediksi"):
    model = load_model()

    kejadian_encoded = le_kejadi.transform([kejadian_input])[0]
    provinsi_encoded = le_prov.transform([provinsi_input])[0]

    data_input = pd.DataFrame([[
        kejadian_encoded, provinsi_encoded, meninggal, hilang, terluka,
        rumah_rusak, rumah_terendam, fasum_rusak
    ]], columns=[
        "Kejadian_enc", "Provinsi_enc", "Meninggal", "Hilang", "Terluka",
        "Rumah Rusak", "Rumah Terendam", "Fasum Rusak"
    ])

    pred_label = model.predict(data_input)[0]
    pred_kategori = le_label.inverse_transform([pred_label])[0]

    st.success(f"🚨 Tingkat Keparahan Diprediksi: **{pred_kategori}**")
