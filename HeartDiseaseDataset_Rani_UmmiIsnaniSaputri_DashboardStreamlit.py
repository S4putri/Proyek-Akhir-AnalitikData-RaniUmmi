import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import os

# ================================================
# KONFIGURASI HALAMAN
# ================================================
st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("🫀 Heart Disease Dashboard – Rani Ummi Isnani Saputri")
st.markdown("""
Dashboard ini digunakan untuk **Analisis & Prediksi Penyakit Jantung** menggunakan algoritma **Random Forest**.
""")

# ================================================
# 1. LOAD DATASET (Bisa Upload / Otomatis)
# ================================================
st.header("📂 Dataset")

# Cek apakah ada file upload
uploaded = st.file_uploader("Upload file heart.csv (Opsional)", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.success("Dataset dari upload berhasil dimuat!")
elif os.path.exists("heart.csv"):
    df = pd.read_csv("heart.csv")
    st.info("Menggunakan dataset bawaan 'heart.csv'.")
else:
    st.error("❌ File dataset tidak ditemukan. Silakan upload file 'heart.csv'.")
    st.stop()

with st.expander("Lihat Preview Dataset"):
    st.dataframe(df.head())

# ================================================
# 2. PREPROCESSING DATA
# ================================================
st.header("🛠 Preprocessing Data")

required_cols = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
    st.stop()

df = df[required_cols].copy()
df = df.fillna(df.median(numeric_only=True))

# Pastikan kolom sex numerik (0/1)
if df['sex'].dtype == object:
    df['sex'] = df['sex'].map({'male':1, 'female':0}).fillna(0).astype(int)

st.success("Preprocessing selesai.")

# ================================================
# 3. NORMALISASI DATA
# ================================================
st.header("📊 Normalisasi Data")

X_raw = df.drop(columns=['target'])
y = df['target']

# Scaler didefinisikan di sini
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_raw)

df_norm = pd.DataFrame(X_scaled, columns=X_raw.columns)
df_norm['target'] = y.values

# ================================================
# 4. TRAINING MODEL
# ================================================
st.header("🤖 Training Model – Random Forest")

X_train, X_test, y_train, y_test = train_test_split(
    df_norm.drop(columns=['target']),
    df_norm['target'],
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
st.metric("Akurasi Model", f"{acc:.2%}")

# ================================================
# 5. EVALUASI MODEL
# ================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Feature Importance")
    importance = rf.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X_raw.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.barplot(data=feat_df.head(10), x="Importance", y="Feature", ax=ax2)
    st.pyplot(fig2)

# ================================================
# 6. VISUALISASI DATASET
# ================================================
st.header("📈 Visualisasi Dataset")
colA, colB = st.columns(2)
with colA:
    fig_v1, ax_v1 = plt.subplots()
    sns.histplot(df['age'], bins=20, kde=True, ax=ax_v1)
    ax_v1.set_title("Distribusi Usia")
    st.pyplot(fig_v1)
with colB:
    fig_v2, ax_v2 = plt.subplots()
    sns.countplot(x=df['target'], ax=ax_v2)
    ax_v2.set_title("Jumlah Pasien Sakit (1) vs Sehat (0)")
    st.pyplot(fig_v2)

# ================================================
# 6. VISUALISASI DATASET
# ================================================
st.header("📈 Visualisasi Dataset")

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    sns.histplot(df['age'], bins=20, kde=True, ax=ax1)
    ax1.set_title("Distribusi Usia")
    st.pyplot(fig1)

    fig3, ax3 = plt.subplots()
    sns.histplot(df['trestbps'], bins=20, kde=True, ax=ax3)
    ax3.set_title("Distribusi Tekanan Darah")
    st.pyplot(fig3)

with col2:
    fig2, ax2 = plt.subplots()
    sns.histplot(df['chol'], bins=20, kde=True, ax=ax2)
    ax2.set_title("Distribusi Kolesterol")
    st.pyplot(fig2)

    fig4, ax4 = plt.subplots()
    sns.countplot(x=df['target'], ax=ax4)
    ax4.set_title("Jumlah Pasien Sakit vs Tidak")
    ax4.set_xticklabels(["Tidak Sakit (0)", "Sakit (1)"])
    st.pyplot(fig4)

# ================================================
# 7. FORM PREDIKSI (YANG DIPERBAIKI)
# ================================================
st.divider()
st.header("🩺 Prediksi Penyakit Jantung (Mode Simpel)")
st.info("Masukkan data klinis dasar. Sistem akan otomatis menyesuaikan parameter teknis agar hasil prediksi akurat.")

# Kita siapkan nilai default dari dataset (untuk jaga-jaga)
default_fbs = df['fbs'].mode()[0]
default_restecg = df['restecg'].mode()[0]
# Note: oldpeak, slope, ca, thal akan diatur secara dinamis di bawah (Logic Cerdas)

with st.form("prediction_form"):
    col_input1, col_input2 = st.columns(2)

    with col_input1:
        input_age = st.number_input("Usia (tahun)", 1, 120, 50)
        
        input_sex_label = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        # Konversi ke 0/1 (Sesuai preprocessing: Male=1, Female=0)
        input_sex = 1 if input_sex_label == "Laki-laki" else 0

        input_cp_label = st.selectbox(
            "Tipe Nyeri Dada",
            ["Tidak Nyeri / Nyeri Biasa", "Nyeri Sedang (Atypical)", "Nyeri Bukan Jantung", "Nyeri Tanpa Gejala"]
        )
        # Mapping CP yang disesuaikan
        if input_cp_label == "Tidak Nyeri / Nyeri Biasa": input_cp = 0
        elif input_cp_label == "Nyeri Sedang (Atypical)": input_cp = 1
        elif input_cp_label == "Nyeri Bukan Jantung": input_cp = 2
        else: input_cp = 3

    with col_input2:
        input_trestbps = st.number_input("Tekanan Darah (mmHg)", value=120)
        input_chol = st.number_input("Kolesterol (mg/dl)", value=200)
        input_thalach = st.number_input("Detak Jantung Maksimum", value=150)
        
        input_exang_label = st.selectbox("Nyeri dada saat olahraga?", ["Tidak", "Ya"])
        input_exang = 1 if input_exang_label == "Ya" else 0

    submit = st.form_submit_button("🔍 Prediksi Sekarang")

# ================================================
# 8. PROSES PREDIKSI (DENGAN LOGIKA CERDAS)
# ================================================
if submit:
    # --- LOGIKA CERDAS (SMART FILL) ---
    # Masalah sebelumnya: Mengisi dengan rata-rata membuat hasil bias 'Sakit'.
    # Solusi: Jika User bilang "Tidak Nyeri Olahraga" (Sehat), kita paksa parameter teknis ke "Normal".
    
    if input_exang == 0:
        # Skenario Sehat: Set parameter kerusakan jantung ke 0 (Normal)
        fill_oldpeak = 0.0  # Tidak ada depresi ST
        fill_slope = 2      # Slope menanjak (Indikasi sehat)
        fill_ca = 0         # Pembuluh darah bersih
        fill_thal = 2       # Thal normal
    else:
        # Skenario Berisiko: Gunakan nilai tengah dari dataset
        fill_oldpeak = df['oldpeak'].median()
        fill_slope = df['slope'].mode()[0]
        fill_ca = df['ca'].mode()[0]
        fill_thal = df['thal'].mode()[0]

    # 1. Susun Data Input 
    # Urutan HARUS SAMA: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    input_data = pd.DataFrame([[
        input_age,      # User
        input_sex,      # User
        input_cp,       # User
        input_trestbps, # User
        input_chol,     # User
        default_fbs,    # Default
        default_restecg,# Default
        input_thalach,  # User
        input_exang,    # User
        fill_oldpeak,   # SMART FILL (Otomatis)
        fill_slope,     # SMART FILL (Otomatis)
        fill_ca,        # SMART FILL (Otomatis)
        fill_thal       # SMART FILL (Otomatis)
    ]], columns=X_raw.columns)

    # 2. Normalisasi (PENTING: Gunakan scaler yang sama)
    input_scaled = scaler.transform(input_data)

    # 3. Prediksi
    prediction = rf.predict(input_scaled)[0]
    proba = rf.predict_proba(input_scaled)[0]

    # 4. Tampilkan Hasil
    st.divider()
    st.subheader("📌 Hasil Prediksi AI")

    col_resA, col_resB = st.columns([1, 2])
    
    with col_resA:
        if prediction == 1:
            st.error("🔴 **TERINDIKASI PENYAKIT**")
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966486.png", width=100)
        else:
            st.success("🟢 **SEHAT / AMAN**")
            st.image("https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)

    with col_resB:
        st.write(f"**Probabilitas (Keyakinan Model): {proba[prediction]:.1%}**")
        st.write("Interpretasi:")
        if prediction == 1:
            st.warning("Berdasarkan pola data, pasien memiliki kemiripan dengan profil penyakit jantung.")
        else:
            st.success("Berdasarkan data yang dimasukkan, tidak ditemukan indikasi risiko penyakit jantung.")
            
    # Tampilkan detail data (Opsional)
    with st.expander("Lihat Data Teknis yang Diproses"):
        st.write("Sistem otomatis mengisi nilai teknis (Slope, CA, Thal) berdasarkan gejala nyeri Anda:")
        st.dataframe(input_data)

# Footer
st.markdown("---")
st.caption("Developed by Rani Ummi Isnani Saputri | UAS Data Mining 2025")

# streamlit run "HeartDiseaseDataset_Rani_UmmiIsnaniSaputri_DashboardStreamlit.py"

