import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

st.title("🫀 Heart Disease Dashboard – Rani Ummi Isnani Saputri")

st.markdown("""
Dashboard ini digunakan untuk Analisis Prediksi Penyakit Jantung dengan Algoritma Random Forest.
""")

# ================================================
# 1. UPLOAD DATASET
# ================================================
st.header("📤 Upload Dataset")

uploaded = st.file_uploader("Unggah dataset (CSV atau Excel)", type=["csv", "xlsx"])

if uploaded is not None:
    # Load dataset
    try:
        if uploaded.name.endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)

        st.success("Dataset berhasil diunggah!")
        st.write("### Preview Dataset")
        st.dataframe(df.head())

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()
else:
    st.warning("Silakan upload dataset terlebih dahulu!")
    st.stop()

# ================================================
# 2. PREPROCESSING
# ================================================
st.header("🛠 Preprocessing Data")

required_cols = [
    'age','sex','cp','trestbps','chol','fbs','restecg',
    'thalach','exang','oldpeak','slope','ca','thal','target'
]

# Cek apakah kolom exist
missing_cols = [c for c in required_cols if c not in df.columns]

if missing_cols:
    st.error(f"Dataset tidak valid! Kolom berikut hilang: {missing_cols}")
    st.stop()

# Filter kolom
df = df[required_cols].copy()

# Isi missing value
df = df.fillna(df.median(numeric_only=True))

# Convert sex jika string
if df['sex'].dtype == object:
    df['sex'] = df['sex'].map({'male':0, 'female':1}).fillna(0).astype(int)

st.success("Preprocessing selesai!")
st.write("### Data Setelah Preprocessing")
st.dataframe(df.head())

# ================================================
# 3. NORMALISASI
# ================================================
st.header("📊 Normalisasi Fitur")

scaler = MinMaxScaler()

X = df.drop(columns=['target'])
y = df['target']

X_norm = scaler.fit_transform(X)

df_norm = pd.DataFrame(X_norm, columns=X.columns)
df_norm['target'] = y.values

st.write("### Data Setelah Normalisasi")
st.dataframe(df_norm.head())

# ================================================
# 4. TRAIN MODEL
# ================================================
st.header("🤖 Training Model – Random Forest")

X_train, X_test, y_train, y_test = train_test_split(
    df_norm.drop(columns=['target']),
    df_norm['target'],
    test_size=0.2,
    random_state=42
)

rf = RandomForestClassifier(
    n_estimators=120,
    max_depth=5,
    min_samples_split=8,
    min_samples_leaf=4,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

acc = accuracy_score(y_test, y_pred)

st.write(f"### 🎯 Akurasi Model: **{acc:.3f}**")
st.text(classification_report(y_test, y_pred))

# ================================================
# 5. CONFUSION MATRIX
# ================================================
st.header("📌 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=ax)
st.pyplot(fig)

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
# 7. FEATURE IMPORTANCE
# ================================================
st.header("📌 Feature Importance – Random Forest")

importance = rf.feature_importances_
feat_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots(figsize=(7,6))
sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax, palette="viridis")
st.pyplot(fig)

st.write("### Ranking Fitur")
st.dataframe(feat_df)

st.success("Dashboard berhasil dijalankan tanpa error! 🎉")
