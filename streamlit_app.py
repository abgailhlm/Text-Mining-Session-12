"""
Sentiment Analysis App
======================
Deploy:
- streamlit run streamlit_app.py
"""

import os
import pickle
import joblib
import streamlit as st

from preprocessing import preprocess

MODEL_DIR = "models"

# ======================================
# PAGE CONFIG
# ======================================

st.set_page_config(
    page_title="Sentiment Analysis - Ann Abigail H.",
    page_icon="💬",
    layout="wide"
)

# ======================================
# CUSTOM CSS
# ======================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
}

.metric-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #e9ecef;
}

.result-positive {
    padding: 20px;
    border-radius: 15px;
    background-color: rgba(40,167,69,0.15);
    border-left: 5px solid #28a745;
}

.result-negative {
    padding: 20px;
    border-radius: 15px;
    background-color: rgba(220,53,69,0.15);
    border-left: 5px solid #dc3545;
}

.stButton button {
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ======================================
# LOAD MODEL
# ======================================

@st.cache_resource
def load_ml():
    model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_resource
def load_dl():
    from tensorflow.keras.models import load_model

    model = load_model(
        os.path.join(MODEL_DIR, "dl_model.h5")
    )

    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, config


# ======================================
# PREDICT ML
# ======================================

def predict_ml(text):
    model, tfidf = load_ml()

    cleaned = preprocess(text)

    vec = tfidf.transform([cleaned])

    proba = float(model.predict_proba(vec)[0][1])

    label = "Positif" if proba >= 0.5 else "Negatif"

    return label, proba, cleaned


# ======================================
# PREDICT DL
# ======================================

def predict_dl(text):

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tok, cfg = load_dl()

    cleaned = preprocess(text)

    seq = pad_sequences(
        tok.texts_to_sequences([cleaned]),
        maxlen=cfg["MAX_LEN"],
        padding="post",
        truncating="post"
    )

    proba = float(
        model.predict(seq, verbose=0)[0][0]
    )

    label = "Positif" if proba >= 0.5 else "Negatif"

    return label, proba, cleaned


# ======================================
# SIDEBAR
# ======================================

with st.sidebar:

    st.title("📊 About")

    st.markdown("""
### Features

✅ Text Cleaning

✅ Stopword Removal

✅ Stemming (Sastrawi)

✅ TF-IDF

✅ Logistic Regression

✅ LSTM Deep Learning
""")

    st.divider()

    st.info(
        "Aplikasi analisis sentimen Bahasa Indonesia "
        "menggunakan Machine Learning dan Deep Learning."
    )

# ======================================
# HEADER
# ======================================

st.markdown("""
<div class="hero">
    <h1>💬 Sentiment Analysis System</h1>
    <p>
        Analisis sentimen ulasan Bahasa Indonesia menggunakan
        Machine Learning dan Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# ======================================
# INPUT SECTION
# ======================================

col1, col2 = st.columns([2, 1])

with col1:

    text = st.text_area(
        "Masukkan Teks",
        height=180,
        placeholder="Contoh: Barangnya bagus banget, pengiriman cepat dan seller sangat ramah."
    )

with col2:

    model_choice = st.radio(
        "Pilih Model",
        [
            "Machine Learning (Logistic Regression)",
            "Deep Learning (LSTM)"
        ]
    )

    st.markdown("### Contoh")

    st.caption(
        "Barangnya bagus banget dan pengiriman cepat."
    )

    st.caption(
        "Produk mengecewakan dan kualitas buruk."
    )

# ======================================
# BUTTON
# ======================================

if st.button("🔍 ANALYZE SENTIMENT"):

    if not text.strip():

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Menganalisis sentimen..."):

            if model_choice.startswith("Deep"):

                label, proba, cleaned = predict_dl(text)

            else:

                label, proba, cleaned = predict_ml(text)

        confidence = (
            proba
            if label == "Positif"
            else (1 - proba)
        )

        st.divider()

        # RESULT

        if label == "Positif":

            st.markdown(
                f"""
                <div class="result-positive">
                    <h2>😊 Sentimen Positif</h2>
                    <p>Model mengidentifikasi teks sebagai sentimen positif.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-negative">
                    <h2>😞 Sentimen Negatif</h2>
                    <p>Model mengidentifikasi teks sebagai sentimen negatif.</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        m1, m2, m3 = st.columns(3)

        with m1:
            st.metric(
                "Prediction",
                label
            )

        with m2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with m3:
            st.metric(
                "P(Positive)",
                f"{proba:.4f}"
            )

        st.write("")

        st.progress(float(proba))

        st.write("")

        with st.expander("🔎 Hasil Preprocessing"):

            st.write(cleaned)

        with st.expander("📄 Original Text"):

            st.write(text)

# ======================================
# FOOTER
# ======================================

st.divider()

st.caption(
    "Sentiment Analysis Dashboard | Streamlit + Scikit-Learn + TensorFlow"
)
