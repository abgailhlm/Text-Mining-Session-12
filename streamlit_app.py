import os
import pickle
import joblib
import streamlit as st

from preprocessing import preprocess

MODEL_DIR = "models"

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

# =========================
# CSS
# =========================

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding: 25px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 20px;
}

.stButton button {
    width: 100%;
    height: 50px;
    font-size: 16px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD ML MODEL
# =========================

@st.cache_resource
def load_ml():
    model = joblib.load(
        os.path.join(MODEL_DIR, "ml_model.pkl")
    )

    tfidf = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    return model, tfidf


# =========================
# LOAD LSTM MODEL
# =========================

@st.cache_resource
def load_dl():

    from tensorflow.keras.models import load_model

    model = load_model(
        os.path.join(MODEL_DIR, "dl_model.h5")
    )

    with open(
        os.path.join(MODEL_DIR, "tokenizer.pkl"),
        "rb"
    ) as f:
        tokenizer = pickle.load(f)

    with open(
        os.path.join(MODEL_DIR, "config.pkl"),
        "rb"
    ) as f:
        config = pickle.load(f)

    return model, tokenizer, config


# =========================
# ML PREDICTION
# =========================

def predict_ml(text):

    model, tfidf = load_ml()

    cleaned = preprocess(text)

    vector = tfidf.transform([cleaned])

    probability = float(
        model.predict_proba(vector)[0][1]
    )

    label = "Positif" if probability >= 0.5 else "Negatif"

    return label, probability, cleaned


# =========================
# DL PREDICTION
# =========================

def predict_dl(text):

    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, config = load_dl()

    cleaned = preprocess(text)

    sequence = tokenizer.texts_to_sequences([cleaned])

    padded = pad_sequences(
        sequence,
        maxlen=config["MAX_LEN"],
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    label = "Positif" if probability >= 0.5 else "Negatif"

    return label, probability, cleaned


# =========================
# HEADER
# =========================

st.markdown("""
<div class="hero">
    <h1>💬 Sentiment Analysis Dashboard</h1>
    <p>Analisis Sentimen Bahasa Indonesia menggunakan Logistic Regression dan LSTM</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.header("Model")

    model_choice = st.radio(
        "Pilih Model",
        [
            "Logistic Regression",
            "LSTM Deep Learning"
        ]
    )

    st.markdown("---")

    st.write("Dataset: 4142 data")

# =========================
# INPUT
# =========================

text = st.text_area(
    "Masukkan teks",
    height=180,
    placeholder="Contoh: Barangnya bagus banget dan pengiriman cepat."
)

# =========================
# ANALYZE
# =========================

if st.button("🔍 ANALYZE"):

    if not text.strip():

        st.warning("Masukkan teks terlebih dahulu.")

    else:

        with st.spinner("Memproses..."):

            if model_choice == "LSTM Deep Learning":

                label, probability, cleaned = predict_dl(text)

            else:

                label, probability, cleaned = predict_ml(text)

        confidence = (
            probability
            if label == "Positif"
            else 1 - probability
        )

        if label == "Positif":

            st.success(f"😊 {label}")

        else:

            st.error(f"😞 {label}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Prediction",
                label
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with col3:
            st.metric(
                "P(Positive)",
                f"{probability:.4f}"
            )

        st.progress(float(probability))

        with st.expander("Hasil Preprocessing"):
            st.write(cleaned)

        with st.expander("Original Text"):
            st.write(text)

# =========================
# FOOTER
# =========================

st.markdown("---")
st.caption("Sentiment Analysis | Logistic Regression + LSTM")
