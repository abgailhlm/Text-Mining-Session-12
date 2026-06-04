"""
Streamlit App - Sentiment Analysis
===================================
Untuk deploy di Streamlit Cloud.

Jalankan lokal:
    streamlit run streamlit_app.py

Deploy:
    1. Push folder ini ke GitHub (sertakan folder models/ hasil train.py)
    2. Buka https://share.streamlit.io -> New app -> pilih repo
    3. Main file: streamlit_app.py
"""
import os
import pickle

import joblib
import streamlit as st

from preprocessing import preprocess

MODEL_DIR = "models"

st.set_page_config(page_title="Analisis Sentimen", page_icon="💬", layout="centered")


@st.cache_resource
def load_ml():
    model = joblib.load(os.path.join(MODEL_DIR, "ml_model.pkl"))
    tfidf = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    return model, tfidf


@st.cache_resource
def load_dl():
    from tensorflow.keras.models import load_model
    model = load_model(os.path.join(MODEL_DIR, "dl_model.h5"))
    with open(os.path.join(MODEL_DIR, "tokenizer.pkl"), "rb") as f:
        tok = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "config.pkl"), "rb") as f:
        cfg = pickle.load(f)
    return model, tok, cfg


def predict_ml(text):
    model, tfidf = load_ml()
    vec = tfidf.transform([preprocess(text)])
    proba = float(model.predict_proba(vec)[0][1])
    return ("Positif" if proba >= 0.5 else "Negatif"), proba


def predict_dl(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    model, tok, cfg = load_dl()
    seq = pad_sequences(tok.texts_to_sequences([preprocess(text)]),
                        maxlen=cfg["MAX_LEN"], padding="post", truncating="post")
    proba = float(model.predict(seq, verbose=0)[0][0])
    return ("Positif" if proba >= 0.5 else "Negatif"), proba


st.title("💬 Analisis Sentimen")
st.caption("Klasifikasi teks menjadi sentimen positif / negatif.")

model_choice = st.radio(
    "Pilih model:",
    ["Machine Learning (Logistic Regression)", "Deep Learning (LSTM)"],
    horizontal=False,
)

text = st.text_area("Masukkan teks:", height=140,
                    placeholder="Contoh: barangnya bagus banget pengiriman cepat...")

if st.button("Analisis", type="primary"):
    if not text.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        with st.spinner("Memproses..."):
            if model_choice.startswith("Deep"):
                label, proba = predict_dl(text)
            else:
                label, proba = predict_ml(text)

        conf = proba if label == "Positif" else 1 - proba
        if label == "Positif":
            st.success(f"😊 **{label}**")
        else:
            st.error(f"😞 **{label}**")

        st.metric("Confidence", f"{conf*100:.1f}%")
        st.progress(proba)
        st.caption(f"P(positif) = {proba:.4f}")
