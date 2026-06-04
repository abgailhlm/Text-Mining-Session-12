import os
import joblib
import streamlit as st

from preprocessing import preprocess

MODEL_DIR = "models"

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Sentiment Analysis - Ann Abigail",
    page_icon="💬",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

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

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    model = joblib.load(
        os.path.join(MODEL_DIR, "ml_model.pkl")
    )

    tfidf = joblib.load(
        os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    )

    return model, tfidf

# ==================================================
# PREDICTION
# ==================================================

def predict_sentiment(text):

    model, tfidf = load_model()

    cleaned = preprocess(text)

    vector = tfidf.transform([cleaned])

    probability = float(
        model.predict_proba(vector)[0][1]
    )

    label = (
        "Positif"
        if probability >= 0.5
        else "Negatif"
    )

    return label, probability, cleaned

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.title("📊 About")

    st.markdown("""
### Model

- Logistic Regression
- TF-IDF Vectorizer

### NLP Pipeline

✅ Case Folding

✅ Cleaning

✅ Tokenization

✅ Stopword Removal

✅ Stemming (Sastrawi)

### Dataset

4142 Data Sentimen
""")

# ==================================================
# HEADER
# ==================================================

st.markdown("""
<div class="hero">
<h1>💬 Sentiment Analysis Dashboard</h1>
<p>
Analisis Sentimen Bahasa Indonesia menggunakan
Machine Learning dan NLP
</p>
</div>
""", unsafe_allow_html=True)

# ==================================================
# INPUT
# ==================================================

col1, col2 = st.columns([3,1])

with col1:

    text = st.text_area(
        "Masukkan Ulasan",
        height=180,
        placeholder="Contoh: Barangnya bagus banget, pengiriman cepat dan seller ramah."
    )

with col2:

    st.markdown("### Contoh Input")

    if st.button("Positif Example"):
        st.session_state.example = (
            "Barangnya bagus banget dan pengiriman cepat."
        )

    if st.button("Negatif Example"):
        st.session_state.example = (
            "Produk mengecewakan dan kualitas buruk."
        )

# ==================================================
# ANALYZE BUTTON
# ==================================================

if st.button("🔍 ANALYZE SENTIMENT", type="primary"):

    if not text.strip():

        st.warning(
            "Masukkan teks terlebih dahulu."
        )

    else:

        with st.spinner(
            "Menganalisis sentimen..."
        ):

            label, proba, cleaned = predict_sentiment(text)

        confidence = (
            proba
            if label == "Positif"
            else 1 - proba
        )

        st.divider()

        if label == "Positif":

            st.markdown(
                """
                <div class="result-positive">
                    <h2>😊 Sentimen Positif</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-negative">
                    <h2>😞 Sentimen Negatif</h2>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Prediction",
                label
            )

        with c2:
            st.metric(
                "Confidence",
                f"{confidence*100:.2f}%"
            )

        with c3:
            st.metric(
                "P(Positive)",
                f"{proba:.4f}"
            )

        st.write("")

        st.progress(float(proba))

        with st.expander(
            "🔎 Hasil Preprocessing"
        ):
            st.write(cleaned)

        with st.expander(
            "📄 Original Text"
        ):
            st.write(text)

# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Text Mining Session 12 | Logistic Regression + TF-IDF"
)
