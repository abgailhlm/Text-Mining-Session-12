import os
import pickle
import joblib
import streamlit as st

from preprocessing import preprocess

MODEL_DIR = "models"

st.set_page_config(
page_title="Sentiment Analysis Ann Abigail",
page_icon="💬",
layout="wide"
)

st.markdown("""

<style>
.hero {
    background: linear-gradient(135deg,#1e3c72,#2a5298);
    padding: 30px;
    border-radius: 20px;
    color: white;
    text-align:center;
    margin-bottom:20px;
}
.stButton button {
    width:100%;
    height:50px;
    font-weight:bold;
}
</style>

""", unsafe_allow_html=True)

@st.cache_resource
def load_ml():
model = joblib.load(
os.path.join(MODEL_DIR, "ml_model.pkl")
)

```
tfidf = joblib.load(
    os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
)

return model, tfidf
```

@st.cache_resource
def load_dl():

```
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
```

def predict_ml(text):

```
model, tfidf = load_ml()

cleaned = preprocess(text)

vector = tfidf.transform([cleaned])

proba = float(
    model.predict_proba(vector)[0][1]
)

label = (
    "Positif"
    if proba >= 0.5
    else "Negatif"
)

return label, proba, cleaned
```

def predict_dl(text):

```
from tensorflow.keras.preprocessing.sequence import pad_sequences

model, tokenizer, config = load_dl()

cleaned = preprocess(text)

sequence = tokenizer.texts_to_sequences(
    [cleaned]
)

padded = pad_sequences(
    sequence,
    maxlen=config["MAX_LEN"],
    padding="post",
    truncating="post"
)

proba = float(
    model.predict(
        padded,
        verbose=0
    )[0][0]
)

label = (
    "Positif"
    if proba >= 0.5
    else "Negatif"
)

return label, proba, cleaned
```

st.markdown("""

<div class="hero">
<h1>💬 Sentiment Analysis Dashboard</h1>
<p>
Machine Learning & Deep Learning
</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:

```
st.title("📊 About")

st.markdown("""
```

### Models

✅ Logistic Regression

✅ LSTM Deep Learning

### Dataset

4142 Reviews
""")

model_choice = st.radio(
"Pilih Model",
[
"Logistic Regression",
"LSTM Deep Learning"
]
)

text = st.text_area(
"Masukkan Teks",
height=180
)

if st.button("🔍 ANALYZE"):

```
if not text.strip():

    st.warning(
        "Masukkan teks terlebih dahulu."
    )

else:

    with st.spinner(
        "Processing..."
    ):

        if model_choice == "LSTM Deep Learning":

            label, proba, cleaned = predict_dl(text)

        else:

            label, proba, cleaned = predict_ml(text)

    confidence = (
        proba
        if label == "Positif"
        else 1 - proba
    )

    if label == "Positif":

        st.success(
            f"😊 {label}"
        )

    else:

        st.error(
            f"😞 {label}"
        )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Prediction",
        label
    )

    c2.metric(
        "Confidence",
        f"{confidence*100:.2f}%"
    )

    c3.metric(
        "P(Positive)",
        f"{proba:.4f}"
    )

    st.progress(float(proba))

    with st.expander(
        "Preprocessed Text"
    ):
        st.write(cleaned)
```

st.divider()

st.caption(
"Sentiment Analysis | Logistic Regression + LSTM"
)
"""
