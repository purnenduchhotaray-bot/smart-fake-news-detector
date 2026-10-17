import os
import re
import joblib
import streamlit as st
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.lower().strip()
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(w) for w in tokens if w not in STOPWORDS and len(w) > 2]
    return " ".join(cleaned)

@st.cache_resource
def load_artifacts():
    if not os.path.exists("models/fake_news_model.pkl") or not os.path.exists("models/tfidf_vectorizer.pkl"):
        return None, None
    model = joblib.load("models/fake_news_model.pkl")
    vec = joblib.load("models/tfidf_vectorizer.pkl")
    return model, vec

st.title("📰 Smart Fake News Detection System")
st.write("---")

model, vectorizer = load_artifacts()

if model is None or vectorizer is None:
    st.error("Model artifacts not found! Run `python train.py` first in your terminal.")
    st.stop()

col_input, col_result = st.columns([1.2, 0.8], gap="large")

with col_input:
    st.subheader("Input Article / Headline")
    headline = st.text_input("Headline (Optional):")
    body = st.text_area("Body / Text:", height=200)
    check_btn = st.button("Verify Authenticity", type="primary")

with col_result:
    st.subheader("Verdict")
    if check_btn:
        full_text = f"{headline} {body}".strip()
        if not full_text:
            st.warning("Please type or paste some news text first.")
        else:
            cleaned = clean_text(full_text)
            transformed = vectorizer.transform([cleaned])
            dist = model.decision_function(transformed)[0]
            pred = model.predict(transformed)[0]
            prob = 1 / (1 + np.exp(-dist))
            conf = prob if pred == 1 else (1 - prob)

            if pred == 1:
                st.success("### Likely REAL News")
                st.metric("Credibility Score", f"{conf * 100:.1f}%")
            else:
                st.error("### Likely FAKE News")
                st.metric("Deception Confidence", f"{conf * 100:.1f}%")