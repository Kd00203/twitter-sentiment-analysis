import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    layout="centered"
)

MODEL_PATH = "tweets_bilstm.h5"
TOKENIZER_PATH = "model_tokenizer.joblib"

ID_TO_LABEL = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

def clean_text(text: str) -> str:
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@[A-Za-z0-9_]+", " ", text)
    text = re.sub(r"[^a-zA-Z\s!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    tokenizer = joblib.load(TOKENIZER_PATH)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=40, padding="post", truncating="post")
    prob = model.predict(padded)[0]
    label_id = int(np.argmax(prob))
    return ID_TO_LABEL[label_id], prob

st.title("🐦 Twitter Sentiment Analysis")

tweet_input = st.text_area(
    "Enter a tweet",
    height=150,
    placeholder="Type your tweet here..."
)

if st.button("Analyze Sentiment"):
    if tweet_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        label, scores = predict_sentiment(tweet_input)
        st.success(f"Sentiment: **{label}**")
        st.write({
            "Negative": float(scores[0]),
            "Neutral": float(scores[1]),
            "Positive": float(scores[2])
        })

