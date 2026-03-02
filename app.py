

import streamlit as st
import joblib
import pickle
import re
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
# import joblib

# ---------------- CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Feedback Intelligence",
    layout="centered"
)

st.sidebar.title("📂 Analysis Mode")
mode = st.sidebar.radio(
    "Choose Mode",
    ["Single Review", "Batch CSV Analysis"]
)

# ---------------- LOAD MODELS ---------------- #

# def load_lr():
#     model = joblib.load("lr_sentiment.pkl")
#     vectorizer = joblib.load("tfidf.pkl")
#     return model, vectorizer

# import pandas as pd




@st.cache_resource
def load_lr():
    model = joblib.load(r'C:\Users\HP\DATA_SCIENCE\Projects\NLP\LR_tfidf_combined.pkl')
    return model

@st.cache_resource

def load_bilstm():
    model = load_model(
    r"C:\Users\HP\DATA_SCIENCE\Projects\NLP\sentiment_bilstm_glove_model.keras"
)


    # model = load_model("bilstm_model.keras")

    with open(r"C:\Users\HP\DATA_SCIENCE\Projects\NLP\tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    # with open("tokenizer.pkl", "rb") as f:
    #     tokenizer = pickle.load(f)
    return model, tokenizer


if mode == "Batch CSV Analysis":
    st.title("📊 Batch Review Intelligence Dashboard")

    uploaded_file = st.file_uploader(
        "Upload CSV file with customer reviews",
        type=["csv"]
    )

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        if "review" not in df.columns:
            st.error("CSV must contain a 'review' column")
        else:
            lr_model, tfidf = load_lr()

            df["review_clean"] = df["review"].astype(str).apply(clean_text)
            X_vec = tfidf.transform(df["review_clean"])

            df["sentiment_prob"] = lr_model.predict_proba(X_vec)[:, 1]
            df["sentiment"] = df["sentiment_prob"].apply(
                lambda x: "Positive" if x >= 0.5 else "Negative"
            )

            df["risk_level"] = df["sentiment_prob"].apply(
                lambda x: "High" if x < 0.4 else "Medium" if x < 0.7 else "Low"
            )

            st.success("Batch analysis completed ✅")
# ---------------- HELPERS ---------------- #
MAX_LEN = 150

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text

def risk_level(prob):
    if prob < 0.4:
        return "🔴 High Risk"
    elif prob < 0.7:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

def business_action(prob):
    if prob < 0.4:
        return "⚠ Escalate to Customer Support\n⚠ Investigate Product Issue"
    elif prob < 0.7:
        return "🔍 Monitor Customer Feedback"
    else:
        return "✅ Eligible for Marketing / Testimonials"

NEGATIVE_KEYWORDS = [
    "refund", "broken", "late", "delay",
    "bad", "poor", "defective", "waste"
]

def extract_issues(text):
    return list(set([w for w in NEGATIVE_KEYWORDS if w in text]))

# ---------------- UI ---------------- #
st.title("🧠 Customer Feedback Intelligence System")
st.write("Transform customer reviews into actionable business insights")

model_choice = st.selectbox(
    "Select Analysis Engine",
    ["Logistic Regression (Fast & Explainable)", "BiLSTM (Deep Learning)"]
)

review = st.text_area("Enter Customer Review", height=160)






if st.button("Analyze Feedback"):
    if review.strip() == "":
        st.warning("Please enter a customer review")
    else:
        review_clean = clean_text(review)

        # -------- Prediction -------- #
       

        title = st.text_input("Review title (optional)")
        # review_clean = st.text_area("Review text")


        if model_choice.startswith("Logistic"):
            model = load_lr()

            X_input = pd.DataFrame({
                "title": [title],
                "review_clean": [review_clean]
            })

            prob = model.predict_proba(X_input)[0][1]

        else:
            model, tokenizer = load_bilstm()
            seq = tokenizer.texts_to_sequences([review_clean])
            pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
            prob = model.predict(pad)[0][0]

        sentiment = "POSITIVE 😊" if prob >= 0.5 else "NEGATIVE 😠"

        # -------- Business Outputs -------- #
        st.divider()
        st.subheader("📊 Analysis Summary")

        col1, col2, col3 = st.columns(3)
        col1.metric("Customer Sentiment", sentiment)
        col2.metric("Confidence", f"{prob:.2f}")
        col3.metric("Risk Level", risk_level(prob))

        st.subheader("🎯 Business Interpretation")
        st.write(f"**Recommended Action:**\n{business_action(prob)}")

        issues = extract_issues(review_clean)
        if issues:
            st.subheader("🚨 Detected Complaint Signals")
            for issue in issues:
                st.write(f"• {issue}")
        else:
            st.subheader("✅ No Critical Issues Detected")

        st.divider()
        st.caption("Designed for Product, Support & Marketing Teams")