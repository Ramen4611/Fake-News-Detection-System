# app.py — Streamlit Fake News Detection App (ZIP Upload Enabled)

import streamlit as st
import pandas as pd
import re
import zipfile
import os
import tempfile
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords

nltk.download('stopwords')

# --- Text Cleaning ---
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    stop_words = set(stopwords.words("english"))
    return " ".join([word for word in text.split() if word not in stop_words])

# --- Load ZIP from Streamlit Upload ---
def load_from_streamlit_zip(zip_file_obj):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_zip_path = os.path.join(tmpdir, "temp.zip")

        with open(temp_zip_path, "wb") as f:
            f.write(zip_file_obj.getvalue())

        with zipfile.ZipFile(temp_zip_path, 'r') as z:
            z.extractall(tmpdir)

        fake_path = os.path.join(tmpdir, "Fake.csv")
        true_path = os.path.join(tmpdir, "True.csv")

        fake = pd.read_csv(fake_path)
        true = pd.read_csv(true_path)

        fake['label'] = 'FAKE'
        true['label'] = 'REAL'

        df = pd.concat([fake[['text', 'label']], true[['text', 'label']]])
        df.dropna(inplace=True)
        df['text'] = df['text'].apply(clean_text)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df

# --- Train model ---
def train_model(df):
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=2000)
    model.fit(X_train_vec, y_train)
    return model, vectorizer, X_test_vec, y_test

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="centered")
st.title("📰 Fake News Detection System")

uploaded_zip = st.file_uploader("Upload ZIP containing Fake.csv and True.csv", type=["zip"])

if uploaded_zip is not None:
    with st.spinner("Processing and training model..."):
        df = load_from_streamlit_zip(uploaded_zip)
        model, vectorizer, X_test_vec, y_test = train_model(df)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred, labels=["REAL", "FAKE"])

        st.success(f"Model trained successfully! Accuracy: {acc:.2%}")

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("📝 Classify a News Article")
        user_input = st.text_area("Paste news article text here")

        if st.button("🔍 Predict"):
            if not user_input.strip():
                st.warning("Please enter some text to classify.")
            else:
                cleaned = clean_text(user_input)
                vec_input = vectorizer.transform([cleaned])
                proba = model.predict_proba(vec_input)[0]
                pred = model.predict(vec_input)[0]
                label_index = list(model.classes_).index(pred)
                confidence = proba[label_index] * 100
                result_label = "🟢 REAL" if pred == "REAL" else "🔴 FAKE"
                st.markdown(f"### 🧠 Prediction: {result_label} (Confidence: {confidence:.2f}%)")

# --- Footer ---
st.markdown("<hr><div style='text-align:center; color:gray;'>Built with ❤️ using Streamlit, Scikit-learn, and NLTK</div>", unsafe_allow_html=True)
