import streamlit as st
import joblib
import numpy as np

# Page settings
st.set_page_config(page_title="Customer Intent Classifier", layout="centered")

# Load TF-IDF model and vectorizer
@st.cache_resource
def load_tfidf_model():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

model, vectorizer = load_tfidf_model()

st.title("🤖 Customer Intent Classifier (TF-IDF + Logistic Regression)")
st.write("Type a customer message below to detect the intent.")

# Input box
user_input = st.text_input("💬 Enter customer query:")

if st.button("Predict Intent"):
    if user_input:
        X = vectorizer.transform([user_input])
        prediction = model.predict(X)[0]
        st.success(f"✅ Predicted Intent: **{prediction}**")
    else:
        st.warning("Please enter a message to classify.")






