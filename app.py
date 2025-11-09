
import streamlit as st, joblib

st.set_page_config(page_title="Intent Classifier - TF-IDF", layout="centered")

@st.cache_resource
def load_model():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    labels = joblib.load("labels.joblib")["classes"]
    return model, vectorizer, labels

model, vectorizer, labels = load_model()

st.title("🤖 Customer Intent Classifier (TF-IDF + Logistic Regression)")
text = st.text_input("Enter a customer query:")

if st.button("Predict") and text:
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    st.success(f"Predicted Intent: **{pred}**")
