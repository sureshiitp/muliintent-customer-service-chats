import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# ------------------ PAGE SETTINGS ------------------
st.set_page_config(page_title="Customer Intent Classifier", layout="centered")

st.title("🤖 Customer Intent Classifier")
st.write("Select a model and enter a customer message to predict intent.")

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_tfidf_model():
    model = joblib.load("tfidf_model.joblib")
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    return model, vectorizer

@st.cache_resource
def load_bilstm_model():
    model = tf.keras.models.load_model("bilstm_model.keras", compile=False)
    tokenizer = joblib.load("tokenizer.joblib")
    label_data = joblib.load("labels.joblib")["classes"]
    return model, tokenizer, label_data

# ------------------ USER SELECTION ------------------
model_choice = st.radio("Choose a model:", ["TF-IDF + Logistic Regression", "BiLSTM + Attention"])

user_input = st.text_input("💬 Enter customer query:")

if st.button("Predict") and user_input:

    if model_choice == "TF-IDF + Logistic Regression":
        model, vectorizer = load_tfidf_model()
        features = vectorizer.transform([user_input])
        prediction = model.predict(features)[0]
        st.success(f"✨ Predicted Intent (TF-IDF): **{prediction}**")

    elif model_choice == "BiLSTM + Attention":
        model, tokenizer, labels = load_bilstm_model()
        seq = tokenizer.texts_to_sequences([user_input])
        padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=40)
        pred = np.argmax(model.predict(padded), axis=1)[0]
        st.success(f"✨ Predicted Intent (BiLSTM): **{labels[pred]}**")






