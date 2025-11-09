import streamlit as st
import joblib
import numpy as np
import tensorflow as tf
import re

st.set_page_config(page_title="Customer Intent Classifier", layout="centered")

# =========================
# ✅ 1) Load TF-IDF model
# =========================
@st.cache_resource
def load_tfidf():
    model = joblib.load("models/tfidf/tfidf_model.joblib")
    vectorizer = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
    labels = joblib.load("models/tfidf/labels.joblib")["classes"]
    return model, vectorizer, labels

# =========================
# ✅ 2) Load BiLSTM (TFLite)
# =========================
@st.cache_resource
def load_bilstm():
    interpreter = tf.lite.Interpreter(model_path="models/bilstm_lite/bilstm_model.tflite")
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    tokenizer = joblib.load("models/bilstm_lite/tokenizer.joblib")
    labels = joblib.load("models/bilstm_lite/labels.joblib")["classes"]
    return interpreter, tokenizer, labels, input_details, output_details

# ✅ Tokenizing for BiLSTM
def preprocess_text(text, tokenizer, max_len=50):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def predict_bilstm(interpreter, input_details, output_details, padded):
    interpreter.set_tensor(input_details[0]['index'], padded.astype(np.int32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# =========================
# 🎛 UI Section
# =========================
st.title("🤖 Customer Intent Classifier")
st.caption("Models: TF-IDF + Logistic Regression | BiLSTM (TFLite)")

model_choice = st.radio("Choose Model:", 
                        ["TF-IDF + Logistic Regression", "BiLSTM (TFLite)"])

user_input = st.text_area("Enter customer message:", placeholder="e.g., I want to cancel my order")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠ Please enter a message!")
    else:
        try:
            if model_choice == "TF-IDF + Logistic Regression":
                model, vectorizer, labels = load_tfidf()
                pred = model.predict(vectorizer.transform([user_input]))[0]
                st.success(f"✅ Predicted Intent: **{pred}**")
            
            elif model_choice == "BiLSTM (TFLite)":
                interpreter, tokenizer, labels, input_details, output_details = load_bilstm()
                padded = preprocess_text(user_input, tokenizer)
                probs = predict_bilstm(interpreter, input_details, output_details, padded)
                pred = labels[int(np.argmax(probs))]
                st.success(f"✅ Predicted Intent: **{pred}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")

