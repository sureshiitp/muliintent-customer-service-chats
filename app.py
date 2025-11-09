import streamlit as st
import joblib
import numpy as np
import re
import tflite_runtime.interpreter as tflite  # ✅ lightweight TFLite runtime

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
# ✅ 2) Load BiLSTM (TFLite model)
# =========================
@st.cache_resource
def load_bilstm():
    interpreter = tflite.Interpreter(model_path="models/bilstm_lite/bilstm_model.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tokenizer = joblib.load("models/bilstm_lite/tokenizer.joblib")
    labels = joblib.load("models/bilstm_lite/labels.joblib")["classes"]
    return interpreter, tokenizer, labels, input_details, output_details

# ✅ Text preprocessing for BiLSTM
def preprocess_text(text, tokenizer, max_len=50):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    seq = tokenizer.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences  # safe for preprocessing only
    padded = pad_sequences(seq, maxlen=max_len, padding='post')
    return padded

def predict_bilstm(interpreter, input_details, output_details, padded):
    interpreter.set_tensor(input_details[0]["index"], padded.astype(np.int32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output

# =========================
# 🎛 UI Layout
# =========================
st.title("🤖 Customer Intent Classifier")
st.caption("Models: TF-IDF + Logistic Regression | BiLSTM (TFLite)")

model_choice = st.radio("Choose a model:", ["TF-IDF + Logistic Regression", "BiLSTM (TFLite)"])

user_input = st.text_area("Enter a customer message:", placeholder="e.g., I want to cancel my order")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠ Please enter a message!")
    else:
        try:
            if model_choice == "TF-IDF + Logistic Regression":
                model, vectorizer, labels = load_tfidf()
                X = vectorizer.transform([user_input])
                pred = model.predict(X)[0]
                st.success(f"✅ Predicted Intent: **{pred}**")

            else:
                interpreter, tokenizer, labels, input_details, output_details = load_bilstm()
                padded = preprocess_text(user_input, tokenizer)
                probs = predict_bilstm(interpreter, input_details, output_details, padded)
                pred = labels[int(np.argmax(probs))]
                st.success(f"✅ Predicted Intent: **{pred}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")


