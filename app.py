import streamlit as st
import joblib
import numpy as np
import re
import tflite_runtime.interpreter as tflite  # ✅ lightweight alternative to TensorFlow Lite

st.set_page_config(page_title="Customer Intent Classifier", layout="centered")

# =============== Load TF-IDF Model ===============
@st.cache_resource
def load_tfidf():
    model = joblib.load("models/tfidf/tfidf_model.joblib")
    vectorizer = joblib.load("models/tfidf/tfidf_vectorizer.joblib")
    labels = joblib.load("models/tfidf/labels.joblib")["classes"]
    return model, vectorizer, labels

# =============== Load BiLSTM TFLite Model ===============
@st.cache_resource
def load_bilstm():
    interpreter = tflite.Interpreter(model_path="models/bilstm_lite/bilstm_model.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    tokenizer = joblib.load("models/bilstm_lite/tokenizer.joblib")   # from training
    labels = joblib.load("models/bilstm_lite/labels.joblib")["classes"]
    return interpreter, tokenizer, labels, input_details, output_details

def preprocess_text(text, tokenizer, max_len=50):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    seq = tokenizer.texts_to_sequences([text])
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    return padded

def predict_bilstm(interpreter, input_details, output_details, input_data):
    interpreter.set_tensor(input_details[0]['index'], input_data.astype(np.int32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return output

# =============== UI ===============
st.title("🤖 Customer Intent Classifier")
st.caption("Models: TF-IDF + Logistic Regression | BiLSTM (TFLite)")

model_option = st.radio("Choose model:", ["TF-IDF", "BiLSTM (TFLite)"])
user_input = st.text_area("Enter a customer query:", placeholder="e.g., I want to cancel my order")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter a message.")
    else:
        try:
            if model_option == "TF-IDF":
                model, vectorizer, labels = load_tfidf()
                prediction = model.predict(vectorizer.transform([user_input]))[0]
                st.success(f"✅ Predicted Intent: **{prediction}**")

            else:  # BiLSTM TFLite
                interpreter, tokenizer, labels, input_details, output_details = load_bilstm()
                data = preprocess_text(user_input, tokenizer)
                probs = predict_bilstm(interpreter, input_details, output_details, data)
                st.success(f"✅ Predicted Intent: **{labels[int(np.argmax(probs))]}**")

        except Exception as e:
            st.error(f"❌ Error: {e}")




