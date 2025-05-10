
import streamlit as st
import pandas as pd
import pickle

st.title("🧠 AutoML Predictor")

uploaded_file = st.file_uploader("Upload CSV to Predict")
if not uploaded_file:
    st.stop()

df = pd.read_csv(uploaded_file)

st.write("Preview of uploaded data:")
st.write(df.head())

try:
    model = pickle.load(open("trained_model.pkl", "rb"))
    preds = model.predict(df)
    st.write("### Predictions:")
    st.write(preds)
except Exception as e:
    st.error(f"Failed to predict: {e}")
