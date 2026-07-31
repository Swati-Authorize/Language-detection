import streamlit as st
import pickle

# Load Model
model = pickle.load(open("model/model.pkl", "rb"))
cv = pickle.load(open("vectorizer.pkl", "rb"))

# Page Configuration
st.set_page_config(
    page_title="Language Detection",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Language Detection")
st.write("Enter any sentence below and click **Predict**.")

# User Input
text = st.text_area(
    "Enter Text",
    height=150,
    placeholder="Example: Bonjour, comment allez-vous?"
)

# Prediction
if st.button("Predict Language"):

    if text.strip() == "":
        st.warning("Please enter some text.")
    else:

        text_vector = cv.transform([text])

        prediction = model.predict(text_vector)[0]

        confidence = model.predict_proba(text_vector).max() * 100

        st.success(f"Detected Language : **{prediction}**")

        st.progress(int(confidence))

        st.write(f"Confidence : **{confidence:.2f}%**")