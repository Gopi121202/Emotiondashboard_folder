import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and labels
model = load_model("model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure required folders exist
os.makedirs("data/captured", exist_ok=True)
log_path = "data/emotion_log.csv"

# Face detection model
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# Sidebar navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["🏫 Login", "📷 Capture & Predict", "📈 Analysis"])

# Shared session state
if 'student_name' not in st.session_state:
    st.session_state.student_name = ""
if 'student_id' not in st.session_state:
    st.session_state.student_id = ""

# ---------------- SECTION: LOGIN ----------------
if section == "🏫 Login":
    st.title("📋 Student Login")
    with st.form("login_form"):
        st.session_state.student_name = st.text_input("Enter Student Name")
        st.session_state.student_id = st.text_input("Enter Student ID")
        submitted = st.form_submit_button("Login")
        if submitted:
            st.success("Login successful! Proceed to Capture section.")

# ---------------- SECTION: CAPTURE & PREDICT ----------------
elif section == "📷 Capture & Predict":
    st.title("📸 Capture Emotion")

    if not st.session_state.student_name or not st.session_state.student_id:
        st.warning("Please login first from the 'Login' section.")
    else:
        camera_image = st.camera_input("Take a picture")
        if camera_image:
            st.success("Image captured successfully!")
            img_path = f"data/captured/{st.session_state.student_name}_{st.session_state.student_id}.jpg"

            # Save image
            with open(img_path, "wb") as f:
                f.write(camera_image.getbuffer())

            # Load and preprocess
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            predicted_emotion = "Unknown"
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48)) / 255.0
                face = np.expand_dims(face, axis=(0, -1))
                prediction = model.predict(face)
                predicted_emotion = emotion_labels[np.argmax(prediction)]
                break

            # Log result
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()},{st.session_state.student_name},{st.session_state.student_id},{predicted_emotion}\n")

            st.subheader(f"Prediction: {predicted_emotion}")
            st.image(Image.open(img_path), caption=f"{st.session_state.student_name} - {predicted_emotion}", width=300)

            if predicted_emotion == "Sad":
                st.warning("The student seems sad. Consider the following improvements:")
                st.markdown("- Offer personal support or mentorship")
                st.markdown("- Encourage breaks or lighter activities")
                st.markdown("- Provide positive feedback or recognition")

# ---------------- SECTION: ANALYSIS ----------------
elif section == "📈 Analysis":
    st.title("📊 Emotion Analysis")
    if os.path.exists(log_path):
        df = pd.read_csv(log_path, names=["Time", "Name", "ID", "Emotion"], parse_dates=["Time"])
        student_df = df[df["ID"] == st.session_state.student_id]

        st.subheader("📈 Emotion Trend for Student")
        if not student_df.empty:
            st.line_chart(student_df["Emotion"].value_counts())
        else:
            st.info("No previous emotion data found for this student.")

        st.subheader("📊 Overall Emotion Distribution")
        st.bar_chart(df["Emotion"].value_counts())
    else:
        st.info("No emotion logs available.")
