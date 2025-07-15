import streamlit as st
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import os
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load model
model = joblib.load ("model.keras")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Ensure folders
os.makedirs("data/captured", exist_ok=True)
log_path = "data/emotion_log.csv"

# Face detection
face_cascade = cv2.CascadeClassifier("haarcascade.xml")

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Login"
if 'student_name' not in st.session_state:
    st.session_state.student_name = ""
if 'student_id' not in st.session_state:
    st.session_state.student_id = ""

# Sidebar navigation
st.sidebar.title("📚 Navigation")
if st.sidebar.button("🔐 Login"):
    st.session_state.page = "Login"
if st.sidebar.button("📸 Capture"):
    st.session_state.page = "Capture"
if st.sidebar.button("📊 Analysis"):
    st.session_state.page = "Analysis"

# ---------------------- PAGE 1: LOGIN ----------------------
if st.session_state.page == "Login":
    st.title("🔐 Student Login")
    with st.form("login_form"):
        st.session_state.student_name = st.text_input("Enter Student Name")
        st.session_state.student_id = st.text_input("Enter Student ID")
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.success("Login successful! Now go to Capture section.")

# ---------------------- PAGE 2: CAPTURE ----------------------
elif st.session_state.page == "Capture":
    st.title("📸 Capture Emotion")

    if not st.session_state.student_name or not st.session_state.student_id:
        st.warning("Please login first.")
    else:
        camera_image = st.camera_input("Take a picture")

        if camera_image:
            img_path = f"data/captured/{st.session_state.student_name}_{st.session_state.student_id}.jpg"
            with open(img_path, "wb") as f:
                f.write(camera_image.getbuffer())

            st.success("Image captured successfully!")

            # Load image
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            predicted_emotion = "Unknown"

            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face = cv2.resize(face, (48, 48)) / 255.0
                face = np.expand_dims(face, axis=(0, -1))
                prediction = model.predict(face)
                emotion_idx = np.argmax(prediction)
                predicted_emotion = emotion_labels[emotion_idx]
                break

            # Log
            with open(log_path, "a") as f:
                f.write(f"{datetime.now().isoformat()},{st.session_state.student_name},{st.session_state.student_id},{predicted_emotion}\n")

            # Show output
            st.subheader(f"🧠 Prediction: {predicted_emotion}")
            st.image(Image.open(img_path), caption=f"{st.session_state.student_name} - {predicted_emotion}", width=300)

            if predicted_emotion == "Sad":
                st.warning("The student seems sad. Consider the following:")
                st.markdown("- Offer support or mentorship")
                st.markdown("- Encourage breaks or relaxation")
                st.markdown("- Provide positive feedback")

# ---------------------- PAGE 3: ANALYSIS ----------------------
elif st.session_state.page == "Analysis":
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
        st.info("No emotion logs found.")
