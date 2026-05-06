import cv2
import streamlit as st
import tempfile
import numpy as np

# Add parent dir to path so imports work from pages/
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analyzer import SquatAnalyzer

st.set_page_config(page_title="Video Analysis", page_icon="📹", layout="wide")
st.title("📹 Video Upload Analysis")

mode = st.sidebar.selectbox("Mode", ["beginner", "pro"])
uploaded = st.file_uploader("Upload a video of squats", type=["mp4", "avi", "mov", "mkv"])

if uploaded is not None:
    # Save to temp file so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    analyzer = SquatAnalyzer(mode=mode)

    stframe = st.empty()
    stop_btn = st.button("Stop")

    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            break

        frame = analyzer.process_frame(frame)
        # Convert BGR -> RGB for Streamlit
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

    cap.release()
    os.unlink(tfile.name)

    st.success(
        f"Done! Correct: {analyzer.correct_count} | "
        f"Incorrect: {analyzer.incorrect_count}"
    )
else:
    st.info("Upload a video to get started. For webcam, run with OpenCV directly (see README).")
