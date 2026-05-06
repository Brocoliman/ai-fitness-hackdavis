import streamlit as st

st.set_page_config(page_title="AI Fitness Trainer", page_icon="🏋️", layout="wide")

st.title("🏋️ AI Fitness Trainer")
st.markdown(
    """
    Squat analysis powered by **MediaPipe Pose**.  
    Use the sidebar to pick **Beginner** or **Pro** mode, then go to the 
    **Webcam** or **Video Upload** page to start analyzing.
    """
)

st.sidebar.success("Select a page above to get started.")

st.markdown("---")
st.subheader("How it works")
st.markdown(
    """
    1. MediaPipe Pose detects 33 body landmarks per frame  
    2. We compute the **knee angle** (hip→knee→ankle) for squat depth  
    3. We compute the **hip angle** (shoulder→hip→knee) for torso lean  
    4. A simple state machine tracks standing↔squatting transitions to count reps  
    5. Each rep is scored as correct or incorrect based on angle thresholds  
    """
)
