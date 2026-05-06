import time
import os
import cv2
import numpy as np
import mediapipe as mp

from utils import (
    LANDMARKS, find_angle, find_offset_angle,
    get_landmark_coords_from_normalized, draw_text_with_bg,
)
from thresholds import OFFSET_THRESH, INACTIVE_THRESH, THRESHOLDS

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# Connections for drawing skeleton
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

MODEL_PATH = os.path.join(os.path.dirname(__file__), "pose_landmarker_heavy.task")


class SquatAnalyzer:
    """
    Stateful squat analyzer using the mp.tasks PoseLandmarker API.
    Feed it frames, it returns annotated frames with squat counting and form feedback.
    """

    def __init__(self, mode="beginner", on_rep_complete=None):
        self.mode = mode
        self.thresh = THRESHOLDS[mode]
        self.on_rep_complete = on_rep_complete

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}\n"
                "Run: python download_model.py"
            )

        # Create PoseLandmarker for VIDEO mode (sequential frames with timestamps)
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.frame_timestamp_ms = 0

        # Counters and state
        self.correct_count = 0
        self.incorrect_count = 0
        self.state = "standing"  # "standing" or "squatting"
        self.feedback = ""
        self.last_detection_time = time.time()

    def reset(self):
        self.correct_count = 0
        self.incorrect_count = 0
        self.state = "standing"
        self.feedback = ""

    def close(self):
        self.landmarker.close()

    def process_frame(self, frame):
        """
        Process a single BGR frame. Returns the annotated frame.
        """
        h, w, _ = frame.shape

        # Convert BGR -> RGB and wrap in mp.Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect — VIDEO mode requires monotonically increasing timestamps
        self.frame_timestamp_ms += 33  # ~30fps
        results = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            elapsed = time.time() - self.last_detection_time
            if elapsed > INACTIVE_THRESH:
                self.reset()
            draw_text_with_bg(frame, "No pose detected", (10, 30),
                              color=(0, 0, 255), bg_color=(40, 40, 40))
            self._draw_hud(frame)
            return frame

        self.last_detection_time = time.time()
        landmarks = results.pose_landmarks[0]  # first person

        # Draw skeleton
        self._draw_landmarks(frame, landmarks, w, h)

        # --- Extract key points ---
        nose = get_landmark_coords_from_normalized(landmarks, LANDMARKS["nose"], w, h)
        l_shoulder = get_landmark_coords_from_normalized(landmarks, LANDMARKS["left_shoulder"], w, h)
        r_shoulder = get_landmark_coords_from_normalized(landmarks, LANDMARKS["right_shoulder"], w, h)
        l_hip = get_landmark_coords_from_normalized(landmarks, LANDMARKS["left_hip"], w, h)
        r_hip = get_landmark_coords_from_normalized(landmarks, LANDMARKS["right_hip"], w, h)
        l_knee = get_landmark_coords_from_normalized(landmarks, LANDMARKS["left_knee"], w, h)
        r_knee = get_landmark_coords_from_normalized(landmarks, LANDMARKS["right_knee"], w, h)
        l_ankle = get_landmark_coords_from_normalized(landmarks, LANDMARKS["left_ankle"], w, h)
        r_ankle = get_landmark_coords_from_normalized(landmarks, LANDMARKS["right_ankle"], w, h)

        # --- Check offset (alignment) ---
        offset_angle = find_offset_angle(l_shoulder, l_hip)
        if offset_angle > OFFSET_THRESH:
            self.feedback = f"Align to side view! (offset: {offset_angle:.0f}deg)"
            draw_text_with_bg(frame, self.feedback, (10, 30),
                              color=(0, 0, 255), bg_color=(40, 40, 40))
            self._draw_hud(frame)
            return frame

        # --- Compute angles (average left + right) ---
        l_knee_angle = find_angle(l_hip, l_knee, l_ankle)
        r_knee_angle = find_angle(r_hip, r_knee, r_ankle)
        knee_angle = (l_knee_angle + r_knee_angle) / 2

        l_hip_angle = find_angle(l_shoulder, l_hip, l_knee)
        r_hip_angle = find_angle(r_shoulder, r_hip, r_knee)
        hip_angle = (l_hip_angle + r_hip_angle) / 2

        # --- State machine for squat counting ---
        t = self.thresh
        form_ok = t["hip_angle_low"] <= hip_angle <= t["hip_angle_high"]

        if self.state == "standing" and knee_angle < t["knee_angle_low"]:
            self.state = "squatting"
            if form_ok:
                self.feedback = "Good squat depth!"
            else:
                self.feedback = f"Watch your torso angle ({hip_angle:.0f}deg)"

        elif self.state == "squatting" and knee_angle > t["knee_angle_high"]:
            if form_ok:
                self.correct_count += 1
                self.feedback = "Good rep!"
            else:
                self.incorrect_count += 1
                self.feedback = "Incorrect form - check torso lean"
            self.state = "standing"

            # Fire callback with rep metrics
            if self.on_rep_complete:
                self.on_rep_complete({
                    "rep_number": self.correct_count + self.incorrect_count,
                    "is_correct": form_ok,
                    "knee_angle": knee_angle,
                    "hip_angle": hip_angle,
                    "mode": self.mode,
                    "correct_count": self.correct_count,
                    "incorrect_count": self.incorrect_count,
                })

        # --- Draw angle info ---
        draw_text_with_bg(frame, f"Knee: {knee_angle:.0f}", (10, 30),
                          color=(0, 255, 0))
        draw_text_with_bg(frame, f"Hip: {hip_angle:.0f}", (10, 60),
                          color=(0, 255, 0))
        draw_text_with_bg(frame, f"State: {self.state}", (10, 90),
                          color=(255, 255, 0))
        if self.feedback:
            draw_text_with_bg(frame, self.feedback, (10, h - 30),
                              color=(0, 255, 255), bg_color=(40, 40, 40))

        self._draw_hud(frame)
        return frame

    def _draw_landmarks(self, frame, landmarks, w, h):
        """Draw pose landmarks and connections on the frame."""
        for connection in POSE_CONNECTIONS:
            start = landmarks[connection.start]
            end = landmarks[connection.end]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (245, 66, 230), 2)

        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (245, 117, 66), -1)

    def _draw_hud(self, frame):
        """Draw the rep counter overlay."""
        h, w, _ = frame.shape
        draw_text_with_bg(frame, f"Correct: {self.correct_count}", (w - 200, 30),
                          color=(0, 255, 0), bg_color=(40, 40, 40))
        draw_text_with_bg(frame, f"Incorrect: {self.incorrect_count}", (w - 200, 60),
                          color=(0, 0, 255), bg_color=(40, 40, 40))
        draw_text_with_bg(frame, f"Mode: {self.mode}", (w - 200, 90),
                          color=(255, 255, 255), bg_color=(40, 40, 40))
