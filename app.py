"""
Gesture Recognition Live Streamlit Application
==============================================
Real-time hand gesture recognition using OpenCV, MediaPipe, and Scikit-Learn.
Features:
- Live webcam feed with 21 hand landmarks and bounding box.
- Neon Cyan predicted label and confidence.
- Live system metrics: Inference Latency (ms) and Pipeline FPS.
- Model selector: Toggle between Invariant Model and Raw Model live.
- Sleek dark-mode dashboard.
"""

import os
import sys
import time
import urllib.request
import cv2
import joblib
import numpy as np
import streamlit as st

from features import GestureFeatureExtractor

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisionaryRaj | Live Gesture AI",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Theme CSS
st.markdown("""
<style>
    /* Dark Theme Background */
    .stApp {
        background-color: #0b0e14;
        color: #e6edf3;
    }

    /* Custom Header */
    .header-container {
        padding: 0.8rem 1.5rem;
        background: linear-gradient(90deg, #131b26 0%, #0d1117 100%);
        border-radius: 10px;
        border: 1px solid #21262d;
        margin-bottom: 1.2rem;
    }
    .header-title {
        color: #00e5ff;
        font-size: 1.6rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.5px;
    }
    .header-subtitle {
        color: #8b949e;
        font-size: 0.9rem;
        margin-top: 0.2rem;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricValue"] {
        color: #00e5ff !important;
        font-size: 1.7rem !important;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e !important;
        font-size: 0.85rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Status Indicator Badge */
    .badge-online {
        display: inline-block;
        padding: 4px 10px;
        background-color: rgba(0, 229, 255, 0.15);
        color: #00e5ff;
        border: 1px solid #00e5ff;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Constants & Model Paths
# -----------------------------------------------------------------------------
INVARIANT_MODEL_PATH = "gesture_model.pkl"
RAW_MODEL_PATH = "raw_gesture_model.pkl"
TASK_MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# MediaPipe Hand Connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                 # Palm
]
FINGERTIP_INDICES = {4, 8, 12, 16, 20}


# -----------------------------------------------------------------------------
# Detector & Model Cache
# -----------------------------------------------------------------------------
def ensure_task_model(model_path=TASK_MODEL_PATH):
    """Ensures MediaPipe Hand Landmarker model file is available."""
    if not os.path.exists(model_path):
        with st.spinner("Downloading MediaPipe task model..."):
            urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


class HandDetector:
    """Supports both MediaPipe Tasks API (>= 0.10.30) and legacy solutions."""
    def __init__(self, model_path=TASK_MODEL_PATH):
        import mediapipe as mp
        self.mp = mp
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.use_legacy = True
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            self.use_legacy = False
            ensure_task_model(model_path)
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr):
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if self.use_legacy:
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                return results.multi_hand_landmarks[0].landmark
            return None
        else:
            mp_image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                return results.hand_landmarks[0]
            return None


@st.cache_resource
def load_models():
    """Loads Invariant and Raw RandomForest models from disk."""
    models = {}
    if os.path.exists(INVARIANT_MODEL_PATH):
        models["Invariant Model"] = joblib.load(INVARIANT_MODEL_PATH)
    if os.path.exists(RAW_MODEL_PATH):
        models["Raw Model"] = joblib.load(RAW_MODEL_PATH)
    return models


@st.cache_resource
def get_detector():
    """Caches the hand detector to avoid re-initializing on each run."""
    return HandDetector()


@st.cache_resource
def get_extractor():
    """Caches the feature extractor."""
    return GestureFeatureExtractor()


# -----------------------------------------------------------------------------
# Drawing Helpers
# -----------------------------------------------------------------------------
def draw_landmarks_and_bbox(frame, landmarks, label_text, confidence_score=None):
    """
    Renders:
    1. Skeleton lines and joints.
    2. Hand bounding box.
    3. Neon Cyan label text above the bounding box.
    """
    h, w, _ = frame.shape
    pixel_coords = []
    xs, ys = [], []

    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        pixel_coords.append((cx, cy))
        xs.append(cx)
        ys.append(cy)

    # 1. Draw Skeleton Lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
            pt1 = pixel_coords[start_idx]
            pt2 = pixel_coords[end_idx]
            cv2.line(frame, pt1, pt2, (0, 230, 115), 2, cv2.LINE_AA)

    # 2. Draw Landmark Joints
    for idx, pt in enumerate(pixel_coords):
        if idx in FINGERTIP_INDICES:
            cv2.circle(frame, pt, 6, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)

    # 3. Compute Bounding Box with Padding
    pad = 20
    x_min = max(0, min(xs) - pad)
    x_max = min(w, max(xs) + pad)
    y_min = max(0, min(ys) - pad)
    y_max = min(h, max(ys) + pad)

    # Neon Cyan BGR is (255, 255, 0)
    neon_cyan = (255, 255, 0)

    # Draw Bounding Box (Thick 2px)
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), neon_cyan, 2, cv2.LINE_AA)

    # 4. Neon Cyan Label Above Bounding Box
    if confidence_score is not None:
        display_str = f"{label_text} ({confidence_score * 100:.1f}%)"
    else:
        display_str = label_text

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(display_str, font, font_scale, thickness)

    # Label background badge
    badge_y1 = max(0, y_min - text_h - 14)
    badge_y2 = y_min
    badge_x2 = min(w, x_min + text_w + 16)
    cv2.rectangle(frame, (x_min, badge_y1), (badge_x2, badge_y2), (15, 18, 24), -1)
    cv2.rectangle(frame, (x_min, badge_y1), (badge_x2, badge_y2), neon_cyan, 1)

    # Neon Cyan Text
    cv2.putText(
        frame,
        display_str,
        (x_min + 8, y_min - 7),
        font,
        font_scale,
        neon_cyan,
        thickness,
        cv2.LINE_AA
    )


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # Top Header Banner
    st.markdown("""
    <div class="header-container">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h1 class="header-title">🖐️ VISIONARY LIVE: GESTURE AI</h1>
                <div class="header-subtitle">Real-time Landmark Tracking & Invariant Machine Learning Classifier</div>
            </div>
            <div>
                <span class="badge-online">● PIPELINE ACTIVE</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Load Models and Services
    models = load_models()
    if not models:
        st.error("No models found! Please run `python train_model.py` to generate `gesture_model.pkl`.")
        return

    detector = get_detector()
    extractor = get_extractor()

    # Split screen into two columns
    col_video, col_dashboard = st.columns([1.6, 1.0], gap="medium")

    # -------------------------------------------------------------------------
    # Right Column: Dashboard & Controls
    # -------------------------------------------------------------------------
    with col_dashboard:
        st.subheader("⚙️ System Metrics & Controls")

        # Model Selection Dropdown
        available_model_names = list(models.keys())
        default_index = 0
        selected_model_name = st.selectbox(
            "Active Model Architecture",
            options=available_model_names,
            index=default_index,
            help="Compare the Invariant Model against the Raw Model live under translation and scale shifts."
        )

        active_model = models[selected_model_name]
        is_invariant = (selected_model_name == "Invariant Model")

        if is_invariant:
            st.info("🛡️ **Invariant Mode Active**: Features are normalized by wrist origin and middle knuckle distance.")
        else:
            st.warning("⚠️ **Raw Mode Active**: Features are absolute camera coordinates. Vulnerable to hand movement.")

        st.markdown("---")

        # Metric Cards Layout
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            metric_fps = st.metric(label="Pipeline FPS", value="0.0 FPS")
        with m_col2:
            metric_latency = st.metric(label="Inference Latency", value="0.00 ms")

        # Status & Class Cards
        s_col1, s_col2 = st.columns(2)
        with s_col1:
            metric_prediction = st.metric(label="Predicted Class", value="Waiting...")
        with s_col2:
            metric_confidence = st.metric(label="Confidence", value="--%")

        st.markdown("---")

        # Camera Configuration
        st.write("**Camera Controls**")
        cam_index = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0, step=1)
        run_camera = st.toggle("Start Webcam Stream", value=True)

        st.caption("Tip: Move your hand left/right or step back to witness Invariant vs Raw model stability.")

    # -------------------------------------------------------------------------
    # Left Column: Video Feed
    # -------------------------------------------------------------------------
    with col_video:
        st.subheader("📹 Live Video Feed")
        video_placeholder = st.empty()

    # -------------------------------------------------------------------------
    # Video Streaming & Inference Loop
    # -------------------------------------------------------------------------
    if run_camera:
        cap = cv2.VideoCapture(int(cam_index))
        if not cap.isOpened():
            st.error(f"Failed to open webcam (device index {cam_index}). Please verify webcam access or choose another index.")
            return

        # Optimization: set resolution for optimal streaming FPS without websocket lag
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()
        fps_smoothing = 0.9
        current_fps = 30.0

        gesture_labels = {
            0: "GESTURE 0",
            1: "GESTURE 1",
            2: "GESTURE 2",
            3: "GESTURE 3"
        }

        frame_count = 0

        try:
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Webcam frame capture failed. Retrying...")
                    time.sleep(0.05)
                    continue

                # Mirror selfie view
                frame = cv2.flip(frame, 1)

                # Detect Hand Landmarks
                landmarks = detector.detect(frame)

                pred_class_str = "No Hand"
                conf_str = "--"
                latency_ms = 0.0

                if landmarks is not None:
                    # Feature Extraction
                    if is_invariant:
                        features = extractor.extract_invariant(landmarks).reshape(1, -1)
                    else:
                        features = extractor.extract_raw(landmarks).reshape(1, -1)

                    # Model Prediction & Latency Timing
                    t0 = time.perf_counter()
                    pred_class = active_model.predict(features)[0]
                    t1 = time.perf_counter()
                    latency_ms = (t1 - t0) * 1000.0

                    # Confidence estimation if available
                    conf_score = None
                    if hasattr(active_model, "predict_proba"):
                        proba = active_model.predict_proba(features)[0]
                        conf_score = float(np.max(proba))
                        conf_str = f"{conf_score * 100:.1f}%"

                    pred_class_str = gesture_labels.get(int(pred_class), f"CLASS {pred_class}")

                    # Draw Skeleton, Bounding Box, and Neon Cyan Tag
                    draw_landmarks_and_bbox(frame, landmarks, pred_class_str, conf_score)

                # FPS Calculation
                now = time.time()
                elapsed = now - prev_time
                prev_time = now
                if elapsed > 0:
                    instant_fps = 1.0 / elapsed
                    current_fps = (fps_smoothing * current_fps) + ((1.0 - fps_smoothing) * instant_fps)

                # Convert frame for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                # Update Dashboard Metrics every 2 frames for smooth UI
                frame_count += 1
                if frame_count % 2 == 0:
                    metric_fps.metric(label="Pipeline FPS", value=f"{current_fps:.1f} FPS")
                    metric_latency.metric(label="Inference Latency", value=f"{latency_ms:.2f} ms")
                    metric_prediction.metric(label="Predicted Class", value=pred_class_str)
                    metric_confidence.metric(label="Confidence", value=conf_str)

        finally:
            cap.release()
    else:
        video_placeholder.info("Camera stream stopped. Toggle 'Start Webcam Stream' in the dashboard to resume.")


if __name__ == "__main__":
    main()
