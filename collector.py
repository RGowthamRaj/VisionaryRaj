"""
Hand Gesture Data Collector
===========================
Uses OpenCV and MediaPipe to detect hand landmarks and collect training data.
- Switch between 4 classes (0, 1, 2, 3) using keys '1', '2', '3', and '4'.
- Press 's' to save a single sample (21 raw coordinates x, y, z).
- Press 'r' to toggle Continuous Auto-Record (collects smoothly while moving hand).
- Displays active class, target progress (e.g. 150/200), and detection status on screen.
"""

import argparse
import csv
import os
import sys
import time
import urllib.request
import cv2
import numpy as np

# Try importing mediapipe
try:
    import mediapipe as mp
except ImportError:
    print("Error: MediaPipe is not installed. Please install it using: pip install mediapipe")
    sys.exit(1)


# Hand landmark connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky finger
    (0, 17)                                 # Palm base
]

FINGERTIP_INDICES = {4, 8, 12, 16, 20}

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
DEFAULT_MODEL_PATH = "hand_landmarker.task"


def ensure_model_file(model_path=DEFAULT_MODEL_PATH):
    """Downloads the MediaPipe hand landmarker model if not present."""
    if not os.path.exists(model_path):
        print(f"Downloading hand landmarker model to '{model_path}'...")
        try:
            urllib.request.urlretrieve(MODEL_URL, model_path)
            print("Model downloaded successfully.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise
    return model_path


class HandDetector:
    """
    Unified Hand Detector supporting both:
    1. MediaPipe Tasks API (modern mediapipe >= 0.10.30, 1.0+)
    2. MediaPipe Solutions API (legacy mediapipe < 0.10.30)
    """
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            print("[INFO] Using legacy MediaPipe Solutions API.")
            self.use_legacy = True
            self.detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        else:
            print("[INFO] Using modern MediaPipe Tasks HandLandmarker API.")
            self.use_legacy = False
            ensure_model_file(model_path)
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
        """
        Processes a BGR frame and returns:
        - landmarks: list of 21 objects with .x, .y, .z attributes, or None if no hand detected.
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.use_legacy:
            results = self.detector.process(rgb_frame)
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
                return results.multi_hand_landmarks[0].landmark
            return None
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = self.detector.detect(mp_image)
            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                return results.hand_landmarks[0]
            return None


def draw_hand_landmarks(frame, landmarks):
    """Renders sleek lines and joints for the detected hand landmarks."""
    h, w, _ = frame.shape
    pixel_coords = []
    for lm in landmarks:
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        pixel_coords.append((cx, cy))

    # Draw connection lines
    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(pixel_coords) and end_idx < len(pixel_coords):
            pt1 = pixel_coords[start_idx]
            pt2 = pixel_coords[end_idx]
            cv2.line(frame, pt1, pt2, (0, 230, 115), 2, cv2.LINE_AA)

    # Draw landmark joints
    for idx, pt in enumerate(pixel_coords):
        if idx in FINGERTIP_INDICES:
            # Highlight fingertips
            cv2.circle(frame, pt, 6, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 8, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 5, (0, 180, 80), 1, cv2.LINE_AA)


def init_csv_file(csv_path):
    """
    Initializes CSV file with header if it does not exist,
    and returns initial sample counts {class_id: count}.
    """
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    file_exists = os.path.isfile(csv_path)

    if not file_exists:
        # Build header: class, x0, y0, z0, ..., x20, y20, z20
        header = ["class"]
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        print(f"[INFO] Created new dataset file: {csv_path}")
    else:
        # Read existing counts
        try:
            with open(csv_path, mode="r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if row:
                        try:
                            cls_val = int(row[0])
                            counts[cls_val] = counts.get(cls_val, 0) + 1
                        except ValueError:
                            pass
            print(f"[INFO] Loaded existing data from {csv_path}: {counts}")
        except Exception as e:
            print(f"[WARNING] Could not parse existing CSV: {e}")

    return counts


def save_landmarks(csv_path, class_label, landmarks):
    """
    Saves the 21 landmarks (x, y, z) as a single row in the CSV file
    along with the class label. Coordinates are saved as raw values.
    """
    row = [class_label]
    for lm in landmarks:
        # Raw absolute coordinates from MediaPipe
        row.extend([lm.x, lm.y, lm.z])

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def draw_hud(frame, active_class, sample_counts, target_per_class, hand_detected, is_recording, status_msg, status_color, csv_name):
    """Draws a clean, informative overlay HUD on the frame."""
    h, w, _ = frame.shape

    # Semi-transparent top banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 115), (20, 20, 25), -1)
    # Semi-transparent bottom banner
    cv2.rectangle(overlay, (0, h - 45), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Title & Target info
    cv2.putText(frame, f"DATA COLLECTOR -> {csv_name}", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Recording badge
    if is_recording:
        # Pulsing red REC indicator
        cv2.circle(frame, (w - 280, 20), 8, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(frame, "AUTO-REC ON (Press 'r' to stop)", (w - 265, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 80, 255), 2, cv2.LINE_AA)

    # Active Class indicator with colored box
    cv2.putText(frame, "Active Class:", (15, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    class_str = f"CLASS {active_class}"
    class_color = (0, 255, 200)
    cv2.putText(frame, class_str, (135, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, class_color, 2, cv2.LINE_AA)

    # Total samples
    total_samples = sum(sample_counts.values())
    total_target = target_per_class * 4 if target_per_class > 0 else 0
    total_str = f"Total: {total_samples}" + (f" / {total_target}" if total_target > 0 else "")
    cv2.putText(frame, total_str, (15, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

    # Per-class counts breakdown with completion highlight
    col_x = 270
    for c in range(4):
        is_active = (c == active_class)
        c_count = sample_counts.get(c, 0)
        c_done = (target_per_class > 0 and c_count >= target_per_class)

        if c_done:
            tag_color = (0, 255, 100) # Bright green done
            thick = 2
            mark = " [OK]"
        elif is_active:
            tag_color = (0, 255, 255) # Cyan active
            thick = 2
            mark = ""
        else:
            tag_color = (175, 175, 175)
            thick = 1
            mark = ""

        if target_per_class > 0:
            text = f"C{c}: {c_count}/{target_per_class}{mark}"
        else:
            text = f"C{c}: {c_count}"

        cv2.putText(frame, text, (col_x + c * 115, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, tag_color, thick, cv2.LINE_AA)

    # Hand detection status indicator
    status_text = "Hand Detected" if hand_detected else "No Hand"
    dot_color = (0, 255, 100) if hand_detected else (0, 60, 255)
    cv2.circle(frame, (w - 145, 95), 7, dot_color, -1, cv2.LINE_AA)
    cv2.putText(frame, status_text, (w - 130, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    # Bottom helper bar
    help_text = "[1-4]: Class  |  [S]: Save 1  |  [R]: Auto-Record  |  [Q/ESC]: Quit"
    cv2.putText(frame, help_text, (15, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 190), 1, cv2.LINE_AA)

    # Temporary action notification message
    if status_msg:
        cv2.putText(frame, status_msg, (w - 420, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser(description="Collect hand gesture landmarks dataset.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--csv", type=str, default="train_data.csv", help="Target CSV file (default: train_data.csv)")
    parser.add_argument("--target", type=int, default=200, help="Target samples per class (default: 200)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_PATH, help="Path to hand_landmarker.task model")
    args = parser.parse_args()

    # Initialize CSV and load existing counts
    sample_counts = init_csv_file(args.csv)

    # Initialize Hand Detector
    print("[INFO] Initializing Hand Landmark Detector...")
    detector = HandDetector(model_path=args.model)

    # Open Camera
    print(f"[INFO] Opening webcam device {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam with index {args.camera}.")
        print("[HINT] If you have another camera, try passing --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    active_class = 0
    is_recording = False
    last_auto_save_time = 0
    auto_save_interval = 0.08  # Save every 80ms (~12 samples/sec) in auto-record mode

    status_message = "Ready. Press 's' (single) or 'r' (auto-record)."
    status_color = (0, 255, 200)
    status_timestamp = time.time()

    print("\n" + "=" * 60)
    print(f" HAND GESTURE COLLECTOR -> {args.csv}")
    print(f" Target per class: {args.target} samples (Total: {args.target * 4})")
    print("=" * 60)
    print(" - Keys '1', '2', '3', '4' -> Switch to Class 0, 1, 2, 3")
    print(" - Key 's'                 -> Save single sample")
    print(" - Key 'r'                 -> Toggle Continuous Auto-Record (move hand!)")
    print(" - Key 'q' or ESC          -> Quit and save")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Failed to grab frame from webcam. Retrying...")
                time.sleep(0.05)
                continue

            # Flip horizontally for natural mirror selfie view
            frame = cv2.flip(frame, 1)

            # Detect hand landmarks
            landmarks = detector.detect(frame)
            hand_detected = landmarks is not None

            # Draw hand landmarks if detected
            if hand_detected:
                draw_hand_landmarks(frame, landmarks)

            # Auto-record logic
            current_time = time.time()
            if is_recording:
                # Check if current class target reached
                if args.target > 0 and sample_counts.get(active_class, 0) >= args.target:
                    is_recording = False
                    status_message = f"Class {active_class} target reached ({args.target})! Switch class."
                    status_color = (0, 255, 100)
                    status_timestamp = current_time
                    print(f"\n[DONE] Class {active_class} reached target {args.target}! Switch to next class (keys 1-4).")
                elif hand_detected and (current_time - last_auto_save_time >= auto_save_interval):
                    save_landmarks(args.csv, active_class, landmarks)
                    sample_counts[active_class] = sample_counts.get(active_class, 0) + 1
                    last_auto_save_time = current_time
                    cur_cnt = sample_counts[active_class]
                    status_message = f"Recording C{active_class}: {cur_cnt}/{args.target}"
                    status_color = (0, 255, 0)
                    status_timestamp = current_time
                    if cur_cnt % 25 == 0 or cur_cnt == args.target:
                        print(f"[{args.csv}] Class {active_class}: {cur_cnt}/{args.target} samples collected")

            # Handle status message timeout (clear after 2.5s if not recording)
            if not is_recording and (current_time - status_timestamp > 2.5):
                displayed_msg = ""
            else:
                displayed_msg = status_message

            # Draw HUD
            draw_hud(
                frame=frame,
                active_class=active_class,
                sample_counts=sample_counts,
                target_per_class=args.target,
                hand_detected=hand_detected,
                is_recording=is_recording,
                status_msg=displayed_msg,
                status_color=status_color,
                csv_name=os.path.basename(args.csv)
            )

            # Show frame
            cv2.imshow("Hand Gesture Data Collector", frame)

            # Handle key events
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                print("[INFO] Quitting application.")
                break

            # Switch classes using '1', '2', '3', '4' (and optional '0')
            elif key == ord('1'):
                active_class = 0
                status_message = f"Switched to Class 0 ({sample_counts.get(0, 0)}/{args.target})"
                status_color = (0, 255, 200)
                status_timestamp = time.time()
            elif key == ord('2'):
                active_class = 1
                status_message = f"Switched to Class 1 ({sample_counts.get(1, 0)}/{args.target})"
                status_color = (0, 255, 200)
                status_timestamp = time.time()
            elif key == ord('3'):
                active_class = 2
                status_message = f"Switched to Class 2 ({sample_counts.get(2, 0)}/{args.target})"
                status_color = (0, 255, 200)
                status_timestamp = time.time()
            elif key == ord('4'):
                active_class = 3
                status_message = f"Switched to Class 3 ({sample_counts.get(3, 0)}/{args.target})"
                status_color = (0, 255, 200)
                status_timestamp = time.time()
            elif key == ord('0'):
                active_class = 0
                status_message = f"Switched to Class 0 ({sample_counts.get(0, 0)}/{args.target})"
                status_color = (0, 255, 200)
                status_timestamp = time.time()

            # Toggle Continuous Auto-Record with 'r' or 'R'
            elif key in [ord('r'), ord('R')]:
                is_recording = not is_recording
                if is_recording:
                    status_message = f"Auto-Record STARTED for Class {active_class} (move hand!)"
                    status_color = (0, 255, 0)
                    last_auto_save_time = time.time()
                else:
                    status_message = f"Auto-Record PAUSED ({sample_counts.get(active_class, 0)}/{args.target})"
                    status_color = (0, 255, 255)
                status_timestamp = time.time()

            # Save single sample when 's' or 'S' is pressed
            elif key in [ord('s'), ord('S')]:
                if hand_detected:
                    save_landmarks(args.csv, active_class, landmarks)
                    sample_counts[active_class] = sample_counts.get(active_class, 0) + 1
                    count_for_class = sample_counts[active_class]
                    status_message = f"Saved Class {active_class} ({count_for_class}/{args.target})"
                    status_color = (0, 255, 0)
                    status_timestamp = time.time()
                    print(f"[SAVED] Class {active_class} sample #{count_for_class} -> {args.csv}")
                else:
                    status_message = "No hand detected to save!"
                    status_color = (0, 60, 255)
                    status_timestamp = time.time()
                    print("[WARNING] Save skipped: No hand detected in frame.")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nCollection session ended. Summary for '{args.csv}':")
        total_collected = sum(sample_counts.values())
        print(f"Total samples: {total_collected}")
        for c in range(4):
            cnt = sample_counts.get(c, 0)
            status_str = "COMPLETE" if (args.target > 0 and cnt >= args.target) else f"{cnt}/{args.target}"
            print(f" - Class {c}: {cnt} samples ({status_str})")
        print(f"Saved in: {os.path.abspath(args.csv)}")


if __name__ == "__main__":
    main()
