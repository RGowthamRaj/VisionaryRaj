# VisionaryRaj: Invariant Hand Gesture Recognition AI 🖐️⚡

An end-to-end Machine Learning and Computer Vision pipeline for real-time hand gesture recognition, featuring **Translation and Scale Invariant Feature Engineering**, cross-session generalization benchmarking, and a **live Streamlit deployment dashboard**.

---

## 🌟 Key Architecture & Highlights

1. **Data Collection Pipeline (`collector.py`)**:
   - Built with **OpenCV** and **MediaPipe** (supporting both modern Tasks API and legacy solutions).
   - High-throughput **Continuous Auto-Record Mode (`r`)** and single capture (`s`).
   - On-screen target counter (e.g. `200/200 [OK]`).
   - Captured **Session A (Gold Standard - `train_data.csv`)** and **Session B (Stress Test - `test_data.csv`)**.

2. **Invariant Feature Engineering (`features.py`)**:
   - **Translation Invariance**: Centers coordinates by subtracting the wrist origin (landmark 0).
   - **Scale Invariance**: Normalizes coordinates by dividing by the Euclidean distance between the wrist (0) and middle-finger MCP (9).
   - Vectorized batch processing for training and low-latency real-time inference.

3. **Model Benchmark (`train_model.py`)**:
   - Compares **Raw Coordinates** vs. **Invariant Features** using Random Forest classifiers.
   - Evaluates **Same-Session** vs. **Cross-Session (Stress Test)** performance.
   - Exports the production model to `gesture_model.pkl`.

4. **Live Streamlit Deployment (`app.py`)**:
   - Two-column dashboard with dark theme.
   - Real-time webcam processing with hand skeleton and bounding box.
   - Neon Cyan prediction tag with confidence score.
   - Live metrics: **Inference Latency (ms)** and **Pipeline FPS**.
   - **Live Model Switcher**: Toggle between Invariant Model and Raw Model to demonstrate invariance live.

---

## 📊 Cross-Session Generalization Benchmark (2x2 Table)

| Feature Set | [Same Session] Accuracy | [Cross Session] Accuracy (Stress Test) |
|---|---|---|
| **[Raw Features]** | 96.88% | 35.50% *(Severe collapse)* |
| **[Invariant Features]** | 93.12% | **47.00% (+11.50% gain)** |

> **Insight:** Invariant features prevent the model from overfitting to absolute camera frame coordinates, improving resilience when lighting, distance, or camera angle changes.

---

## 🚀 Quickstart

### 1. Installation

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Collector

```powershell
# Session A: Training Set (200 samples/class)
python collector.py --csv train_data.csv --target 200

# Session B: Test Set (100 samples/class under changed environment)
python collector.py --csv test_data.csv --target 100
```
- Keys `1`, `2`, `3`, `4`: Switch active class (0, 1, 2, 3)
- Key `r`: Toggle continuous auto-recording
- Key `s`: Save single sample
- Key `q` or `ESC`: Quit

### 3. Train & Benchmark Models

```powershell
python train_model.py
```
Outputs the 2x2 accuracy comparison table and saves `gesture_model.pkl`.

### 4. Launch Live Streamlit App

```powershell
streamlit run app.py
```
Opens interactive dashboard at `http://localhost:8501`.

---

## 📁 Repository Structure

```
.
├── collector.py           # Real-time webcam data collection tool
├── features.py            # GestureFeatureExtractor (Raw & Invariant features)
├── train_model.py         # Model training, evaluation, & 2x2 benchmark
├── app.py                 # Live Streamlit deployment application
├── train_data.csv         # Session A dataset (800 samples)
├── test_data.csv          # Session B cross-session dataset (400 samples)
├── gesture_model.pkl      # Production Invariant Random Forest model
├── raw_gesture_model.pkl  # Baseline Raw Random Forest model
├── hand_landmarker.task   # MediaPipe Hand Landmarker model bundle
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
```
