# Hand Gesture Data Collector

A Python data collection tool built with **OpenCV** and **MediaPipe** to capture 21 hand landmark coordinates (x, y, z) for machine learning models.

## Setup & Requirements

Activate the virtual environment or install dependencies from `requirements.txt`:

```bash
# If using existing virtual environment:
.\.venv\Scripts\activate

# Or install dependencies:
pip install -r requirements.txt
```

## Running the Collector

```bash
python collector.py
```

### Optional Arguments
- `--camera 0`: Specify webcam index (default: `0`).
- `--csv gesture_data.csv`: Specify output CSV file path (default: `gesture_data.csv`).
- `--model hand_landmarker.task`: Specify custom path to MediaPipe model bundle.

## Controls & Hotkeys

| Key | Action |
|---|---|
| `1` | Switch to **Class 0** |
| `2` | Switch to **Class 1** |
| `3` | Switch to **Class 2** |
| `4` | Switch to **Class 3** |
| `s` | **Save** current 21 raw landmarks (x, y, z) + class label to CSV |
| `q` or `ESC` | **Quit** collector |

## Output Format (`gesture_data.csv`)

The CSV file contains 64 columns:
- `class`: Class label (`0`, `1`, `2`, or `3`)
- `x0, y0, z0, ..., x20, y20, z20`: 21 hand landmarks in raw absolute coordinates (MediaPipe format).
