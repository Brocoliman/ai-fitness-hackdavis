# AI Fitness Trainer — Squat Analysis

Squat rep counter and form checker using MediaPipe Pose + OpenCV.

## Setup

```bash
pip install -r requirements.txt
```

## Run


**Webcam (standalone, no Streamlit)**:
```bash
python webcam_demo.py              # beginner mode
```

Press `q` to quit, `r` to reset counters.

## Project Structure

```
├── 🏠️_Demo.py          # Streamlit home page
├── pages/
│   └── 1_📹_Video.py    # Video upload analysis page
├── webcam_demo.py        # Standalone webcam script
├── analyzer.py           # Core SquatAnalyzer class (state machine + angles)
├── utils.py              # Angle math, landmark helpers, drawing utils
├── thresholds.py         # Angle thresholds for beginner/pro modes
└── requirements.txt
```

## How It Works

1. MediaPipe Pose extracts 33 body landmarks per frame
2. Knee angle (hip→knee→ankle) measures squat depth
3. Hip angle (shoulder→hip→knee) measures torso lean
4. State machine: standing → squatting (knee angle drops) → standing (knee angle rises) = 1 rep
5. Each rep scored correct/incorrect based on threshold ranges
```
