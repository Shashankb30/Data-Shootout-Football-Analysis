# ⚽ Football Analysis System — AI + Computer Vision

>Project is inspired by a previous competition from Kaggle hosted by DFL Bundesliga Datashoot ,were the football images and videos were uploaded by official Bundesliga League . Mianly this project focusses on detecting players, track movement, measure speed & distance, and analyse ball possession — all from a raw match video. 

![Football Analysis Demo](output_videos/screenshot.png)

---

## 🚀 What This Does

This project builds a full football analysis pipeline using machine learning and computer vision. Feed it a match video and it outputs an annotated video with:

- **Player & ball detection** using YOLOv8 (state-of-the-art object detection)
- **Multi-object tracking** with persistent player IDs across frames
- **Team assignment** by clustering t-shirt colors with KMeans
- **Ball possession tracking** — which team controls the ball, frame by frame
- **Camera movement compensation** via Lucas-Kanade optical flow
- **Real-world position mapping** using perspective transformation (pixels → metres)
- **Speed & distance stats** per player (km/h and metres covered)

---

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Object detection | YOLOv8 (Ultralytics) |
| Object tracking | ByteTrack (Supervision) |
| Team color clustering | KMeans (scikit-learn) |
| Camera motion | Lucas-Kanade Optical Flow (OpenCV) |
| Perspective transform | Homography matrix (OpenCV) |
| Ball interpolation | Pandas interpolation |
| Video I/O | OpenCV |

---

## 📁 Project Structure

```
football_analysis/
├── main.py                          # Entry point — runs the full pipeline
├── trackers/                        # YOLOv8 detection + ByteTrack
├── team_assigner/                   # KMeans t-shirt color clustering
├── player_ball_assigner/            # Ball possession per frame
├── camera_movement_estimator/       # Optical flow camera compensation
├── view_transformer/                # Perspective transform (px → metres)
├── speed_and_distance_estimator/    # Speed (km/h) + distance (m) per player
├── utils/                           # Bounding box helpers, video I/O
├── models/                          # Place your trained YOLO weights here
├── input_videos/                    # Place your input video here
├── output_videos/                   # Annotated output video saved here
├── stubs/                           # Pre-computed tracking cache (sample video)
└── training/                        # YOLOv5 fine-tuning notebook
```

---

## ⚙️ Setup

### Prerequisites

- Python **3.11** (required — Python 3.12+ has dependency issues with this stack)
- Git

### 1. Clone the repo

```bash
git clone https://github.com/Shashankb30/Data-Shootout-Football-Analysis.git
cd football_analysis
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required files

| File | Link | Destination |
|---|---|---|
| Sample input video | [Google Drive](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view) | `input_videos/bun.mp4` |
| Trained YOLO weights | [Google Drive](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view) | `models/best.pt` |

### 5. Run

```bash
python main.py
```

Output video will be saved to `output_videos/output_video.avi`.

---

## 🔍 How It Works

### Pipeline Overview

```
Input Video
    │
    ▼
YOLOv8 Detection  ──►  ByteTrack (persistent IDs)
    │
    ├──►  KMeans Team Assigner  (t-shirt color → team 1 / team 2)
    │
    ├──►  Ball Interpolation    (fills missing ball detections)
    │
    ▼
Camera Movement Estimator  (optical flow → adjusts all positions)
    │
    ▼
Perspective Transformer    (pixel coords → real-world metres)
    │
    ▼
Speed & Distance Estimator (km/h + metres per player)
    │
    ▼
Annotated Output Video
```

### Key modules

**Team Assigner** — crops the top half of each detected player bounding box (the t-shirt region), runs KMeans with 2 clusters to find dominant colors, then classifies each player by their color distance to each team centroid. Corner pixels are used to separate the player from the background.

**Camera Movement Estimator** — tracks static features in the left and right edge columns of the frame using Lucas-Kanade optical flow. Any large movement in those regions is attributed to camera pan/zoom and subtracted from all player positions.

**View Transformer** — uses four manually defined pitch corner points to compute a homography matrix. All adjusted player positions are then transformed from pixel space into real-world metres on the pitch.

**Speed & Distance Estimator** — for each player, compares their transformed position across a sliding 5-frame window (at 24 fps) to compute instantaneous speed in km/h and cumulative distance in metres.

---

## 📊 Output Annotations

Each frame in the output video shows:

- Colored ellipses under players (color = team)
- Player tracking IDs
- Red triangle = player with ball possession
- Green triangle = ball position
- Speed (km/h) and distance (m) below each player
- Camera movement X/Y overlay (top left)
- Team ball control percentage (bottom right)

---

## 🏋️ Training Your Own Model

A training notebook is included at `training/football_training_yolo_v5.ipynb`.

Dataset: [Roboflow Football Players Detection](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

The fine-tuned model detects four classes: `player`, `goalkeeper`, `referee`, `ball`.

---

## 📦 Requirements

```
ultralytics>=8.0.0
supervision>=0.18.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
```

---

## 🙏 Credits

- Tutorial : [DFL Football Analysis](https://youtu.be/neBZ6huolkg?si=rypJ8ok9bU5V-cuS)
- Dataset: [DFL Bundesliga Data Shootout (Kaggle)](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
- Detection dataset: [Roboflow Universe](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)

---

## 📄 License

This project is for educational purposes. Dataset and video used from kaggle dataset from bundesliga
