# AI Proctor

Real-time webcam exam proctoring built with OpenCV, MediaPipe and YOLOv8. It watches the
candidate through the webcam, flags suspicious behaviour, keeps a running risk score and saves
every violation to a CSV log.

<!-- Add a demo GIF or screenshot here, e.g.: ![Demo](docs/demo.gif) -->

## Features
- **Face detection** with live bounding boxes (OpenCV Haar cascade)
- **Multiple-face detection**: flags when more than one person is in frame
- **Candidate absence**: flags when no face is visible for more than 5 seconds
- **Gaze tracking**: MediaPipe FaceMesh eye landmarks and iris position; flags looking away for more than 1.5 seconds
- **Phone detection**: YOLOv8n ("cell phone" class), runs on GPU when available and falls back to CPU
- **Risk scoring** with a per-violation cooldown, so one event is not counted every frame
- **CSV logging** of every violation plus a session summary on exit

## Scoring
| Violation | Points | Trigger |
|---|---|---|
| Phone detected | 10 | YOLO "cell phone", confidence > 0.3 |
| Multiple faces | 5 | more than one face in frame |
| Looking away | 3 | gaze off-centre for more than 1.5 s |
| Absence | 2 | no face for more than 5 s |

Each violation type counts at most once every 3 seconds.
Risk level: **LOW** below 10, **MEDIUM** below 30, **HIGH** otherwise.

## Requirements
- Windows with a webcam (the camera code is written for Windows; Linux/macOS may need small changes)
- **Python 3.12** (3.9-3.12 work; 3.13+ is not supported by the pinned `mediapipe`)
- Optional: NVIDIA GPU with CUDA for faster phone detection

## Setup
```powershell
git clone https://github.com/Myselfayush1805/ai-proctor.git
cd ai-proctor

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version          # must say 3.12.x

# PyTorch first. Pick the command for your CUDA version at https://pytorch.org
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```
No NVIDIA GPU? Install the CPU build instead: `pip install torch torchvision`.

The YOLO weights are expected at `models/yolov8n.pt`. If the file is missing, download
`yolov8n.pt` from the [Ultralytics releases](https://github.com/ultralytics/assets/releases) and place it in `models/`.

## Run
```powershell
python app.py
```
Press **q** to quit. On exit the session summary is printed and the violations are saved to `logs/violations.csv`.

## Project structure
```
ai-proctor/
├── app.py                      main loop: detection, scoring, overlay
├── requirements.txt
├── models/
│   └── yolov8n.pt              YOLOv8n weights
└── src/
    ├── camera.py               webcam wrapper
    ├── face_detection.py       face detection + absence check
    ├── eye_tracking.py         MediaPipe eye landmarks + gaze
    ├── phone_detection_yolo.py YOLO phone detector
    ├── violation_state.py      score, counters, cooldown, risk level
    ├── violation_logger.py     in-memory log + CSV export
    └── gpu_check.py            CUDA check with CPU fallback
```

## Tuning
| What | Where |
|---|---|
| Gaze sensitivity / delay | `EyeTracker(low, high, away_seconds)` in `src/eye_tracking.py` |
| Phone confidence / frame skipping | `PhoneDetector(conf, every_n)` in `app.py` |
| Absence delay | `absence_start` check (5 s) in `src/face_detection.py` |
| Cooldown between counts | `cooldown` in `ViolationState.add` |
| Points per violation | the `flag(...)` calls in `app.py` |

## Troubleshooting
- **`No matching distribution found for mediapipe==0.10.21`**: you are on Python 3.13 or newer. Create the venv with Python 3.12.
- **`Webcam not accessible`**: close other apps using the camera, or try `Camera(cam_id=1)`.
- **Slow / low FPS**: install the CUDA build of PyTorch, or set `every_n=2` in `PhoneDetector`.
- **Gaze alerts too sensitive or too lax**: adjust `low` / `high` in `EyeTracker`.

## Limitations
Face detection uses Haar cascades, so it can miss faces at steep angles or in poor lighting. Gaze tracking is a heuristic
based on iris position and varies with camera angle. This project is for learning and demonstration, not for
high-stakes exam decisions.

## Privacy
Proctoring processes video of a person. If you use anything like this on real candidates, get their informed consent and
follow the privacy laws that apply to you. This project stores only violation timestamps and types, not video.

## License
MIT. See [LICENSE](LICENSE).