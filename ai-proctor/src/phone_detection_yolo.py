from ultralytics import YOLO
from src.gpu_check import check_gpu

PHONE_CLASS_ID = 67  # COCO "cell phone"


class PhoneDetector:
    def __init__(self, model_path="models/yolov8n.pt", conf=0.3, every_n=1):
        dev = check_gpu()
        self.device = 0 if dev.type == "cuda" else "cpu"
        self.model = YOLO(model_path)
        self.conf = conf
        self.every_n = every_n      # run YOLO every n-th frame to keep FPS up
        self._count = 0
        self._last = []

    def detect(self, frame):
        """Returns list of (x1, y1, x2, y2, conf) for phones in the frame."""
        self._count += 1
        if self._count % self.every_n:
            return self._last
        results = self.model(frame, device=self.device, conf=self.conf,
                             classes=[PHONE_CLASS_ID], verbose=False)
        self._last = [(*map(int, b.xyxy[0]), float(b.conf[0]))
                      for r in results for b in r.boxes]
        return self._last