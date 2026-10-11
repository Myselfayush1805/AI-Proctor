import time
import cv2
import mediapipe as mp


class EyeTracker:
    def __init__(self, low=0.35, high=0.65, away_seconds=1.5):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS, self.RIGHT_IRIS = 468, 473
        self.low, self.high, self.away_seconds = low, high, away_seconds
        self.gaze = None
        self._away_since = None

    def get_landmarks(self, frame):
        """Returns (left_eye, right_eye) pixel-point lists, or None. Also updates self.gaze."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            self.gaze, self._away_since = None, None
            return None
        h, w, _ = frame.shape
        lm = results.multi_face_landmarks[0].landmark
        left_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.LEFT_EYE]
        right_eye = [(int(lm[i].x * w), int(lm[i].y * h)) for i in self.RIGHT_EYE]

        # iris position inside each eye: ~0.5 = looking straight
        l = (lm[self.LEFT_IRIS].x - lm[33].x) / (lm[133].x - lm[33].x + 1e-6)
        r = (lm[self.RIGHT_IRIS].x - lm[362].x) / (lm[263].x - lm[362].x + 1e-6)
        ratio = (l + r) / 2
        self.gaze = "LEFT" if ratio < self.low else "RIGHT" if ratio > self.high else "CENTER"
        if self.gaze == "CENTER":
            self._away_since = None
        elif self._away_since is None:
            self._away_since = time.monotonic()
        return left_eye, right_eye

    get_eye_landmarks = get_landmarks  # keeps your old method name working

    def is_looking_away(self):
        return (self._away_since is not None
                and time.monotonic() - self._away_since > self.away_seconds)