import cv2
from src.camera import Camera
from src.face_detection import FaceDetector
from src.violation_state import ViolationState
from src.violation_logger import ViolationLogger
from src.eye_tracking import EyeTracker
from src.phone_detection_yolo import PhoneDetector

cam = Camera()
face_detector = FaceDetector()
violations = ViolationState()
logger = ViolationLogger()
eye_tracker = EyeTracker()
phone_detector = PhoneDetector(conf=0.3, every_n=1)

FONT = cv2.FONT_HERSHEY_SIMPLEX
RED = (0, 0, 255)


def flag(name, label, weight, y, frame):
    """Score + log once per cooldown; always show the warning while the condition holds."""
    if violations.add(name, weight):
        logger.log(label, weight)
    cv2.putText(frame, label.replace("_", " "), (20, y), FONT, 0.9, RED, 2)


try:
    while True:
        frame = cam.read()
        if frame is None:
            break
        frame = cv2.resize(frame, (640, 480))

        # run YOLO on the clean frame, before anything is drawn on it
        phones = phone_detector.detect(frame)

        faces = face_detector.detect(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eyes = eye_tracker.get_landmarks(frame)
        if eyes:
            left_eye, right_eye = eyes
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            cv2.putText(frame, f"Gaze: {eye_tracker.gaze}", (470, 30), FONT, 0.6, (255, 255, 255), 1)

        for (x1, y1, x2, y2, conf) in phones:
            cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2)
            cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10), FONT, 0.6, RED, 2)

        if len(faces) > 1:
            flag("multiple_faces", "MULTIPLE_FACES", 5, 40, frame)
        if face_detector.check_absence(faces):
            flag("absence", "CANDIDATE_ABSENT", 2, 80, frame)
        if phones:
            flag("phone", "PHONE_DETECTED", 10, 160, frame)
        if eye_tracker.is_looking_away():
            flag("looking_away", "LOOKING_AWAY", 3, 200, frame)

        cv2.putText(frame, f"Risk Score:{violations.total_score} ({violations.risk_level()})",
                    (20, 120), FONT, 0.8, (255, 255, 0), 2)
        if logger.logs:
            last = logger.logs[-1]
            cv2.putText(frame, f"Last: {last['type']} @ {last['timestamp']}",
                        (10, 460), FONT, 0.5, RED, 1)

        cv2.imshow("AI Proctoring", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cam.release()
    logger.print_summary(violations)