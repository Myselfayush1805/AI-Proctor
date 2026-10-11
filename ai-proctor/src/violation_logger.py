import csv
import os
import time

class ViolationLogger:
    def __init__(self):
        self.logs = []

    def log(self, violation_type, score):
        entry = {"type": violation_type, "score": score,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.logs.append(entry)
        print("LOG:", entry)

    def save(self, path="logs/violations.csv"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["timestamp", "type", "score"])
            w.writeheader()
            w.writerows(self.logs)
        return path

    def print_summary(self, state):
        print("\n--- Session summary ---")
        print(f"Total score: {state.total_score}  Risk: {state.risk_level()}")
        print(f"Multiple faces: {state.multiple_faces}  Absence: {state.absence}  "
              f"Phone: {state.phone}  Looking away: {state.looking_away}")
        print(f"Saved to {self.save()}")