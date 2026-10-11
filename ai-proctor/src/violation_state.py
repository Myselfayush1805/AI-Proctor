import time

class ViolationState:
    def __init__(self):
        self.multiple_faces = 0
        self.absence = 0
        self.phone = 0
        self.looking_away = 0
        self.total_score = 0
        self._last = {}

    def add(self, violation, weight, cooldown=3.0):
        """Counts a violation at most once per `cooldown` seconds. Returns True if counted."""
        now = time.monotonic()
        if now - self._last.get(violation, -1e9) < cooldown:
            return False
        self._last[violation] = now
        setattr(self, violation, getattr(self, violation) + 1)
        self.total_score += weight
        return True

    def risk_level(self):
        if self.total_score < 10:
            return "LOW"
        if self.total_score < 30:
            return "MEDIUM"
        return "HIGH"