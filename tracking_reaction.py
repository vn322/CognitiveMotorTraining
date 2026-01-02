# blocks/tracking_reaction.py 
import cv2
import numpy as np
import time
import math
import random

class TrackingReactionBlock:
    def __init__(self, config):
        self.config = config
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.signal_phase = None
        self.signal_start = None
        self.trial_start_time = None
        self.tracking_start = None
        self.trajectory = []
        self.stimulus_pos = None
        self.tracking_points = []
        self.trials_total = config.get("tracking_reaction", {}).get("trials", 3)
        self.duration = 5.0

    def draw_skeleton(self, frame, landmarks):
        left_color = (76, 175, 80)
        right_color = (255, 193, 7)
        white = (255, 255, 255)
        def g(n): return landmarks.get(n)
        for name, color in [('LEFT_WRIST', left_color), ('RIGHT_WRIST', right_color)]:
            p = g(name)
            if p:
                cv2.circle(frame, p, 16, white, -1)
                cv2.circle(frame, p, 14, color, -1)

    def generate_spiral(self, center, duration, fps=30):
        cx, cy = center
        points = []
        steps = int(duration * fps)
        for i in range(steps):
            t = i / steps
            angle = t * 4 * math.pi
            radius = 50 + t * 100
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append((x, y))
        return points

    def generate_zigzag(self, center, duration, fps=30):
        cx, cy = center
        points = []
        steps = int(duration * fps)
        for i in range(steps):
            t = i / steps
            x = cx - 150 + (t * 300)
            y = cy + 50 * math.sin(t * 4 * math.pi)
            points.append((x, y))
        return points

    def generate_sine(self, center, duration, fps=30):
        cx, cy = center
        points = []
        steps = int(duration * fps)
        for i in range(steps):
            t = i / steps
            y = cy - 100 + (t * 200)
            x = cx + 80 * math.sin(t * 2 * math.pi)
            points.append((x, y))
        return points

    def start_trial(self):
        self.signal_phase = "start"
        self.signal_start = time.time()

    def process_frame(self, frame, landmarks):
        current_time = time.time()
        h, w = frame.shape[:2]

        if self.signal_phase == "start":
            cv2.putText(frame, "Start", (w//2 - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
            if current_time - self.signal_start > 2.0:
                self.signal_phase = None
                self.phase = "start_signal"
                self.trial_start_time = current_time
            return False

        if self.signal_phase == "end":
            cv2.putText(frame, "Stop", (w//2 - 70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
            if current_time - self.signal_start > 2.0:
                self.phase = "complete"
            return self.phase == "complete"

        self.draw_skeleton(frame, landmarks)

        cv2.putText(frame, f"Trial: {self.current_trial + 1} / {self.trials_total}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, f"Trial: {self.current_trial + 1} / {self.trials_total}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "start_signal":
            cv2.putText(frame, "Track!", (w//2 - 70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 2)
            if current_time - self.trial_start_time > 1.2:
                self.phase = "tracking"
                self.tracking_start = current_time
                center = (w//2, h//2)
                traj_type = random.choice(["spiral", "zigzag", "sine"])
                if traj_type == "spiral":
                    self.trajectory = self.generate_spiral(center, self.duration)
                elif traj_type == "zigzag":
                    self.trajectory = self.generate_zigzag(center, self.duration)
                else:
                    self.trajectory = self.generate_sine(center, self.duration)
                self.stimulus_idx = 0
                self.tracking_points = []

        elif self.phase == "tracking":
            elapsed = current_time - self.tracking_start
            if elapsed > self.duration:
                self.compute_metrics()
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            idx = min(int(elapsed * 30), len(self.trajectory) - 1)
            self.stimulus_pos = self.trajectory[idx]
            x, y = self.stimulus_pos
            cv2.circle(frame, (int(x), int(y)), 20, (0, 0, 200), -1)
            cv2.circle(frame, (int(x), int(y)), 20, (255, 255, 255), 2)

            wrist = landmarks.get('LEFT_WRIST') or landmarks.get('RIGHT_WRIST')
            if wrist:
                self.tracking_points.append((
                    elapsed, wrist[0], wrist[1], x, y
                ))

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            cv2.putText(frame, f"Error: {last.get('tracking_error_mm', 0):.1f} mm", 
                       (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 0), 2)
            if current_time - self.feedback_start > 2.0:
                if self.current_trial < self.trials_total - 1:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.trial_start_time = current_time
                else:
                    self.signal_phase = "end"
                    self.signal_start = current_time

        return self.phase == "complete" and self.signal_phase == "end"

    def compute_metrics(self):
        if not self.tracking_points:
            error = 1000.0
            coverage = 0.0
        else:
            errors = [math.hypot(wx - sx, wy - sy) for _, wx, wy, sx, sy in self.tracking_points]
            error = np.mean(errors)
            coverage = (np.array(errors) <= 40).mean() * 100

        trial = {
            "trial": self.current_trial + 1,
            "tracking_error_mm": round(error, 1),
            "coverage_%": round(coverage, 1)
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data