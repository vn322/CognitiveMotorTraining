# blocks/moving_object.py
import cv2
import numpy as np
import time
import random
import math

class MovingObjectBlock:
    def __init__(self, config):
        self.config = config
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.signal_phase = None
        self.signal_start = None
        self.stimulus_start = None
        self.zone_center = None
        self.stimulus_pos = None
        self.stimulus_velocity = None
        self.direction = 1
        self.trials_total = config.get("moving_object", {}).get("trials", 12)
        self.joint_hist = {}

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

    def anatomical_joint(self, joint):
        flip = self.config.get("camera", {}).get("flip_horizontal", False)
        if not flip:
            return joint
        if joint == "LEFT_WRIST":
            return "RIGHT_WRIST"
        elif joint == "RIGHT_WRIST":
            return "LEFT_WRIST"
        return joint

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

        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "start_signal":
            cv2.putText(frame, "Ready", (w//2 - 70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 2)
            if current_time - self.trial_start_time > 1.2:
                self.phase = "stimulus"
                self.stimulus_start = current_time
                self.zone_center = (w // 2, h // 2)
                self.stimulus_velocity = random.uniform(800, 1500)
                self.direction = random.choice([1, -1])
                if self.direction == 1:
                    self.stimulus_pos = [-40, h // 2]
                else:
                    self.stimulus_pos = [w + 40, h // 2]

        elif self.phase == "stimulus":
            elapsed = current_time - self.stimulus_start
            if elapsed > 2.0:
                self.record_trial(2000.0, "timeout", "none", 0.0, False)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            self.stimulus_pos[0] += self.direction * self.stimulus_velocity * (1/30)
            x, y = int(self.stimulus_pos[0]), int(self.stimulus_pos[1])
            zx, zy = self.zone_center

            # ✅ СТИМУЛ 80 PX (в 2 раза больше)
            cv2.circle(frame, (zx, zy), 80, (255, 255, 255), 2)
            cv2.circle(frame, (zx, zy), 80, (100, 100, 100), 1)
            cv2.circle(frame, (x, y), 80, (0, 0, 200), -1)  # ← 80 px

            if self.direction == 1:
                t_align = (zx - (-40)) / self.stimulus_velocity
            else:
                t_align = ((w + 40) - zx) / self.stimulus_velocity

            wrists = {k: v for k, v in landmarks.items() if k.endswith("WRIST")}
            for name, wrist in wrists.items():
                if wrist and (wrist[0] - zx)**2 + (wrist[1] - zy)**2 <= 120**2:  # ← 120 px радиус
                    touch_time = current_time
                    timing_error = (touch_time - self.stimulus_start) - t_align
                    dist = math.hypot(wrist[0] - zx, wrist[1] - zy)
                    success = abs(timing_error * 1000) < 50
                    joint = self.anatomical_joint(name)
                    self.record_trial(
                        (touch_time - self.stimulus_start) * 1000,
                        dist,
                        joint,
                        timing_error * 1000,
                        success
                    )
                    self.phase = "feedback"
                    self.feedback_start = current_time
                    return False

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("success", False)
            cv2.putText(frame, "OK" if ok else "ERR", (w//2 - 30, h//2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0) if ok else (0, 0, 200), 2)
            if current_time - self.feedback_start > 0.6:
                if self.current_trial < self.trials_total:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.trial_start_time = current_time
                else:
                    self.signal_phase = "end"
                    self.signal_start = current_time

        return self.phase == "complete" and self.signal_phase == "end"

    def record_trial(self, rt_ms, accuracy, joint, timing_error_ms, success):
        trial = {
            "trial": self.current_trial + 1,
            "total_rt_ms": round(rt_ms, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int, float)) else accuracy,
            "joint_used": joint,
            "timing_error_ms": round(timing_error_ms, 1),
            "success": success
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data