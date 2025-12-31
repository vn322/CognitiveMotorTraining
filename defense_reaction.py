# blocks/defense_reaction.py
import cv2
import numpy as np
import time
import random
import math

class DefenseReactionBlock:
    def __init__(self, config):
        self.config = config
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.signal_phase = None
        self.signal_start = None
        self.stimulus_show_time = None
        self.movement_start_time = None
        self.trial_start_time = None
        self.target_joint = None
        self.original_pos = None
        self.timeout_limit = 1.0
        self.joint_hist = {}
        self.trials_total = config.get("defense_reaction", {}).get("trials", 12)

    def draw_skeleton(self, frame, landmarks):
        left_color = (76, 175, 80)
        right_color = (255, 193, 7)
        pink = (255, 105, 180)
        white = (255, 255, 255)
        def g(n): return landmarks.get(n)
        ls, rs = g('LEFT_SHOULDER'), g('RIGHT_SHOULDER')
        if ls and rs: cv2.line(frame, ls, rs, (200,200,200), 2)
        for a,b,c in [('LEFT_SHOULDER','LEFT_WRIST',left_color),('RIGHT_SHOULDER','RIGHT_WRIST',right_color)]:
            p1,p2 = g(a), g(b)
            if p1 and p2: cv2.line(frame, p1, p2, c, 2)
        for name, color in [
            ('HEAD', pink), ('LEFT_SHOULDER', left_color), ('RIGHT_SHOULDER', right_color),
            ('LEFT_WRIST', left_color), ('RIGHT_WRIST', right_color)
        ]:
            p = g(name)
            if p:
                cv2.circle(frame, p, 22, white, -1)
                cv2.circle(frame, p, 20, color, -1)
                if name == self.target_joint and self.phase == "stimulus":
                    if int(time.time() * 2) % 2 == 0:
                        cv2.line(frame, (p[0]-30, p[1]), (p[0]+30, p[1]), (0,0,255), 5)
                        cv2.line(frame, (p[0], p[1]-30), (p[0], p[1]+30), (0,0,255), 5)

    def detect_movement_start(self, joint_name, current_pos, current_time):
        if not joint_name or not current_pos:
            return False
        hist = self.joint_hist.setdefault(joint_name, [])
        hist.append((current_pos[0], current_pos[1], current_time))
        if len(hist) > 5: hist.pop(0)
        if len(hist) < 3: return False
        t0, t1, t2 = hist[-3][2], hist[-2][2], hist[-1][2]
        dt1, dt2 = t1 - t0, t2 - t1
        if dt1 < 0.001 or dt2 < 0.001: return False
        v1x = (hist[-2][0] - hist[-3][0]) / dt1
        v1y = (hist[-2][1] - hist[-3][1]) / dt1
        v2x = (hist[-1][0] - hist[-2][0]) / dt2
        v2y = (hist[-1][1] - hist[-2][1]) / dt2
        acc = math.sqrt((v2x - v1x)**2 + (v2y - v1y)**2)
        return acc > 300

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
            cv2.putText(frame, "Defend!", (w//2 - 70, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 0), 2)
            if current_time - self.trial_start_time > 1.2:
                self.phase = "stimulus"
                self.stimulus_show_time = current_time
                self.target_joint = random.choice(['HEAD', 'LEFT_SHOULDER', 'RIGHT_SHOULDER'])
                pos = landmarks.get(self.target_joint)
                self.original_pos = pos[:] if pos else None

        elif self.phase == "stimulus":
            if current_time - self.stimulus_show_time > self.timeout_limit:
                displacement = 0.0
                if self.original_pos and landmarks.get(self.target_joint):
                    curr = landmarks[self.target_joint]
                    displacement = math.hypot(curr[0] - self.original_pos[0], curr[1] - self.original_pos[1])
                trial = {
                    "trial": self.current_trial + 1,
                    "target_joint": self.target_joint,
                    "latency_ms": 1000.0,
                    "movement_ms": 0.0,
                    "total_rt_ms": 1000.0,
                    "displacement_mm": round(displacement, 1),
                    "success": False,
                    "timeout": True
                }
                self.trial_data.append(trial)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            curr_pos = landmarks.get(self.target_joint)
            if self.original_pos and curr_pos:
                displacement = math.hypot(curr_pos[0] - self.original_pos[0], curr_pos[1] - self.original_pos[1])
                if displacement > 50:
                    cv2.circle(frame, self.original_pos, 80, (0, 200, 0), 2)
                if displacement > 50:
                    touch_time = current_time
                    latency = max(0, (self.movement_start_time or self.stimulus_show_time) - self.stimulus_show_time)
                    movement = max(0, touch_time - (self.movement_start_time or self.stimulus_show_time))
                    total_rt = latency + movement
                    anatomical = self.anatomical_joint(self.target_joint.replace("_", ""))
                    trial = {
                        "trial": self.current_trial + 1,
                        "target_joint": anatomical,
                        "latency_ms": round(latency * 1000, 1),
                        "movement_ms": round(movement * 1000, 1),
                        "total_rt_ms": round(total_rt * 1000, 1),
                        "displacement_mm": round(displacement, 1),
                        "success": True,
                        "timeout": False
                    }
                    self.trial_data.append(trial)
                    self.phase = "feedback"
                    self.feedback_start = current_time
                    return False

            if not self.movement_start_time and curr_pos:
                if self.detect_movement_start(self.target_joint, curr_pos, current_time):
                    if current_time >= self.stimulus_show_time:
                        self.movement_start_time = current_time

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("success", False)
            cv2.putText(frame, "OK" if ok else "Timeout", (w//2 - 60, h//2 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 0) if ok else (0, 0, 200), 2)
            if current_time - self.feedback_start > 0.8:
                if self.current_trial < self.trials_total:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.trial_start_time = current_time
                    self.movement_start_time = None
                else:
                    self.signal_phase = "end"
                    self.signal_start = current_time

        return self.phase == "complete" and self.signal_phase == "end"

    def get_results(self):
        return self.trial_data