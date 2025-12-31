# blocks/simple_reaction.py
import cv2
import numpy as np
import time
import math
import random
 
class SimpleReactionBlock:
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
        self.joint_hist = {}
        self.timeout_limit = 2.0
        self.trials_total = config.get("simple_reaction", {}).get("trials", 12)

    def get_shoulder_center(self, landmarks):
        ls = landmarks.get('LEFT_SHOULDER')
        rs = landmarks.get('RIGHT_SHOULDER')
        if ls and rs:
            return ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
        return (640, 360)

    def draw_skeleton(self, frame, landmarks):
        left_color = (76, 175, 80)
        right_color = (255, 193, 7)
        white = (255, 255, 255)
        def g(n): return landmarks.get(n)
        ls, rs = g('LEFT_SHOULDER'), g('RIGHT_SHOULDER')
        if ls and rs:
            cv2.line(frame, ls, rs, (200,200,200), 2)
        for a,b,c in [('LEFT_SHOULDER','LEFT_WRIST',left_color),('RIGHT_SHOULDER','RIGHT_WRIST',right_color)]:
            p1,p2 = g(a), g(b)
            if p1 and p2:
                cv2.line(frame, p1, p2, c, 2)
        for name, color in [('LEFT_WRIST', left_color), ('RIGHT_WRIST', right_color)]:
            p = g(name)
            if p:
                cv2.circle(frame, p, 16, white, -1)
                cv2.circle(frame, p, 14, color, -1)

    def draw_axes(self, frame, center):
        cx, cy = center
        h, w = frame.shape[:2]
        gray = (158, 158, 158)
        cv2.line(frame, (cx, 0), (cx, h), gray, 1)
        cv2.line(frame, (0, cy), (w, cy), gray, 1)

    def stimulus_position(self, center, radius, angle_deg):
        rad = math.radians(angle_deg)
        x = int(center[0] + radius * math.cos(rad))
        y = int(center[1] - radius * math.sin(rad))
        return (x, y)

    def classify_position(self, stim_pos, center):
        cx, cy = center
        dx = stim_pos[0] - cx
        dy = stim_pos[1] - cy
        return {
            "left_of_midline": dx < 0,
            "above_shoulder_line": dy < 0,
            "quadrant": "upper-left" if dx < 0 and dy < 0 else
                       "upper-right" if dx >= 0 and dy < 0 else
                       "lower-left" if dx < 0 and dy >= 0 else "lower-right"
        }

    def detect_movement_start(self, joint_name, pos, t):
        if not pos:
            return False
        hist = self.joint_hist.setdefault(joint_name, [])
        hist.append((pos[0], pos[1], t))
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
        return acc > 400 and (abs(v2x) + abs(v2y)) > 150

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
            cv2.putText(frame, "Start", (w//2-80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3)
            if current_time - self.signal_start > 2.0:
                self.signal_phase = None
                self.phase = "fixation"
                self.trial_start_time = current_time
            return False

        if self.signal_phase == "end":
            cv2.putText(frame, "Stop", (w//2-70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3)
            if current_time - self.signal_start > 2.0:
                self.phase = "complete"
            return self.phase == "complete"

        center = self.get_shoulder_center(landmarks)
        self.draw_skeleton(frame, landmarks)
        self.draw_axes(frame, center)

        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "fixation":
            cv2.putText(frame, "+", (center[0]-15, center[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
            if current_time - self.trial_start_time > 0.5:
                self.phase = "stimulus"
                self.stimulus_show_time = current_time
                angle = random.choice(self.config.get("simple_reaction", {}).get("angles_deg", [0,90,180,270]))
                radius = min(w, h) * 0.4 - 80
                self.stimulus_pos = self.stimulus_position(center, radius, angle)
                self.pos_info = self.classify_position(self.stimulus_pos, center)

        elif self.phase == "stimulus":
            if current_time - self.stimulus_show_time > self.timeout_limit:
                self.record_trial(2000.0, "timeout", "none", False)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            if self.stimulus_pos:
                x, y = self.stimulus_pos
                cv2.circle(frame, (x, y), 40, (0,0,200), -1)
                cv2.circle(frame, (x, y), 40, (255,255,255), 2)

            wrists = {'LEFT_WRIST': landmarks.get('LEFT_WRIST'), 'RIGHT_WRIST': landmarks.get('RIGHT_WRIST')}
            if not self.movement_start_time:
                for name, pos in wrists.items():
                    if pos and self.detect_movement_start(name, pos, current_time):
                        if current_time >= self.stimulus_show_time:
                            self.movement_start_time = current_time
                            break

            touched_joint = None
            dist = float('inf')
            for name, pos in wrists.items():
                if pos:
                    d = math.hypot(pos[0] - x, pos[1] - y)
                    if d < dist and d <= 45:
                        dist = d
                        touched_joint = name

            if touched_joint:
                touch_time = current_time
                latency = max(0, (self.movement_start_time or self.stimulus_show_time) - self.stimulus_show_time)
                movement = max(0, touch_time - (self.movement_start_time or self.stimulus_show_time))
                total = latency + movement
                anatomical = self.anatomical_joint(touched_joint)
                self.record_trial(total * 1000, dist, anatomical, True)
                self.phase = "feedback"
                self.feedback_start = current_time

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("correct", False)
            cv2.putText(frame, "OK" if ok else "Timeout", (w//2-60, h//2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,0) if ok else (0,0,200), 2)
            if current_time - self.feedback_start > 0.8:
                if self.current_trial < self.trials_total:
                    self.current_trial += 1
                    self.phase = "fixation"
                    self.trial_start_time = current_time
                    self.movement_start_time = None
                else:
                    self.signal_phase = "end"
                    self.signal_start = current_time

        return self.phase == "complete" and self.signal_phase == "end"

    def record_trial(self, total_rt, accuracy, joint, correct):
        latency = max(0, total_rt - (accuracy if isinstance(accuracy, (int,float)) else 0))
        movement = total_rt - latency
        trial = {
            "trial": self.current_trial + 1,
            "latency_ms": round(latency, 1),
            "movement_ms": round(movement, 1),
            "total_rt_ms": round(total_rt, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int,float)) else accuracy,
            "joint_used": joint,
            "correct": correct,
            "left_of_midline": self.pos_info["left_of_midline"],
            "above_shoulder_line": self.pos_info["above_shoulder_line"],
            "quadrant": self.pos_info["quadrant"]
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data