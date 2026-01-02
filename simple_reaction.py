# blocks/simple_reaction.py 
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

    def get_safe_position(self, frame_shape):
        h, w = frame_shape
        margin = min(w, h) // 3
        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            return (random.randint(margin, w - margin), margin)
        elif edge == 'bottom':
            return (random.randint(margin, w - margin), h - margin)
        elif edge == 'left':
            return (margin, random.randint(margin, h - margin))
        else:
            return (w - margin, random.randint(margin, h - margin))

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

    def detect_movement_start_toward_stimulus(self, joint_name, wrist_pos, stim_pos, current_time):
        if not wrist_pos or not stim_pos:
            return False, 0.0, 0.0
        hist = self.joint_hist.setdefault(joint_name, [])
        hist.append((wrist_pos[0], wrist_pos[1], current_time))
        if len(hist) < 5: return False, 0.0, 0.0
        t0, t1 = hist[-2][2], hist[-1][2]
        dt = t1 - t0
        if dt < 0.001: return False, 0.0, 0.0
        v_x = (hist[-1][0] - hist[-2][0]) / dt
        v_y = (hist[-1][1] - hist[-2][1]) / dt
        speed = math.hypot(v_x, v_y)
        if speed < 100: return False, 0.0, 0.0
        dx_target = stim_pos[0] - wrist_pos[0]
        dy_target = stim_pos[1] - wrist_pos[1]
        dist_target = math.hypot(dx_target, dy_target)
        if dist_target < 1: return False, 0.0, 0.0
        ux_target = dx_target / dist_target
        uy_target = dy_target / dist_target
        cos_angle = (v_x * ux_target + v_y * uy_target) / (speed + 1e-6)
        return cos_angle > 0.8, cos_angle, current_time - dt

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

        self.draw_skeleton(frame, landmarks)

        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "fixation":
            cv2.putText(frame, "+", (w//2-15, h//2+15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
            if current_time - self.trial_start_time > 0.5:
                self.phase = "stimulus"
                self.stimulus_show_time = current_time
                self.stimulus_pos = self.get_safe_position((h, w))

        elif self.phase == "stimulus":
            if current_time - self.stimulus_show_time > self.timeout_limit:
                self.record_trial(2000.0, "timeout", "none", 0.0, 0.0, False, 0.0)
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
                    if pos:
                        is_start, _, t_start = self.detect_movement_start_toward_stimulus(name, pos, self.stimulus_pos, current_time)
                        if is_start and current_time >= self.stimulus_show_time:
                            self.movement_start_time = t_start
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
                _, economy, _ = self.detect_movement_start_toward_stimulus(touched_joint, landmarks[touched_joint], self.stimulus_pos, touch_time)
                self.record_trial(total * 1000, dist, anatomical, latency * 1000, movement * 1000, True, economy)
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

    def record_trial(self, total_rt, accuracy, joint, latency, movement, correct, economy):
        trial = {
            "trial": self.current_trial + 1,
            "latency_ms": round(latency, 1),
            "movement_ms": round(movement, 1),
            "total_rt_ms": round(total_rt, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int,float)) else accuracy,
            "joint_used": joint,
            "correct": correct,
            "movement_economy": round(economy, 2)
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data