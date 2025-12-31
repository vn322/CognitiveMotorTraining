# blocks/choice_reaction.py — ФИНАЛЬНАЯ РАБОЧАЯ ВЕРСИЯ
import cv2
import numpy as np
import time
import random
import math

class ChoiceReactionBlock:
    def __init__(self, config, mode="simple"):
        self.config = config
        self.mode = mode
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.target = None
        self.stimuli = []
        self.user_pressed = False
        self.trials_total = config.get("choice_reaction", {}).get("trials", 12)
        self.start_time = None
        self.timeout = 2.0

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

    def draw_model_panel(self, frame, item):
        """✅ РАБОТАЕТ ДАЖЕ ЕСЛИ item = ['red','blue'] (sequence)"""
        w = frame.shape[1]
        x0, y0 = w - 180, 60
        color_map = {"red": (0,0,200), "blue": (200,0,0), "yellow": (0,200,200), "green": (0,200,0)}
        if isinstance(item, list):  # sequence
            for i, color in enumerate(item):
                x = x0 + i * 70
                y = y0
                c = color_map[color]
                cv2.circle(frame, (x, y), 20, c, -1)
                cv2.circle(frame, (x, y), 20, (255,255,255), 2)
        else:  # simple/complex: (color, shape)
            x, y = w - 100, 60
            color, shape = item
            c = color_map[color]
            if shape == "circle":
                cv2.circle(frame, (x, y), 20, c, -1)
                cv2.circle(frame, (x, y), 20, (255,255,255), 2)
            elif shape == "square":
                cv2.rectangle(frame, (x-20,y-20), (x+20,y+20), c, -1)
                cv2.rectangle(frame, (x-20,y-20), (x+20,y+20), (255,255,255), 2)
            elif shape == "triangle":
                pts = np.array([[x,y-20], [x-18,y+10], [x+18,y+10]])
                cv2.fillPoly(frame, [pts], c)
                cv2.polylines(frame, [pts], True, (255,255,255), 2)

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
        self.current_trial = 0
        self.phase = "start_signal"
        self.start_time = time.time()
        self.user_pressed = False

        # ✅ ИНИЦИАЛИЗАЦИЯ TARGET 
        if self.mode == "simple":
            self.target = ("red", "circle")
        elif self.mode == "complex":
            colors = ["red", "blue", "green", "yellow"]
            shapes = ["circle", "square", "triangle"]
            self.target = (random.choice(colors), random.choice(shapes))
        elif self.mode == "sequence":
            colors = ["red", "blue", "yellow", "green"]
            self.target = random.sample(colors, 2)

    def process_frame(self, frame, landmarks):
        current_time = time.time()
        h, w = frame.shape[:2]

        self.draw_skeleton(frame, landmarks)

        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 4)
        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "start_signal":
            cv2.putText(frame, "Start", (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
            if current_time - self.start_time > 1.2:
                self.phase = "stimulus"
                self.stimulus_start = current_time

                if self.mode == "simple":
                    if random.random() < 0.7:
                        self.stimuli = [("red", "circle", (w//2, h//2))]
                    else:
                        self.stimuli = [("blue", "circle", (w//2, h//2))]

                elif self.mode == "complex":
                    colors = ["red", "blue", "green", "yellow"]
                    shapes = ["circle", "square", "triangle"]
                    fake_candidates = []
                    for c in colors:
                        if c != self.target[0]:
                            fake_candidates.append((c, self.target[1]))
                    for s in shapes:
                        if s != self.target[1]:
                            fake_candidates.append((self.target[0], s))
                    fake = random.choice(fake_candidates)
                    pos1 = (w//2 - 120, h//2)
                    pos2 = (w//2 + 120, h//2)
                    if random.choice([True, False]):
                        self.stimuli = [
                            (self.target[0], self.target[1], pos1),
                            (fake[0], fake[1], pos2)
                        ]
                    else:
                        self.stimuli = [
                            (fake[0], fake[1], pos1),
                            (self.target[0], self.target[1], pos2)
                        ]

        elif self.phase == "stimulus":
            elapsed = current_time - self.stimulus_start
            if elapsed > self.timeout:
                self.record_trial(self.timeout*1000, "timeout", "none", False)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            self.draw_model_panel(frame, self.target)

            color_map = {"red":(0,0,200), "blue":(200,0,0), "yellow":(0,200,200), "green":(0,200,0)}
            for color, shape, (x, y) in self.stimuli:
                c = color_map[color]
                if shape == "circle":
                    cv2.circle(frame, (x, y), 40, c, -1)
                    cv2.circle(frame, (x, y), 40, (255,255,255), 2)
                elif shape == "square":
                    cv2.rectangle(frame, (x-40,y-40), (x+40,y+40), c, -1)
                    cv2.rectangle(frame, (x-40,y-40), (x+40,y+40), (255,255,255), 2)
                elif shape == "triangle":
                    pts = np.array([[x,y-40], [x-35,y+20], [x+35,y+20]])
                    cv2.fillPoly(frame, [pts], c)
                    cv2.polylines(frame, [pts], True, (255,255,255), 2)

            wrists = {k: v for k, v in landmarks.items() if k.endswith("WRIST")}
            for name, wrist in wrists.items():
                if not wrist:
                    continue
                for i, (color, shape, pos) in enumerate(self.stimuli):
                    dx = wrist[0] - pos[0]
                    dy = wrist[1] - pos[1]
                    if dx*dx + dy*dy <= 50**2:
                        if not self.user_pressed:
                            self.user_pressed = True
                            if self.mode == "simple":
                                is_target = (color, shape) == self.target
                                correct = is_target
                            elif self.mode == "complex":
                                correct = (color, shape) == self.target
                            dist = math.hypot(dx, dy)
                            joint = self.anatomical_joint(name)
                            self.record_trial(elapsed*1000, dist, joint, correct)
                            self.phase = "feedback"
                            self.feedback_start = current_time
                        break

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("correct", False)
            cv2.putText(frame, "OK" if ok else "ERR", (w//2-30, h//2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,0) if ok else (0,0,200), 2)
            if current_time - self.feedback_start > 0.8:
                if self.current_trial < self.trials_total:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.start_time = current_time
                    self.user_pressed = False
                else:
                    self.phase = "complete"

        return self.phase == "complete"
    def record_trial(self, rt_ms, accuracy, joint, correct):
        trial = {
            "trial": self.current_trial + 1,
            "mode": self.mode,
            "target_color": self.target[0] if isinstance(self.target, tuple) else "",
            "target_shape": self.target[1] if isinstance(self.target, tuple) else "",
            "correct": correct,
            "total_rt_ms": round(rt_ms, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int, float)) else accuracy,
            "joint_used": joint
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data