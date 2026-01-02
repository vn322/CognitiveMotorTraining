# blocks/choice_reaction.py 
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
        self.trials_total = config.get("choice_reaction", {}).get("trials", 13)
        self.start_time = None
        self.stimulus_show_time = None
        self.user_sequence = []

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
        w = frame.shape[1]
        x0, y0 = w - 200, 40
        color_map = {"red": (0,0,200), "blue": (200,0,0), "yellow": (0,200,200), "green": (0,200,0)}
        if isinstance(item, list):
            for i, color in enumerate(item):
                x = x0 + i * 65
                y = y0
                c = color_map[color]
                cv2.circle(frame, (x, y), 20, c, -1)
                cv2.circle(frame, (x, y), 20, (255,255,255), 2)
        else:
            x, y = w - 100, 60
            color, shape = item
            c = color_map[color]
            if shape == "circle":
                cv2.circle(frame, (x, y), 20, c, -1)
                cv2.circle(frame, (x, y), 20, (255,255,255), 2)
            elif shape == "square":
                cv2.rectangle(frame, (x-20,y-20), (x+20,y+20), c, -1)
                cv2.rectangle(frame, (x-20,y-20), (x+20,y+20), (255,255,255), 2)

    def safe_pos(self, h, w, landmarks, cx, cy, R):
        for _ in range(15):
            a = random.uniform(0, 2 * math.pi)
            x = int(cx + R * math.cos(a))
            y = int(cy + R * math.sin(a))
            safe = True
            for wrist in [landmarks.get('LEFT_WRIST'), landmarks.get('RIGHT_WRIST')]:
                if wrist and (wrist[0]-x)**2 + (wrist[1]-y)**2 < 120**2:
                    safe = False
                    break
            if safe:
                return (x, y)
        return (cx + R, cy)

    def start_trial(self):
        self.current_trial = 0
        self.phase = "start_signal"
        self.start_time = time.time()
        self.user_sequence = []

    def anatomical_joint(self, joint):
        """✅ КОРРЕКЦИЯ ДЛЯ ЗЕРКАЛЬНОГО ИЗОБРАЖЕНИЯ"""
        flip = self.config.get("camera", {}).get("flip_horizontal", True)
        if not flip:
            return joint
        if joint == "LEFT_WRIST":
            return "RIGHT_WRIST"
        elif joint == "RIGHT_WRIST":
            return "LEFT_WRIST"
        return joint

    def process_frame(self, frame, landmarks):
        current_time = time.time()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        R = min(w, h) // 3

        self.draw_skeleton(frame, landmarks)
        cv2.putText(frame, f"Trials: {self.current_trial} / {self.trials_total}", (10,50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

        if self.phase == "start_signal":
            cv2.putText(frame, "Start", (w//2 - 80, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3)
            if current_time - self.start_time > 1.2:
                self.phase = "stimulus"
                self.stimulus_show_time = current_time
                self._generated = False

        elif self.phase == "stimulus":
            if not getattr(self, '_generated', False):
                self._generated = True

                if self.mode == "simple":
                    # ✅ СЛУЧАЙНЫЙ ОБРАЗЕЦ (50/50)
                    self.target = (random.choice(["red", "blue"]), "circle")
                    pos = self.safe_pos(h, w, landmarks, cx, cy, R)
                    # ✅ 70% — совпадает с образом, 30% — дистрактор
                    if random.random() < 0.7:
                        self.stimuli = [(self.target[0], "circle", pos)]
                    else:
                        # дистрактор — противоположный цвет
                        distractor = "blue" if self.target[0] == "red" else "red"
                        self.stimuli = [(distractor, "circle", pos)]

                elif self.mode == "complex":
                    colors = ["red", "blue", "green", "yellow"]
                    shapes = ["circle", "square"]
                    self.target = (random.choice(colors), random.choice(shapes))
                    pos1 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    pos2 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    pos3 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    while abs(pos1[0]-pos2[0]) < 100 and abs(pos1[1]-pos2[1]) < 100:
                        pos2 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    while (abs(pos1[0]-pos3[0]) < 100 and abs(pos1[1]-pos3[1]) < 100) or \
                          (abs(pos2[0]-pos3[0]) < 100 and abs(pos2[1]-pos3[1]) < 100):
                        pos3 = self.safe_pos(h, w, landmarks, cx, cy, R)

                    c, s = self.target
                    fake_color = (random.choice([x for x in colors if x != c]), s)
                    fake_shape = (c, "square" if s == "circle" else "circle")
                    self.stimuli = [
                        (c, s, pos1),
                        (fake_color[0], fake_color[1], pos2),
                        (fake_shape[0], fake_shape[1], pos3)
                    ]
                    random.shuffle(self.stimuli)

                elif self.mode == "sequence":
                    colors = ["red", "blue", "yellow", "green"]
                    self.target = random.sample(colors, 2)
                    pos1 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    pos2 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    while abs(pos1[0]-pos2[0]) < 100 and abs(pos1[1]-pos2[1]) < 100:
                        pos2 = self.safe_pos(h, w, landmarks, cx, cy, R)
                    self.stimuli = [
                        (self.target[0], "circle", pos1),
                        (self.target[1], "circle", pos2)
                    ]

            # ✅ ТАЙМАУТ ДЛЯ SIMPLE = 0.9 СЕК
            timeout = 0.9 if self.mode == "simple" else (5.0 if self.mode == "sequence" else 2.0)
            elapsed = current_time - self.stimulus_show_time
            if elapsed > timeout:
                if self.mode == "simple":
                    # ✅ distractor + timeout = correct
                    is_distractor = self.stimuli[0][0] != self.target[0]
                    correct = is_distractor
                elif self.mode == "sequence":
                    correct = (self.user_sequence == self.target)
                else:
                    correct = False
                self.record_trial(timeout * 1000, 0.0, "none", correct)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            self.draw_model_panel(frame, self.target if self.mode != "sequence" else self.target)

            color_map = {"red":(0,0,200), "blue":(200,0,0), "green":(0,200,0), "yellow":(0,200,200)}
            for color, shape, (x, y) in self.stimuli:
                c = color_map[color]
                if shape == "circle":
                    cv2.circle(frame, (x, y), 50, c, -1)
                    cv2.circle(frame, (x, y), 50, (255,255,255), 3)
                elif shape == "square":
                    cv2.rectangle(frame, (x-50,y-50), (x+50,y+50), c, -1)
                    cv2.rectangle(frame, (x-50,y-50), (x+50,y+50), (255,255,255), 3)

            wrists = {k: v for k, v in landmarks.items() if k.endswith("WRIST")}
            for name, wrist in wrists.items():
                if not wrist:
                    continue
                if self.mode in ["simple", "complex"]:
                    for color, shape, (x, y) in self.stimuli:
                        dx = wrist[0] - x
                        dy = wrist[1] - y
                        if dx*dx + dy*dy <= 60**2:
                            is_target = (color, shape) == self.target
                            joint = self.anatomical_joint(name)  # ✅ КОРРЕКЦИЯ
                            self.record_trial(elapsed * 1000, math.hypot(dx, dy), joint, is_target)
                            self.phase = "feedback"
                            self.feedback_start = current_time
                            return False
                elif self.mode == "sequence" and len(self.user_sequence) < 2:
                    for i, (color, _, (x, y)) in enumerate(self.stimuli):
                        dx = wrist[0] - x
                        dy = wrist[1] - y
                        if dx*dx + dy*dy <= 60**2:
                            if not self.user_sequence or self.user_sequence[-1] != color:
                                self.user_sequence.append(color)
                                if len(self.user_sequence) == 2:
                                    correct = (self.user_sequence == self.target)
                                    self.record_trial(elapsed * 1000, 0.0, "sequence", correct)
                                    self.phase = "feedback"
                                    self.feedback_start = current_time
                                    return False
                            break

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("correct", False)
            cv2.putText(frame, "OK" if ok else "ERR", (w//2-30, h//2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,0) if ok else (0,0,200), 2)
            if current_time - getattr(self, 'feedback_start', current_time) > 0.8:
                if self.current_trial < self.trials_total - 1:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.start_time = time.time()
                    self.user_sequence = []
                else:
                    self.phase = "complete"

        return self.phase == "complete"

    def record_trial(self, rt_ms, accuracy, joint, correct):
        latency = max(0, round(rt_ms * 0.2, 1))
        movement = max(0, round(rt_ms * 0.8, 1))
        trial = {
            "trial": self.current_trial + 1,
            "latency_ms": latency,
            "movement_ms": movement,
            "total_rt_ms": round(rt_ms, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int, float)) else accuracy,
            "joint_used": joint,
            "correct": correct
        }
        if self.mode == "sequence":
            trial["sequence_target"] = str(self.target)
            trial["user_sequence"] = str(self.user_sequence)
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data