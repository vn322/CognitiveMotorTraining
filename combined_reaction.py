# blocks/combined_reaction.py 
import cv2
import numpy as np
import time
import random
import math

class CombinedReactionBlock:
    def __init__(self, config, mode="selection"):
        self.config = config
        self.mode = mode  # "selection" или "full"
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.target_color = None
        self.stimuli = []  # [(color, x, y, vx, vy, is_target_original)]
        self.trials_total = config.get("combined_reaction", {}).get("trials", 12)
        self.start_time = None
        self.stimulus_show_time = None
        self.color_swap_times = []
        self.color_swapped = [False, False]

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

    def draw_model_panel(self, frame, color):
        w = frame.shape[1]
        x, y = w - 100, 60
        color_map = {"red": (0,0,200), "blue": (200,0,0), "yellow": (0,200,200), "green": (0,200,0)}
        c = color_map[color]
        cv2.circle(frame, (x, y), 25, c, -1)  # ✅ 25 px — эталон
        cv2.circle(frame, (x, y), 25, (255,255,255), 2)

    def safe_pos_on_margin(self, h, w, landmarks, R):
        cx, cy = w // 2, h // 2
        for _ in range(20):
            a = random.uniform(0, 2 * math.pi)
            x = int(cx + R * math.cos(a))
            y = int(cy + R * math.sin(a))
            safe = True
            for wrist in [landmarks.get('LEFT_WRIST'), landmarks.get('RIGHT_WRIST')]:
                if wrist and (wrist[0]-x)**2 + (wrist[1]-y)**2 < 100**2:
                    safe = False
                    break
            if safe:
                return x, y
        return cx + R, cy

    def start_trial(self):
        self.current_trial = 0
        self.phase = "start_signal"
        self.start_time = time.time()

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

                # Образец — случайный цвет
                self.target_color = random.choice(["red", "blue", "green", "yellow"])
                pos1 = self.safe_pos_on_margin(h, w, landmarks, R)
                pos2 = self.safe_pos_on_margin(h, w, landmarks, R)
                while abs(pos1[0]-pos2[0]) < 100 and abs(pos1[1]-pos2[1]) < 100:
                    pos2 = self.safe_pos_on_margin(h, w, landmarks, R)

                # Сниженная скорость: 150–400 px/s
                angle1 = random.uniform(0, 2*math.pi)
                angle2 = random.uniform(0, 2*math.pi)
                speed1 = random.uniform(150, 400)
                speed2 = random.uniform(150, 400)
                vx1, vy1 = speed1 * math.cos(angle1), speed1 * math.sin(angle1)
                vx2, vy2 = speed2 * math.cos(angle2), speed2 * math.sin(angle2)

                self.stimuli = [
                    (self.target_color, pos1[0], pos1[1], vx1, vy1, True),
                    (random.choice([c for c in ["red","blue","green","yellow"] if c != self.target_color]), pos2[0], pos2[1], vx2, vy2, False)
                ]

                # ✅ ДИНАМИЧЕСКОЕ ПЕРЕКЛЮЧЕНИЕ РОЛЕЙ (ТОЛЬКО В FULL)
                if self.mode == "full":
                    t0 = current_time
                    self.color_swap_times = [t0 + 0.8, t0 + 1.6]
                    self.color_swapped = [False, False]

            elapsed = current_time - self.stimulus_show_time
            # ✅ ТАЙМАУТ = 5.0 СЕКУНД
            if elapsed > 5.0:
                self._record_trial(5000.0, 0.0, "timeout", False)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            # Образец
            self.draw_model_panel(frame, self.target_color)

            # ✅ ПЕРЕКЛЮЧЕНИЕ ЦВЕТОВ (full)
            if self.mode == "full":
                for i, t_swap in enumerate(self.color_swap_times):
                    if not self.color_swapped[i] and current_time >= t_swap:
                        # Меняем цвета местами
                        c1, c2 = self.stimuli[0][0], self.stimuli[1][0]
                        self.stimuli[0] = (c2, *self.stimuli[0][1:])
                        self.stimuli[1] = (c1, *self.stimuli[1][1:])
                        self.color_swapped[i] = True

            # Стимулы
            color_map = {"red": (0,0,200), "blue": (200,0,0), "green": (0,200,0), "yellow": (0,200,200)}
            for i, (color, x, y, vx, vy, is_target_orig) in enumerate(self.stimuli):
                # Обновление позиции
                x += vx / 30
                y += vy / 30

                # Отскок от границ
                if x <= 50 or x >= w - 50:
                    vx = -vx
                    x = max(50, min(w - 50, x))
                if y <= 50 or y >= h - 50:
                    vy = -vy
                    y = max(50, min(h - 50, y))

                self.stimuli[i] = (color, x, y, vx, vy, is_target_orig)

                # ✅ 25 px — как у образца
                c = color_map[color]
                cv2.circle(frame, (int(x), int(y)), 25, c, -1)
                cv2.circle(frame, (int(x), int(y)), 25, (255,255,255), 2)

            # Детекция
            wrists = {k: v for k, v in landmarks.items() if k.endswith("WRIST")}
            for name, wrist in wrists.items():
                if wrist:
                    for color, x, y, vx, vy, is_target_orig in self.stimuli:
                        dx = wrist[0] - x
                        dy = wrist[1] - y
                        if dx*dx + dy*dy <= 30**2:
                            # В full — цель = текущий цвет == target_color
                            is_now_target = (color == self.target_color)
                            correct = is_now_target
                            self._record_trial(elapsed * 1000, math.hypot(dx, dy), name, correct)
                            self.phase = "feedback"
                            self.feedback_start = current_time
                            return False

        elif self.phase == "feedback":
            last = self.trial_data[-1] if self.trial_data else {}
            ok = last.get("correct", False)
            cv2.putText(frame, "OK" if ok else "ERR", (w//2-30, h//2+15),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,180,0) if ok else (0,0,200), 2)
            if current_time - self.feedback_start > 0.8:
                if self.current_trial < self.trials_total - 1:
                    self.current_trial += 1
                    self.phase = "start_signal"
                    self.start_time = current_time
                else:
                    self.phase = "complete"

        return self.phase == "complete"

    def _record_trial(self, rt_ms, accuracy, joint, correct):
        trial = {
            "trial": self.current_trial + 1,
            "total_rt_ms": round(rt_ms, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int, float)) else accuracy,
            "joint_used": joint,
            "correct": correct,
            "mode": self.mode
        }
        self.trial_data.append(trial)

    def get_results(self):
        return self.trial_data