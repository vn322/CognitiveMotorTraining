# blocks/perturbation_recovery.py 
import cv2
import numpy as np
import time
import random
import math

class PerturbationRecoveryBlock:
    def __init__(self, config):
        self.config = config
        self.trial_data = []
        self.current_trial = 0
        self.phase = "idle"
        self.signal_phase = None
        self.signal_start = None
        self.trial_start_time = None
        self.stimulus_start = None
        self.zone_center = None
        self.stimulus_pos = None
        self.stimulus_velocity = None
        self.perturbation_time = None
        self.perturbation_applied = False
        self.perturbation_velocity_delta = 0
        self.trials_total = config.get("perturbation_recovery", {}).get("trials", 12)
        self.joint_hist = {}
        self.prediction_error = None
        self.update_latency = None
        self.is_perturbation_trial = False

    def draw_skeleton(self, frame, landmarks):
        left_color = (76, 175, 80)    # зелёный — левая
        right_color = (255, 193, 7)   # жёлтый — правая
        white = (255, 255, 255)
        gray = (200, 200, 200)

        def g(n):
            return landmarks.get(n)

        # Плечи
        ls, rs = g('LEFT_SHOULDER'), g('RIGHT_SHOULDER')
        # Локти
        le, re = g('LEFT_ELBOW'), g('RIGHT_ELBOW')
        # Запястья
        lw, rw = g('LEFT_WRIST'), g('RIGHT_WRIST')

        # Соединения
        if ls and le:
            cv2.line(frame, ls, le, left_color, 2)
        if le and lw:
            cv2.line(frame, le, lw, left_color, 2)
        if rs and re:
            cv2.line(frame, rs, re, right_color, 2)
        if re and rw:
            cv2.line(frame, re, rw, right_color, 2)
        if ls and rs:
            cv2.line(frame, ls, rs, gray, 2)

        # Запястья (круги с обводкой)
        for p, color in [(lw, left_color), (rw, right_color)]:
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

    def detect_movement_start_toward_zone_after_perturbation(self, joint_name, wrist_pos, zone_center, current_time, perturb_time):
        if not wrist_pos or not zone_center:
            return False, 0.0, None
        hist = self.joint_hist.setdefault(joint_name, [])
        hist.append((wrist_pos[0], wrist_pos[1], current_time))
        if len(hist) < 5:
            return False, 0.0, None
        # Используем только точки после пертурбации
        hist_after = [h for h in hist if h[2] >= perturb_time]
        if len(hist_after) < 3:
            return False, 0.0, None
        t0, t1 = hist_after[-2][2], hist_after[-1][2]
        dt = t1 - t0
        if dt < 0.001:
            return False, 0.0, None
        v_x = (hist_after[-1][0] - hist_after[-2][0]) / dt
        v_y = (hist_after[-1][1] - hist_after[-2][1]) / dt
        speed = math.hypot(v_x, v_y)
        if speed < 100:
            return False, 0.0, None
        dx_zone = zone_center[0] - wrist_pos[0]
        dy_zone = zone_center[1] - wrist_pos[1]
        dist_zone = math.hypot(dx_zone, dy_zone)
        if dist_zone < 1:
            return False, 0.0, None
        ux_zone = dx_zone / dist_zone
        uy_zone = dy_zone / dist_zone
        cos_angle = (v_x * ux_zone + v_y * uy_zone) / (speed + 1e-6)
        if cos_angle > 0.7:
            return True, cos_angle, t1
        return False, 0.0, None

    def process_frame(self, frame, landmarks):
        current_time = time.time()
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

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
                self.zone_center = (cx, cy)
                self.stimulus_velocity = random.uniform(800, 1200)
                self.direction = random.choice([1, -1])
                if self.direction == 1:
                    self.stimulus_pos = [-60, cy]
                else:
                    self.stimulus_pos = [w + 60, cy]

                self.is_perturbation_trial = (random.random() < 0.3)
                if self.is_perturbation_trial:
                    self.perturbation_velocity_delta = random.choice([-300, 300])
                    self.perturbation_time = self.stimulus_start + random.uniform(0.4, 0.6)
                else:
                    self.perturbation_time = None
                    self.perturbation_velocity_delta = 0

                self.perturbation_applied = False
                self.prediction_error = None
                self.update_latency = None

        elif self.phase == "stimulus":
            elapsed = current_time - self.stimulus_start
            if elapsed > 2.0:
                self.record_trial(2000.0, "timeout", "none", 0.0, False, 0.0, 0.0, 0.0, 0.0)
                self.phase = "feedback"
                self.feedback_start = current_time
                return False

            # Обновление позиции
            base_dx = self.direction * self.stimulus_velocity / 30
            if self.is_perturbation_trial and not self.perturbation_applied and current_time >= self.perturbation_time:
                self.perturbation_applied = True
                # Идеальное положение в момент пертурбации:
                ideal_x = self.stimulus_pos[0] + base_dx * (current_time - self.perturbation_time) * 30
                # Фактическое с новой скоростью:
                new_v = self.stimulus_velocity + self.perturbation_velocity_delta
                actual_x = self.stimulus_pos[0] + base_dx * (current_time - self.perturbation_time) * 30 + (self.perturbation_velocity_delta / 30) * (current_time - self.perturbation_time) * 30
                self.prediction_error = abs(actual_x - ideal_x)

            dx = base_dx
            if self.perturbation_applied:
                dx += self.perturbation_velocity_delta / 30
            self.stimulus_pos[0] += dx
            x, y = int(self.stimulus_pos[0]), int(self.stimulus_pos[1])
            zx, zy = self.zone_center

            # Оси и зона — всегда видны
            gray = (158, 158, 158)
            cv2.line(frame, (zx, 0), (zx, h), gray, 1)
            cv2.line(frame, (0, zy), (w, zy), gray, 1)
            cv2.circle(frame, (zx, zy), 80, (255, 255, 255), 2)
            cv2.circle(frame, (zx, zy), 80, (100, 100, 100), 1)
            cv2.circle(frame, (x, y), 40, (0, 0, 200), -1)

            wrists = {k: v for k, v in landmarks.items() if k.endswith("WRIST")}
            for name, wrist in wrists.items():
                if wrist and (wrist[0] - zx)**2 + (wrist[1] - zy)**2 <= 100**2:
                    touch_time = current_time

                    # ✅ БЕЗОПАСНАЯ ИНИЦИАЛИЗАЦИЯ economy
                    economy = 0.0
                    if self.is_perturbation_trial and self.prediction_error is not None:
                        _, econ_val, t_start = self.detect_movement_start_toward_zone_after_perturbation(
                            name, wrist, (zx, zy), current_time, self.perturbation_time
                        )
                        if econ_val is not None:
                            economy = econ_val
                        if t_start and t_start > self.perturbation_time:
                            self.update_latency = (t_start - self.perturbation_time) * 1000

                    # Время совпадения
                    if self.perturbation_applied:
                        t1 = self.perturbation_time - self.stimulus_start
                        d1 = self.stimulus_velocity * t1
                        d2 = (self.stimulus_velocity + self.perturbation_velocity_delta) * (touch_time - self.perturbation_time)
                        actual_distance = d1 + d2
                        if self.direction == 1:
                            t_align_actual = (-60 + actual_distance) / (self.stimulus_velocity + self.perturbation_velocity_delta)
                        else:
                            t_align_actual = (w + 60 - actual_distance) / (self.stimulus_velocity + self.perturbation_velocity_delta)
                    else:
                        if self.direction == 1:
                            t_align_actual = (zx - (-60)) / self.stimulus_velocity
                        else:
                            t_align_actual = ((w + 60) - zx) / self.stimulus_velocity

                    timing_error = (touch_time - self.stimulus_start) - t_align_actual
                    dist = math.hypot(wrist[0] - zx, wrist[1] - zy)
                    success = abs(timing_error * 1000) < 50
                    joint = self.anatomical_joint(name)

                    self.record_trial(
                        (touch_time - self.stimulus_start) * 1000,
                        dist,
                        joint,
                        timing_error * 1000,
                        success,
                        self.prediction_error or 0.0,
                        self.update_latency or 0.0,
                        0.0,  # adaptation_rate — post-hoc
                        economy
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

    def record_trial(self, rt_ms, accuracy, joint, timing_error_ms, success,
                     prediction_error_px, update_latency_ms, adaptation_rate, economy):
        trial = {
            "trial": self.current_trial + 1,
            "total_rt_ms": round(rt_ms, 1),
            "accuracy_mm": round(accuracy, 1) if isinstance(accuracy, (int, float)) else accuracy,
            "joint_used": joint,
            "timing_error_ms": round(timing_error_ms, 1),
            "success": success,
            "prediction_error_px": round(prediction_error_px, 1),
            "update_latency_ms": round(update_latency_ms, 1),
            "adaptation_rate": round(adaptation_rate, 3),
            "movement_economy": round(economy, 2)
        }
        self.trial_data.append(trial)

    def get_results(self):
        # Post-hoc: вычисление adaptation_rate по группе пертурбационных триалов
        perturbation_trials = [t for t in self.trial_data if t.get("prediction_error_px", 0) > 0.1]
        if len(perturbation_trials) >= 3:
            errors = [t["prediction_error_px"] for t in perturbation_trials]
            try:
                import numpy as np
                trials = np.arange(1, len(errors)+1)
                log_errors = np.log(np.array(errors) + 1e-3)
                coeffs = np.polyfit(trials, log_errors, 1)
                adaptation_rate = -coeffs[0]
                for t in perturbation_trials:
                    t["adaptation_rate"] = round(adaptation_rate, 3)
            except:
                pass
        return self.trial_data