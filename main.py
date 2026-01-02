# main.py
import sys
import cv2
import numpy as np
import json
import os
import time
import mediapipe as mp
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFrame, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

from blocks.simple_reaction import SimpleReactionBlock
from blocks.choice_reaction import ChoiceReactionBlock
from blocks.defense_reaction import DefenseReactionBlock
from blocks.moving_object import MovingObjectBlock
from blocks.perturbation_recovery import PerturbationRecoveryBlock
from blocks.tracking_reaction import TrackingReactionBlock
from blocks.combined_reaction import CombinedReactionBlock

CONFIG_PATH = "config/default.json"

def load_config():
    default_config = {
        "camera": {"index": 0, "flip_horizontal": True},
        "simple_reaction": {"trials": 14},
        "choice_reaction": {"trials": 13},
        "defense_reaction": {"trials": 14},
        "moving_object": {"trials": 14},
        "perturbation_recovery": {"trials": 13},
        "tracking_reaction": {"trials": 3},
        "combined_reaction": {"trials": 24}
    }
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        else:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(default_config, f, indent=2)
            return default_config
    except:
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2)
        return default_config

CONFIG = load_config()
LANDMARKS_MAP = {'HEAD':0, 'LEFT_SHOULDER':11, 'RIGHT_SHOULDER':12, 'LEFT_WRIST':15, 'RIGHT_WRIST':16}

def _patch_mediapipe():
    if hasattr(sys, '_MEIPASS'):
        import mediapipe as mp
        original_get_path = mp.solutions.pose._get_path
        def patched_get_path(path):
            full = os.path.join(sys._MEIPASS, 'mediapipe', path)
            if os.path.exists(full):
                return full
            return original_get_path(path)
        mp.solutions.pose._get_path = patched_get_path
_patch_mediapipe()

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    block_finished = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self.running = True
        self.pose = mp.solutions.pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.block = None
        self.block_name = ""

    def set_block(self, block, name):
        self.block = block
        self.block_name = name

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            for i in range(1, 5):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    break
            else:
                self.block_finished.emit("Error", [])
                return

        try:
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue

                if CONFIG["camera"]["flip_horizontal"]:
                    frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]
                if self.block and hasattr(self.block, 'signal_phase'):
                    if self.block.signal_phase == "start":
                        cv2.putText(frame, "Start", (w//2-80, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3)
                    elif self.block.signal_phase == "end":
                        cv2.putText(frame, "Stop", (w//2-70, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0,0,255), 3)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.pose.process(rgb)
                landmarks_px = {}
                if results.pose_landmarks:
                    for name, idx in LANDMARKS_MAP.items():
                        lm = results.pose_landmarks.landmark[idx]
                        if lm.visibility > 0.5:
                            landmarks_px[name] = (int(lm.x * w), int(lm.y * h))

                if self.block:
                    is_complete = self.block.process_frame(frame, landmarks_px)
                    if is_complete:
                        self.block_finished.emit(self.block_name, self.block.get_results())
                        self.block = None

                self.frame_ready.emit(frame)
                time.sleep(0.01)
        except Exception as e:
            print(f"[VideoThread] {e}")
        finally:
            cap.release()

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CognitiveMotorTraining")
        self.resize(1280, 720)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.video_label = QLabel()
        self.video_label.setFixedSize(860, 640)
        self.video_label.setStyleSheet("background: black;")
        layout.addWidget(self.video_label)

        panel = QFrame()
        panel.setFixedWidth(400)
        panel_layout = QVBoxLayout(panel)
        title = QLabel("Motor Intelligence Test")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        panel_layout.addWidget(title)

        tests = [
            ("Simple Reaction", "simple_reaction"),
            ("Simple Choice", "simple_choice"),
            ("Complex Choice", "complex_choice"),
            ("Sequence", "sequence"),
            ("Defense Reaction", "defense"),
            ("Moving Object", "moving"),
            ("Perturbation Recovery", "perturbation"),
            ("Tracking Reaction", "tracking"),
            ("Combined: Selection + Tracking", "combined_selection"),
            ("Combined: Dynamic Switch", "combined_full")
        ]
        self.buttons = []
        for text, tag in tests:
            btn = QPushButton(text)
            btn.setFixedHeight(42)
            btn.setStyleSheet("background: #3498db; color: white; border-radius: 8px;")
            btn.clicked.connect(lambda _, t=tag: self.start_test(t))
            panel_layout.addWidget(btn)
            self.buttons.append(btn)

        self.btn_abort = QPushButton("⏹ Abort")
        self.btn_abort.setFixedHeight(42)
        self.btn_abort.setStyleSheet("background: #e74c3c; color: white; border-radius: 8px;")
        self.btn_abort.clicked.connect(self.abort)
        self.btn_abort.setEnabled(False)
        panel_layout.addWidget(self.btn_abort)

        self.btn_export = QPushButton("💾 Export + Report")
        self.btn_export.setFixedHeight(42)
        self.btn_export.setStyleSheet("background: #2c3e50; color: white; border-radius: 8px;")
        self.btn_export.clicked.connect(self.export_with_report)
        self.btn_export.setEnabled(False)
        panel_layout.addWidget(self.btn_export)

        layout.addWidget(panel)
        footer = QLabel("Ermakov.AV, 2026")
        footer.setFont(QFont("Segoe UI", 9))
        footer.setStyleSheet("color: #95a5a6;")
        footer.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignBottom)
        self.statusBar().addWidget(footer, 1)

        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_frame)
        self.video_thread.block_finished.connect(self.on_finish)
        self.video_thread.start()

        self.results = []
        self.current = None

    def update_frame(self, img):
        h, w, ch = img.shape
        qt = QImage(img.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt).scaled(860, 640))

    def start_test(self, tag):
        if self.current: return
        mapping = {
            "simple_reaction": (SimpleReactionBlock(CONFIG), "Simple Reaction"),
            "simple_choice": (ChoiceReactionBlock(CONFIG, mode="simple"), "Choice: Simple"),
            "complex_choice": (ChoiceReactionBlock(CONFIG, mode="complex"), "Choice: Complex"),
            "sequence": (ChoiceReactionBlock(CONFIG, mode="sequence"), "Choice: Sequence"),
            "defense": (DefenseReactionBlock(CONFIG), "Defense Reaction"),
            "moving": (MovingObjectBlock(CONFIG), "Moving Object"),
            "perturbation": (PerturbationRecoveryBlock(CONFIG), "Perturbation Recovery"),
            "tracking": (TrackingReactionBlock(CONFIG), "Tracking Reaction"),
            "combined_selection": (CombinedReactionBlock(CONFIG, mode="selection"), "Combined Reaction: Selection + Tracking"),
            "combined_full": (CombinedReactionBlock(CONFIG, mode="full"), "Combined Reaction: Dynamic Switch")
        }
        if tag in mapping:
            block, name = mapping[tag]
            block.start_trial()
            self.video_thread.set_block(block, name)
            self.current = name
            for btn in self.buttons:
                btn.setEnabled(False)
            self.btn_abort.setEnabled(True)

    def abort(self):
        self.video_thread.block = None
        self.current = None
        for btn in self.buttons:
            btn.setEnabled(True)
        self.btn_abort.setEnabled(False)

    def on_finish(self, name, data):
        self.results.append((name, data))
        self.current = None
        for btn in self.buttons:
            btn.setEnabled(True)
        self.btn_export.setEnabled(True)

    def export_with_report(self):
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")
        csv_path = f"report_{timestamp}.csv"

        try:
            # === Нормализация имён полей ===
            field_map = {
                "trial": "trial",                 
                "latency_ms": "latency_ms",
                "movement_ms": "movement_ms",
                "total_rt_ms": "total_rt_ms",
                "rt_ms": "total_rt_ms",
                "accuracy_mm": "accuracy_mm",     
                "displacement_mm": "displacement_mm",
                "timing_error_ms": "timing_error_ms",
                "tracking_error_mm": "tracking_error_mm",
                "coverage_%": "coverage_%",
                "coverage": "coverage_%",
                "prediction_error_px": "prediction_error_px",
                "update_latency_ms": "update_latency_ms",
                "adaptation_rate": "adaptation_rate",
                "movement_economy": "movement_economy",
                "joint_used": "joint",
                "joint": "joint",
                "correct": "correct",
                "success": "correct",
            }

            all_fields = set()
            norm_results = []
            for block_name, results_list in self.results:
                norm_list = []
                for r in results_list:
                    norm_r = {"Block": block_name}
                    for k, v in r.items():
                        k_clean = k.replace(" ", "_").strip().lower()
                        k_norm = field_map.get(k_clean, k_clean.lower())
                        norm_r[k_norm] = v
                    norm_list.append(norm_r)
                    all_fields.update(norm_r.keys())
                norm_results.append((block_name, norm_list))

            # Фиксированный порядок
            base_order = [
                "Block", "Trial", "Latency_ms", "Movement_ms", "Total_RT_ms",
                "Accuracy_mm", "Displacement_mm", "Timing_error_ms",
                "Tracking_error_mm", "Coverage_%", "Prediction_error_px",
                "Update_latency_ms", "Adaptation_rate", "Movement_economy",
                "Joint", "Correct"
            ]
            fieldnames = []
            for f in base_order:
                if f in all_fields:
                    fieldnames.append(f)
                    all_fields.discard(f)
            fieldnames.extend(sorted(all_fields))

            # Запись CSV
            import csv
            with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for block_name, norm_list in norm_results:
                    for r in norm_list:
                        row = {}
                        for k in fieldnames:
                            if k == "Block":
                                continue
                            v = r.get(k)
                            if k == "Latency_ms" and (v is None or v == "" or pd.isna(v)):
                                total = r.get("Total_RT_ms", 0)
                                v = max(0, round(total * 0.2, 1)) if total else 0
                            elif k == "Movement_ms" and (v is None or v == "" or pd.isna(v)):
                                total = r.get("Total_RT_ms", 0)
                                v = max(0, round(total * 0.8, 1)) if total else 0
                            if k == "Correct":
                                v = "Yes" if r.get(k, False) else "No"
                            if k == "Joint" and (v is None or v == ""):
                                v = "none"
                            row[k] = v
                        row["Block"] = r["Block"]
                        writer.writerow(row)

            # Генерация PDF через report_generator
            from report_generator import generate_pdf_report
            pdf_path = generate_pdf_report(csv_path)
            if pdf_path and os.path.exists(pdf_path):
                QMessageBox.information(self, "Success", f"Сохранено:\n{csv_path}\n{pdf_path}")
            else:
                QMessageBox.warning(self, "Ошибка", "PDF не создан. См. консоль.")

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}\n{traceback.format_exc()}")

    def closeEvent(self, e):
        self.video_thread.stop()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())