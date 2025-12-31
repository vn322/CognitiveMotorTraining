# main.py — ФИНАЛЬНАЯ ВЕРСИЯ
import sys
import cv2
import numpy as np
import json
import os
import mediapipe as mp
from datetime import datetime
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
    QWidget, QPushButton, QFrame, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

# Импорты блоков
from blocks.simple_reaction import SimpleReactionBlock
from blocks.choice_reaction import ChoiceReactionBlock
from blocks.defense_reaction import DefenseReactionBlock
from blocks.moving_object import MovingObjectBlock
from blocks.tracking_reaction import TrackingReactionBlock

CONFIG_PATH = "config/default.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
        config = {
            "camera": {"index": 0, "flip_horizontal": True},
            "simple_reaction": {"trials": 12, "angles_deg": [0, 45, 90, 135, 180, 225, 270, 315]},
            "choice_reaction": {"trials": 12},
            "defense_reaction": {"trials": 12},
            "moving_object": {"trials": 12},
            "tracking_reaction": {"trials": 3}
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
        return config
    with open(CONFIG_PATH) as f:
        return json.load(f)

CONFIG = load_config()
LANDMARKS_MAP = {'HEAD':0, 'LEFT_SHOULDER':11, 'RIGHT_SHOULDER':12, 'LEFT_WRIST':15, 'RIGHT_WRIST':16}

class VideoThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    block_finished = pyqtSignal(str, list)

    def __init__(self):
        super().__init__()
        self.running = True
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.block = None

    def set_block(self, block):
        self.block = block

    def run(self):
        cap = cv2.VideoCapture(0)
        while self.running and cap.isOpened():
            ret, frame = cap.read()
            if not ret: continue
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
                    self.block_finished.emit(self.block.__class__.__name__, self.block.get_results())
                    self.block = None
            self.frame_ready.emit(frame)
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
            ("Tracking", "tracking")
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

        self.btn_export = QPushButton("💾 Export")
        self.btn_export.setFixedHeight(42)
        self.btn_export.setStyleSheet("background: #2c3e50; color: white; border-radius: 8px;")
        self.btn_export.clicked.connect(self.export)
        self.btn_export.setEnabled(False)
        panel_layout.addWidget(self.btn_export)

        layout.addWidget(panel)
        footer = QLabel("Ermakov.AV, 2025")
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
        if tag == "simple_reaction":
            block = SimpleReactionBlock(CONFIG)
        elif tag == "simple_choice":
            block = ChoiceReactionBlock(CONFIG, mode="simple")
        elif tag == "complex_choice":
            block = ChoiceReactionBlock(CONFIG, mode="complex")
        elif tag == "sequence":
            block = ChoiceReactionBlock(CONFIG, mode="sequence")
        elif tag == "defense":
            block = DefenseReactionBlock(CONFIG)
        elif tag == "moving":
            block = MovingObjectBlock(CONFIG)
        else:  # tracking
            block = TrackingReactionBlock(CONFIG)
        block.start_trial()
        self.video_thread.set_block(block)
        self.current = tag
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
        self.btn_abort.setEnabled(False)
        self.btn_export.setEnabled(True)

    def export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"report_{int(time.time())}.csv", "CSV (*.csv)")
        if not path or not path.endswith(".csv"): 
            path += ".csv"
        import csv
        try:
            with open(path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Block", "Trial", "Latency_ms", "Movement_ms", "Total_RT_ms",
                    "Accuracy_mm", "Displacement_mm", "Timing_error_ms",
                    "Tracking_error_mm", "Coverage_%", "Joint", "Correct"
                ])
                for block_name, results_list in self.results:
                    for r in results_list:
                        writer.writerow([
                            block_name,
                            r.get("trial", ""),
                            r.get("latency_ms", ""),
                            r.get("movement_ms", ""),
                            r.get("total_rt_ms", ""),
                            r.get("accuracy_mm", ""),
                            r.get("displacement_mm", ""),
                            r.get("timing_error_ms", ""),
                            r.get("tracking_error_mm", ""),
                            r.get("coverage_%", ""),
                            r.get("joint_used", ""),
                            "Yes" if r.get("correct", r.get("success", False)) else "No"
                        ])
            QMessageBox.information(self, "Success", f"Saved: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{e}")

    def closeEvent(self, e):
        self.video_thread.stop()
        e.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())