# utils/video_overlay.py
import cv2
import numpy as np
import os

class VideoOverlay:
    def __init__(self, font_path="fonts/DejaVuSans.ttf"):
        self.font_path = font_path
        self._font_cache = {}

        # Попытка загрузить шрифт через PIL (для кириллицы)
        try:
            from PIL import Image, ImageDraw, ImageFont
            if os.path.exists(font_path):
                self._pil_font = ImageFont.truetype(font_path, 18)
                self._pil_font_small = ImageFont.truetype(font_path, 14)
                self._use_pil = True
            else:
                print(f"[WARN] Шрифт не найден: {font_path}")
                self._use_pil = False
        except Exception as e:
            print(f"[WARN] PIL недоступен или ошибка шрифта: {e}")
            self._use_pil = False

    def put_text(self, frame, text, org, color=(255, 255, 255), font_size=18):
        """Наложение текста с поддержкой кириллицы через PIL"""
        if self._use_pil:
            from PIL import Image, ImageDraw
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(img_pil)
            font = self._pil_font if font_size >= 18 else self._pil_font_small
            draw.text(org, text, fill=color[::-1], font=font)
            frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        else:
            # Fallback на OpenCV (может не поддерживать кириллицу)
            cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)

    def draw_axes(self, frame, center, angles_deg):
        """Рисует оси и угловые метки по тригонометрической окружности"""
        cx, cy = center
        h, w = frame.shape[:2]
        gray = (158, 158, 158)

        # Оси
        cv2.line(frame, (cx, 0), (cx, h), gray, 1, cv2.LINE_AA)
        cv2.line(frame, (0, cy), (w, cy), gray, 1, cv2.LINE_AA)

        # Угловые метки (только основные)
        for deg in [0, 90, 180, 270]:
            rad = np.radians(deg)
            x = int(cx + 200 * np.cos(rad))
            y = int(cy - 200 * np.sin(rad))  # минус — ось Y вниз
            self.put_text(frame, f"{deg}°", (x + 5, y - 10), gray, 14)

    def draw_stimulus(self, frame, center, angle_deg, radius_px, color_bgr=(0, 0, 255)):
        """Рисует красный круг на тригонометрической окружности"""
        cx, cy = center
        rad = np.radians(angle_deg)
        x = int(cx + radius_px * np.cos(rad))
        y = int(cy - radius_px * np.sin(rad))  # минус — ось Y вниз
        cv2.circle(frame, (x, y), 40, color_bgr, -1)
        cv2.circle(frame, (x, y), 40, (255, 255, 255), 2)
        return (x, y)

    def draw_joint(self, frame, pos, color_bgr, label=""):
        """Увеличенная точка сустава (16 px)"""
        if pos:
            x, y = pos
            cv2.circle(frame, (x, y), 16, (255, 255, 255), -1)
            cv2.circle(frame, (x, y), 14, color_bgr, -1)
            if label:
                self.put_text(frame, label, (x + 10, y - 10), color_bgr, 14)