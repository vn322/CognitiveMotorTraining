# utils/angle_utils.py
import math

def get_torso_center(landmarks):
    """
    Середина туловища = середина между левой и правой линиями SHOULDER → HIP
    """
    ls, lh = landmarks.get('LEFT_SHOULDER'), landmarks.get('LEFT_HIP')
    rs, rh = landmarks.get('RIGHT_SHOULDER'), landmarks.get('RIGHT_HIP')
    
    left_mid = ((ls[0] + lh[0]) // 2, (ls[1] + lh[1]) // 2) if ls and lh else None
    right_mid = ((rs[0] + rh[0]) // 2, (rs[1] + rh[1]) // 2) if rs and rh else None

    if left_mid and right_mid:
        return ((left_mid[0] + right_mid[0]) // 2, (left_mid[1] + right_mid[1]) // 2)
    return left_mid or right_mid or (320, 240)

def stimulus_position(center, radius, angle_deg):
    """
    Позиция стимула на окружности. 0° = вправо, 90° = вверх.
    """
    rad = math.radians(angle_deg)
    x = int(center[0] + radius * math.cos(rad))
    y = int(center[1] - radius * math.sin(rad))  # инверсия Y
    return (x, y)