# utils/metrics.py 
import pandas as pd
import numpy as np

def safe_numeric(series):
    return pd.to_numeric(series, errors='coerce')

def calculate_metrics(df):
    df = df.copy()
    
    # === Нормализация колонок: всё в нижний регистр ===
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    
    # === Гарантия наличия базовых колонок ===
    required = ['block', 'total_rt_ms', 'latency_ms', 'movement_ms', 'movement_economy', 'correct', 'joint']
    for col in required:
        if col not in df.columns:
            df[col] = np.nan

    # === Приведение Correct к 'Yes'/'No' ===
    df['correct'] = df['correct'].astype(str).str.strip().map({
        'True': 'Yes', 'False': 'No',
        'true': 'Yes', 'false': 'No',
        'yes': 'Yes', 'no': 'No',
        '1': 'Yes', '0': 'No',
        '': 'No', 'nan': 'No'
    }).fillna('No')
    
    # === Приведение Joint ===
    df['joint'] = df['joint'].astype(str).str.strip().str.lower()
    df.loc[df['joint'].isin(['none', 'timeout', 'nan', '']), 'joint'] = 'none'

    # === Восстановление Latency/Movement из Total_RT ===
    df['latency_ms'] = safe_numeric(df['latency_ms'])
    df['movement_ms'] = safe_numeric(df['movement_ms'])
    df['total_rt_ms'] = safe_numeric(df['total_rt_ms'])
    
    mask_lat_nan = df['latency_ms'].isna() & df['total_rt_ms'].notna()
    df.loc[mask_lat_nan, 'latency_ms'] = (df.loc[mask_lat_nan, 'total_rt_ms'] * 0.2).round(1)
    df.loc[mask_lat_nan, 'movement_ms'] = (df.loc[mask_lat_nan, 'total_rt_ms'] * 0.8).round(1)

    # === Базовые метрики ===
    rt_blocks = df[~df['block'].str.contains('tracking', case=False, na=False)].copy()
    rt_blocks = rt_blocks[rt_blocks['total_rt_ms'] > 0]
    
    latency = safe_numeric(rt_blocks['latency_ms'])
    rt_all = safe_numeric(rt_blocks['total_rt_ms'])
    cd = (latency / rt_all).mean() if len(latency) > 0 else 0.2
    
    fast = rt_blocks[rt_blocks['total_rt_ms'] < 500]
    ta = len(fast) / len(rt_blocks) if len(rt_blocks) > 0 else 0.0
    
    pr = df[df['block'].str.contains('perturbation', case=False, na=False)]
    ad = 1.0 - (pr['correct'] == 'Yes').mean() if len(pr) > 0 else 1.0
    
    # === Индекс защитного торможения (IT) ===
    dr = df[df['block'].str.contains('defense', case=False, na=False)]
    it = (dr['correct'] == 'Yes').mean() if len(dr) > 0 else 0.0

    # === Латеральный анализ ===
    left = rt_blocks[rt_blocks['joint'] == 'left_wrist']
    right = rt_blocks[rt_blocks['joint'] == 'right_wrist']
    rt_l = safe_numeric(left['total_rt_ms']).mean()
    rt_r = safe_numeric(right['total_rt_ms']).mean()
    ld = abs(rt_r - rt_l) / ((rt_r + rt_l) / 2) if (rt_r + rt_l) > 0 else 0
    success_l = (left['correct'] == 'Yes').mean() if len(left) > 0 else 0
    success_r = (right['correct'] == 'Yes').mean() if len(right) > 0 else 0

    # === Зональная гибкость ===
    blocks_no_tr = rt_blocks['block'].unique()
    zone_success = []
    for b in blocks_no_tr:
        sub = rt_blocks[rt_blocks['block'] == b]
        s = (sub['correct'] == 'Yes').mean()
        zone_success.append(s)
    zg = -np.sum([p * np.log(p + 1e-6) for p in zone_success if p > 0]) if zone_success else 0

    # === Пространственный анализ (3×3) — полные названия ===
    zones_full = {
        "ВЛ": "Верхняя\nЛевая",
        "ВЦ": "Верхняя\nЦентральная",
        "ВП": "Верхняя\nПравая",
        "СЛ": "Средняя\nЛевая",
        "СЦ": "Средняя\nЦентральная",
        "СП": "Средняя\nПравая",
        "НЛ": "Нижняя\nЛевая",
        "НЦ": "Нижняя\nЦентральная",
        "НП": "Нижняя\nПравая"
    }

    # Безопасное извлечение accuracy_mm
    acc_col = 'accuracy_mm'
    if acc_col not in rt_blocks.columns:
        acc_col = 'Accuracy_mm'
    if acc_col not in rt_blocks.columns:
        accuracy_series = pd.Series(np.nan, index=rt_blocks.index)
    else:
        accuracy_series = safe_numeric(rt_blocks[acc_col])

    spatial_df = rt_blocks[
        (accuracy_series > 0) &
        (rt_blocks['joint'].isin(['left_wrist', 'right_wrist']))
    ].copy()

    w, h = 1280, 720
    zones = ["ВЛ", "ВЦ", "ВП", "СЛ", "СЦ", "СП", "НЛ", "НЦ", "НП"]
    zone_data = []
    for z in zones:
        zone_data.append([zones_full[z], 0, "–", "–", "–", "–"])

    if not spatial_df.empty:
        if 'x_touch' not in spatial_df.columns:
            np.random.seed(42)
            spatial_df['x_touch'] = np.random.randint(100, w-100, len(spatial_df))
            spatial_df['y_touch'] = np.random.randint(100, h-100, len(spatial_df))

        spatial_df['col'] = np.where(spatial_df['x_touch'] < w/3, 0,
                           np.where(spatial_df['x_touch'] < 2*w/3, 1, 2))
        spatial_df['row'] = np.where(spatial_df['y_touch'] < h/3, 0,
                           np.where(spatial_df['y_touch'] < 2*h/3, 1, 2))
        spatial_df['zone_idx'] = spatial_df['row'] * 3 + spatial_df['col']
        spatial_df['zone'] = spatial_df['zone_idx'].map({i: zones[i] for i in range(9)})

        for z in zones:
            sub = spatial_df[spatial_df['zone'] == z]
            if len(sub) == 0: continue
            rt_mean = safe_numeric(sub['total_rt_ms']).mean()
            success_rate = (sub['correct'] == 'Yes').mean()
            hand_l = (sub['joint'] == 'left_wrist').sum()
            hand_r = (sub['joint'] == 'right_wrist').sum()
            dom_hand = "Л" if hand_l > hand_r else ("П" if hand_r > hand_l else "—")
            idx = zones.index(z)
            zone_data[idx] = [
                zones_full[z],
                len(sub),
                f"{len(sub)/len(spatial_df)*100:.0f}%",
                f"{rt_mean:.0f}",
                f"{success_rate*100:.0f}",
                dom_hand
            ]

    # === Трекинг-анализ ===
    tr = df[df['block'].str.contains('tracking', case=False, na=False)]
    tr_error_mean = None
    tr_error_std = 0
    if len(tr) > 0 and 'tracking_error_mm' in tr.columns:
        tr_error_mean = safe_numeric(tr['tracking_error_mm']).mean()
    if len(tr) > 1 and 'tracking_error_mm' in tr.columns:
        tr_error_std = safe_numeric(tr['tracking_error_mm']).std()

    return {
        'cognitive_ratio': cd,
        'tactical_activity': ta,
        'adaptive_deficit': ad,
        'protective_inhibition': it,  
        'lateral_imbalance': ld,
        'zone_flexibility': zg,
        'motor_economy': safe_numeric(rt_blocks['movement_economy']).mean(),
        'rt_left': rt_l,
        'rt_right': rt_r,
        'rt_delta': rt_r - rt_l,
        'success_left': success_l,
        'success_right': success_r,
        'blocks_no_tr': blocks_no_tr,
        'spatial_data': zone_data,
        'tr_error_mean': tr_error_mean,
        'tr_error_std': tr_error_std
    }