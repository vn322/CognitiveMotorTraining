# report_generator.py  
import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

from utils.metrics import calculate_metrics
from utils.style import create_table_style

def load_font():
    try:
        font_path = 'DejaVuSans.ttf'
        if not os.path.exists(font_path):
            import sys
            if hasattr(sys, '_MEIPASS'):
                font_path = os.path.join(sys._MEIPASS, 'DejaVuSans.ttf')
        pdfmetrics.registerFont(TTFont('DejaVuSans', font_path))
        return 'DejaVuSans'
    except:
        return 'Helvetica'

def safe_text(s):
    if pd.isna(s) or s is None:
        return ""
    return str(s).replace('’', "'").replace('–', '-')

def generate_pdf_report(csv_path) -> str:
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # ✅ КЛЮЧЕВОЕ: нормализация имён колонок ДО всего остального
        df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
        if 'block' not in df.columns:
            raise ValueError(f"Колонка 'block' отсутствует. Доступные: {list(df.columns)}")

        metrics = calculate_metrics(df)
        FONT_NAME = load_font()
        pdf_path = csv_path.replace('.csv', '_analysis.pdf')

        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.6*72, bottomMargin=0.5*72)
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', fontName=FONT_NAME, fontSize=18, alignment=1, spaceAfter=12)
        h1_style = ParagraphStyle('H1', fontName=FONT_NAME, fontSize=14, spaceAfter=8)
        normal_style = ParagraphStyle('Normal', fontName=FONT_NAME, fontSize=10, spaceAfter=6, leading=14)
        small_style = ParagraphStyle('Small', fontName=FONT_NAME, fontSize=9, spaceAfter=4)

        story = []
        story.append(Paragraph("Когнитивно-моторный профиль", title_style))
        story.append(Paragraph("<i>Отчёт собран автоматически. Требует осмысления и верификации экспертом.</i>", small_style))
        story.append(Spacer(1, 10))

        # 1. Ключевые расчётные метрики 
        story.append(Paragraph("1. Ключевые расчётные метрики", h1_style))
        metrics_data = [
            ["Метрика", "Значение", "Норма", "Интерпретация"],
            ["Когнитивная\nдоля (CD)", f"{metrics['cognitive_ratio']:.2f}", "0.20–0.35", "Доля времени на\nпринятие решения"],
            ["Тактическая\nактивность (TA)", f"{metrics['tactical_activity']:.2f}", "0.3–0.6", "Доля быстрых\nрешений (RT < 500 мс)"],
            ["Адаптивный\nдефицит (AD)", f"{metrics['adaptive_deficit']:.2f}", "<0.30", "Неспособность\nадаптироваться к\nнеожиданностям"],
            ["Индекс\nизбегания (IА)", f"{metrics['protective_inhibition']:.2f}", ">0.70", "Способность\nизбегать\nвнезапных помех"],
            ["Моторная\nэкономичность (ME)", f"{metrics['motor_economy']:.2f}", ">0.85", "Эффективность\nдвижения"],
            ["Латеральный\nдисбаланс (LD)", f"{metrics['lateral_imbalance']:.2f}", "<0.20", "Асимметрия\nлевой и правой руки"],
            ["Тактическая\nгибкость (ТG)", f"{metrics['zone_flexibility']:.2f}", "1.5–2.5", "Разнообразие\nуспешных стратегий"]
        ]
        table = Table(metrics_data, colWidths=[120, 60, 70, 180], rowHeights=[24] + [36]*7)
        table.setStyle(create_table_style())
        story.append(table)
        story.append(Spacer(1, 12))

        # 2. Латеральный профиль
        story.append(Paragraph("2. Латеральный профиль (левая vs правая рука)", h1_style))
        lateral_data = [
            ["Параметр", "Левая рука", "Правая рука", "Дельта"],
            ["Ср. RT, мс", f"{metrics['rt_left']:.0f}" if not np.isnan(metrics['rt_left']) else "–", f"{metrics['rt_right']:.0f}" if not np.isnan(metrics['rt_right']) else "–", f"{metrics['rt_delta']:+.0f}" if not (np.isnan(metrics['rt_left']) or np.isnan(metrics['rt_right'])) else "–"],
            ["Успех, %", f"{metrics['success_left']*100:.0f}" if metrics['success_left'] > 0 else "–", f"{metrics['success_right']*100:.0f}" if metrics['success_right'] > 0 else "–", f"{(metrics['success_right'] - metrics['success_left'])*100:+.0f}" if not (np.isnan(metrics['success_left']) or np.isnan(metrics['success_right'])) else "–"]
        ]
        table = Table(lateral_data, colWidths=[100, 70, 70, 70])
        table.setStyle(create_table_style())
        story.append(table)
        story.append(Spacer(1, 8))
        if metrics['lateral_imbalance'] > 0.3:
            story.append(Paragraph(f"• Выраженный латеральный дисбаланс ({metrics['lateral_imbalance']:.2f}): предпочтение { 'правой' if metrics['rt_right'] < metrics['rt_left'] else 'левой' } руке.", normal_style))
        else:
            story.append(Paragraph("• Сбалансированное межполушарное взаимодействие.", normal_style))
        story.append(Spacer(1, 12))

        # 3. Активность в различных условиях (тестах) 
        story.append(Paragraph("3. Активность в различных условиях (тестах)", h1_style))
        zone_data_simple = [["Условие", "Попыток", "Успех, %", "Ср. RT, мс", "CD"]]
        for b in sorted(metrics['blocks_no_tr']):
            sub = df[df['block'] == b]
            total = len(sub)
            success = (sub['correct'] == 'Yes').sum()
            rt_b = sub['total_rt_ms'].mean()
            
            # ✅ Безопасное извлечение latency_ms с fallback
            latency_b = 0.2 * rt_b  # fallback: 20% latency, 80% movement
            if 'latency_ms' in sub.columns:
                lat_series = pd.to_numeric(sub['latency_ms'], errors='coerce')
                if lat_series.notna().any():
                    latency_b = lat_series.mean()
            cd_b = (latency_b / rt_b) if rt_b > 0 else 0.2
            
            # ✅ Перенос для комбинированных реакций
            display_name = b
            if "selection + tracking" in b.lower():
                display_name = "Combined Reaction:\nSelection + Tracking"
            elif "dynamic switch" in b.lower():
                display_name = "Combined Reaction:\nDynamic Switch"
            zone_data_simple.append([
                display_name,
                str(total),
                f"{success/total*100:.0f}" if total > 0 else "–",
                f"{rt_b:.0f}" if not np.isnan(rt_b) else "–",
                f"{cd_b:.2f}"
            ])
        table = Table(zone_data_simple, colWidths=[160, 50, 60, 70, 60], rowHeights=[24] + [30]*len(zone_data_simple[1:]))
        table.setStyle(create_table_style())
        story.append(table)
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"• Зональная гибкость: {metrics['zone_flexibility']:.2f} → {'оптимальная' if 1.5 <= metrics['zone_flexibility'] <= 2.5 else 'низкая' if metrics['zone_flexibility'] < 1.5 else 'избыточная'}.", normal_style))
        story.append(Spacer(1, 12))

        # 4. Пространственное распределение активности (3×3) 
        story.append(Paragraph("4. Пространственное распределение активности (3×3)", h1_style))
        spatial_table = [["Зона", "Попыток", "% от общ.", "Ср. RT, мс", "Успех, %", "Домин.\nрука"]]
        for row in metrics['spatial_data']:
            spatial_table.append(row)
        table = Table(spatial_table, colWidths=[90, 50, 60, 70, 60, 50], rowHeights=[24] + [36]*len(metrics['spatial_data']))
        table.setStyle(create_table_style())
        story.append(table)
        story.append(Spacer(1, 8))
        if any("Верхняя" in z[0] for z in metrics['spatial_data']):
            story.append(Paragraph("• Верхняя зона: более высокая RT — сложность координации при подъёме руки.", normal_style))
        story.append(Spacer(1, 12))

        # 5. Анализ слежения
        if metrics['tr_error_mean'] is not None:
            story.append(Paragraph("5. Анализ слежения", h1_style))
            story.append(Paragraph(f"• Средняя ошибка слежения: {metrics['tr_error_mean']:.1f} мм.", normal_style))
            if metrics['tr_error_std'] > 30:
                story.append(Paragraph(f"• Высокая вариативность (σ = {metrics['tr_error_std']:.1f} мм) — нестабильность траектории.", normal_style))
            elif metrics['tr_error_mean'] > 80:
                story.append(Paragraph("• Системное отставание: кривая пользователя «сдвинута» относительно модели.", normal_style))
            else:
                story.append(Paragraph("• Удовлетворительное совпадение с траекторией.", normal_style))
            story.append(Spacer(1, 12))

        # 6. Рекомендации
        story.append(Paragraph("6. Тактический диагноз и рекомендации", h1_style))
        diag = []
        if metrics['adaptive_deficit'] > 0.7:
            diag.append("• Низкая адаптивность: отсутствие предиктивного анализа ситуации.")
        if metrics['lateral_imbalance'] > 0.3:
            diag.append(f"• Ярко выраженная латеральность: доминирование { 'правой' if metrics['rt_right'] < metrics['rt_left'] else 'левой' } руки.")
        if metrics['protective_inhibition'] < 0.5:
            diag.append("• Низкий уровень торможения: трудности с подавлением импульсивных реакций при помехах.")
        if metrics['tr_error_mean'] and metrics['tr_error_mean'] > 80:
            diag.append("• Слабая способность к плавному слежению за динамическим объектом.")
        if not diag:
            diag.append("• Сбалансированная тактика.")

        for d in diag:
            story.append(Paragraph(d, normal_style))

        story.append(Spacer(1, 8))

        recs = []
        if metrics['adaptive_deficit'] > 0.7:
            recs.append("→ Добавить упражнений в условиях требующих адаптивности».")
        if metrics['lateral_imbalance'] > 0.3:
            weaker = "левой" if metrics['rt_left'] > metrics['rt_right'] else "правой"
            recs.append(f"→ Увеличить долю задач с {weaker} рукой до 60%.")
        if metrics['protective_inhibition'] < 0.5:
            recs.append("→ Ввести тренировку «ложные старты».")
        if metrics['tr_error_mean'] and metrics['tr_error_mean'] > 80:
            recs.append("→ Тренировку слежения за движущимися объектами лучше начинать с медленных движений, постепенно усложняя задачи.")

        if recs:
            story.append(Paragraph("Рекомендации:", h1_style))
            for r in recs:
                story.append(Paragraph(f"• {r}", normal_style))

        story.append(Spacer(1, 20))
        story.append(Paragraph("Всегда сверяйтесь с данными из CSV отчёта", ParagraphStyle('Footer', fontSize=9, alignment=1, fontName=FONT_NAME)))

        doc.build(story)
        return pdf_path

    except Exception as e:
        import traceback
        print("=== ОШИБКА ГЕНЕРАЦИИ PDF ===")
        traceback.print_exc()
        print("==============================")
        return None