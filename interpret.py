# utils/interpret.py — интерпретация и рекомендации
def interpret_metrics(metrics):
    insights = []
    recommendations = []

    # Скорость и стабильность
    rt = metrics.get('sr_mean_rt')
    if rt is not None:
        if rt < 400:
            insights.append("Высокая скорость простой реакции — база быстрого отклика на стимулы.")
        elif rt > 800:
            insights.append("Замедленная простая реакция — вероятна задержка на этапе сенсорной регистрации или инициации движения.")

    lat_sd = metrics.get('latency_stability')
    if lat_sd is not None and lat_sd > 150:
        insights.append("Высокая вариативность времени принятия решения — нестабильность когнитивной готовности.")

    # Асимметрия
    ma = metrics.get('motor_asymmetry')
    if ma is not None and ma > 0.3:
        dominant = "правая" if 'RIGHT_WRIST' in ['RIGHT_WRIST'] else "левая"
        insights.append(f"Выраженная моторная асимметрия в пользу {dominant} руки.")
        recommendations.append(f"Включить упражнения с ведущей ролью { 'левой' if dominant == 'правая' else 'правой' } руки.")

    # Адаптивность
    ad = metrics.get('adaptive_deficit')
    if ad is not None and ad > 0.5:
        insights.append("Низкая адаптивность к неожиданным изменениям — стратегия «реагировать-исправлять» вместо «предсказывать-выполнять».")
        recommendations.append("Тренировка предиктивного кодирования: визуальная обратная связь по ошибке предсказания (ожидалось vs произошло).")

    # Импульсивность
    ii = metrics.get('impulsivity_index')
    if ii is not None and ii > 0.3:
        insights.append("Повышенная импульсивность — принятие решений без достаточной обработки стимула.")
        recommendations.append("Увеличить долю дистракторов (до 50%), ввести «стоимость ошибки» (визуальное наказание).")

    # Интеграция
    ci = metrics.get('integration_score')
    if ci is not None:
        if ci > 0.8:
            insights.append("Высокий уровень когнитивно-моторной интеграции — баланс скорости, точности и адаптивности.")
        elif ci < 0.5:
            insights.append("Фрагментарность когнитивно-моторного профиля — дисбаланс между компонентами.")
            recommendations.append("Комплексный тренинг: сначала стабилизация простой реакции, затем — выбор в условиях неопределённости.")

    return insights, recommendations