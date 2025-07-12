# 📘 Инструкция по использованию проекта

Этот файл поможет вам разобраться, как использовать проект на практике — от анализа данных до запуска модели и API.

---

## 📊 1. Анализ данных

Перейдите в ноутбук [`notebooks/feature-engineering.ipynb`](https://github.com/ruslanprashchurovich/NextBestChoice/blob/master/RecSysPro/notebooks/feature-engineering_base.ipynb), чтобы провести EDA (исследовательский анализ данных). Внутри вы найдете:

- 📌 Описание и интерпретацию признаков
- 📈 Визуализацию распределений
- 🧼 Анализ пропущенных значений и выбросов
- 🔁 Преобразования для дальнейшего обучения

📎 Рекомендуется запускать ячейки по порядку в интерактивной среде (`Jupyter Notebook` или `VS Code + Jupyter`).

---

## 🧠 2. Обучение модели

Запустите ноутбук [`notebooks/end-to_end.ipynb`](https://github.com/ruslanprashchurovich/NextBestChoice/blob/master/RecSysPro/notebooks/end-to_end_ml_base_pipeline.ipynb), который содержит полный ML-пайплайн:

1. **Предобработка**: очистка данных, заполнение пропусков, one-hot/label encoding
2. **Отбор признаков**: на основе важности, корреляции или SHAP
3. **Обучение модели**: пример — `CatBoostClassifier`, `XGBoost`
4. **Оценка качества**: метрики (`ROC-AUC`, `F1`, `Precision-Recall`), графики
5. **Сохранение модели**: модель сохраняется в формате `.cbm` или `.pkl`

> 💾 Итоговая модель будет сохранена в директории `models/`

---

## 🌐 3. Использование API (FastAPI)

Модель можно использовать через REST API. Для запуска:

### Шаг 1: Убедитесь, что модель сохранена

По умолчанию путь к модели — `models/model.cbm`  
Если у вас другой файл — обновите путь в `src/api.py`

### Шаг 2: Запустите API-сервер

```bash
uvicorn src.api:app --reload
```
