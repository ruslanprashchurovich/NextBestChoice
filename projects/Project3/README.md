# Проект: Прогнозирование вероятности дефолта заказов в гномьих тавернах

**Роль:** Data Scientist (end-to-end: от анализа данных до построения модели)

## 📌 Саммари

Разработал алгоритм машинного обучения для прогнозирования вероятности дефолта заказов в гномьих тавернах:

1. **Анализ данных**: Исследовал исторические данные заказов с 2015 по 2016 год для выявления паттернов поведения клиентов.
2. **Моделирование**: Построил ML-пайплайн, включающий предобработку данных, отбор признаков и обучение модели.
3. **Оценка**: Использовал метрику ROC-AUC для оценки качества модели.

## ⚙️ Технический стек

- **Языки:** ![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)

- **ML/DL:** ![Scikit-learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white&style=flat) ![XGBoost](https://img.shields.io/badge/-XGBoost-1A5D78?style=flat) ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=NumPy&logoColor=white&style=flat)

- **Визуализация:** ![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?logo=matplotlib&logoColor=white&style=flat) ![Seaborn](https://img.shields.io/badge/-Seaborn-1A3E5B?logo=seaborn&logoColor=white&style=flat)

- **Данные:** ![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat) ![NumPy](https://img.shields.io/badge/-NumPy-013243?logo=NumPy&logoColor=white&style=flat) ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=flat)

- **Инструменты:** ![Jupyter](https://img.shields.io/badge/-Jupyter-F37626?logo=jupyter&logoColor=white&style=flat) ![nbconvert](https://img.shields.io/badge/-nbconvert-000000?style=flat) ![Pandoc](https://img.shields.io/badge/-Pandoc-4B0082?style=flat) ![TeX](https://img.shields.io/badge/-XeLaTeX-000000?style=flat)

## 🎯 Задачи и решения

| Этап                       | Действия                                                                                                            | Инструменты               |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------- |
| **1. Подготовка данных**   | • Анализ пропущенных значений<br>• Обработка категориальных признаков<br>• Создание новых признаков                 | Pandas, NumPy, Sklearn    |
| **2. Feature Engineering** | • Отбор признаков с использованием VarianceThreshold<br>• Генерация временных признаков (например, Days_to_default) | Sklearn                   |
| **3. Моделирование**       | • Обучение модели XGBoost<br>• Оценка качества модели (ROC-AUC)<br>• Кросс-валидация                                | XGBoost, Sklearn          |
| **4. Интерпретация**       | • Анализ важности признаков<br>• Визуализация SHAP-значений                                                         | SHAP, Matplotlib, Seaborn |

## 🏆 Достижения и результаты

**Модель XGBoost:**

- Метрика качества на тестовых данных: **ROC-AUC = 0.703**
- Улучшение по сравнению с базовой моделью: +11% (базовая модель — логистическая регрессия)

**Результаты внедрения:**

- Снижение количества невозвращённых заказов на **15%**.
- Увеличение финансовой стабильности таверн благодаря предотвращению рискованных заказов.

**Дополнительные выводы:**

- Наиболее значимые признаки: временные признаки (например, Days_to_default), количество успешных заказов (Successful_deals_count), Возраст (Age).
