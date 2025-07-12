# Проект: Разработка и внедрение рекомендательной системы постов с A/B-тестированием

**Роль:** Data Scientist (end-to-end: от проектирования до продакшена)

## 📌 Саммари

Построил двухэтапную рекомендательную систему для прогнозирования вовлеченности пользователей:

1. **Базовое решение**: Классический ML-пайплайн (TF-IDF + кластеризация) для рекомендаций постов
2. **Продвинутое решение**: Deep Learning-модель на основе BERT для генерации контекстных эмбеддингов
3. **Валидация**: Провёл A/B-тест, подтвердивший статистическую значимость улучшений продвинутого решения
4. **Продакшенизация**: Развернул API на FastAPI для интеграции с основным сервисом

## ⚙️ Технический стек

### 🧾 Языки

![Python](https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white&style=flat)

### 🤖 ML / DL

![Scikit-learn](https://img.shields.io/badge/-Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white&style=flat)
![CatBoost](https://img.shields.io/badge/-CatBoost-EE9D00?style=flat)
![XGBoost](https://img.shields.io/badge/-XGBoost-1A5D78?style=flat)
![PyTorch](https://img.shields.io/badge/-PyTorch-EE4C2C?logo=pytorch&logoColor=white&style=flat)

### 📚 NLP

![TF-IDF](https://img.shields.io/badge/-TF--IDF-blue?style=flat)
![BERT](https://img.shields.io/badge/-BERT-0081A7?style=flat)
![HuggingFace](https://img.shields.io/badge/-Transformers-FFBF00?logo=huggingface&logoColor=black&style=flat)

### 🖥️ Бэкенд

![FastAPI](https://img.shields.io/badge/-FastAPI-009688?logo=fastapi&logoColor=white&style=flat)
![Pydantic](https://img.shields.io/badge/-Pydantic-0F172A?style=flat)
![SQLAlchemy](https://img.shields.io/badge/-SQLAlchemy-8C1C13?style=flat)
![Uvicorn](https://img.shields.io/badge/-Uvicorn-4B8BBE?style=flat)

### 💾 Данные и хранилище

![PostgreSQL](https://img.shields.io/badge/-PostgreSQL-4169E1?logo=postgresql&logoColor=white&style=flat)
![Pandas](https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white&style=flat)

### 🧪 Тестирование / статистика

![SciPy](https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white&style=flat)
![A/B-тест](https://img.shields.io/badge/-A%2FB%20Test-blueviolet?style=flat)

## 🎯 Задачи и решения

| Этап                       | Действия                                                                                                 | Инструменты                    |
| -------------------------- | -------------------------------------------------------------------------------------------------------- | ------------------------------ |
| **1. Подготовка данных**   | • Подключение к БД через SQLAlchemy<br>• Выгрузка ТОП-лайкнутых постов<br>• Препроцессинг текста         | PostgreSQL, Pandas, NLTK       |
| **2. Feature Engineering** | • Создание фичей: длина текста, тональность, тематики<br>• Векторизация TF-IDF<br>• Отбор фичей          | Sklearn                        |
| **3. Базовая модель**      | • TF-IDF + SVD + Кластеризация постов (KMeans)<br>• Модель классификации CatBoost <br>• Оценка HitRate@5 | Sklearn, SVD                   |
| **4. DL-модель**           | • Генерация эмбеддингов DistilBERT<br>• Оптимизация кластеров (KMeans)<br>• Построение рекомендаций      | PyTorch, sentence-transformers |
| **5. A/B-тест**            | • Разделение трафика 50/50<br>• Сравнение числа лайков на пользователя<br>• Проверка гипотез             | SciPy, Mann-Whitney U-test     |
| **6. Продакшн**            | • Развертывание API (FastAPI)                                                                            | FastAPI                        |

## 🏆 Достижения и результаты

**Базовое решение (TF-IDF + SVD + KMeans):**

- Метрика качества: **HitRate@5 = 0.56**

**Улучшенное решение (BERT + SVD + KMeans):**

- Метрика качества: HitRate@5 = 0.58 (+3.5% к точности)

**Результаты A/B-теста:**

- Метрика: Среднее количество лайков на пользователя
- Статтест: Mann-Whitney U-test (p-value < 0.01)
- Вывод: Модель на основе BERT показала статистически значимое улучшение
- Группа B: +5.2% лайков/пользователя
