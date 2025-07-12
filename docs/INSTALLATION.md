# 🚀 Установка проекта

Добро пожаловать! Ниже — пошаговая инструкция по установке проекта. Она подойдёт для Linux, macOS и Windows.

---

## 📋 Предварительные требования

Убедитесь, что на вашем компьютере установлены:

- [Python](https://www.python.org/downloads/) 3.9–3.12
- [Git](https://git-scm.com/)
- [Pandoc](https://pandoc.org/) (для конвертации Markdown → PDF/HTML)
- [Jupyter Notebook](https://jupyter.org/)
- [TeX Live / XeLaTeX](https://www.tug.org/xetex/) (опционально — для экспорта в PDF)
- pip или pipenv / poetry

Также потребуется Python-библиотеки:

- `pandas`, `numpy`
- `scikit-learn`, `catboost`, `xgboost` (если используется)
- `fastapi`, `uvicorn` (если есть API-часть)

---

## 🧰 Установка

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/ruslanprashchurovich/NextBestChoice.git
cd NextBestChoice
```

---

### 2. Создайте и активируйте виртуальное окружение

```bash
# Для Linux/macOS
python -m venv venv
source venv/bin/activate

# Для Windows
python -m venv venv
venv\Scripts\activate
```

---

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

Если используете poetry:

```bash
poetry install
```

---

### 4. Настройка окружения (опционально)

Если проект использует переменные окружения:

- создайте `.env` файл в корне проекта
- укажите переменные, например:

```dotenv
API_KEY=ваш_ключ
DEBUG=True
```

---

### 5. Проверка установки

Убедитесь, что всё установлено корректно:

```bash
jupyter nbconvert --version
pandoc --version
python -m fastapi --help
```

(если установлен `xelatex` для PDF-выгрузки):

```bash
xelatex --version
```

---

## ▶️ Запуск

Если в проекте есть API:

```bash
uvicorn app.main:app --reload
```

Если Jupyter-ноутбук:

```bash
jupyter notebook
```

---

## 📦 Полезные команды

```bash
# Обновить зависимости
pip freeze > requirements.txt

# Установить dev-зависимости
pip install black isort flake8

# Форматирование кода
black .
isort .
```

---

## ❓ Возникли сложности?

Проверьте:

- Активировано ли окружение?
- Не забыли ли `pip install -r requirements.txt`?
- Установлены ли системные пакеты (pandoc, tex)?

Если всё ещё не получается — создайте Issue или напишите мне.

---

С благодарностью,
[Руслан](https://github.com/ruslanprashchurovich)
