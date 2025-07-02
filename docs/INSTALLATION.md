# Установка проекта

## Предварительные требования

Перед установкой убедитесь, что у вас установлены следующие компоненты:

- Python (версии 3.9–3.12)
- Git
- Pandoc (для конвертации Markdown)
- scikit-learn
- Pandas
- fastapi

## Шаги установки

### 1. Клонируйте репозиторий

```bash
git clone https://github.com/ruslanprashchurovich/ruslanprashchurovich.git
cd ruslanprashchurovich
```

### 2. Создайте виртуальное окружение

```python
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 3. Установите зависимости

```bash
pip install -r requirements.txt
```

### 4. Установите дополнительные зависимости

### 5. Проверьте установку

```bash
jupyter nbconvert --version
pandoc --version
xelatex --version
```
