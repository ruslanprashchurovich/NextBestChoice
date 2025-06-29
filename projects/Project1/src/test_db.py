from sqlalchemy import create_engine
from dotenv import load_dotenv
import os


# Путь к файлу .env относительно текущего файла
dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "notebooks", ".env"
)

# Загружаем переменные окружения из .env файла
load_dotenv(dotenv_path)

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

try:
    with engine.connect() as conn:
        result = conn.execute("SELECT * FROM post_text_df LIMIT 1")
        print("Connection successful!")
except Exception as e:
    print(f"Connection failed: {e}")
