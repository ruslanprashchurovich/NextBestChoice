from sqlalchemy import create_engine
import pandas as pd
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError

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


def test_query(query, timeout=30):
    """Выполняет SQL-запрос с ограничением по времени."""

    def run_query():
        return pd.read_sql(query, con=engine)

    with ThreadPoolExecutor() as executor:
        future = executor.submit(run_query)
        try:
            df = future.result(timeout=timeout)
            print(f"✅ Query executed successfully. Rows: {len(df)}")
            return df
        except TimeoutError:
            print("❌ Query timed out (took longer than {} seconds)".format(timeout))
        except Exception as e:
            print(f"❌ Query failed: {e}")


# Тест каждого запроса отдельно
test_query("SELECT * FROM posts_info_features_ruslan_prashchurovich LIMIT 5")
test_query("SELECT * FROM users_info_features_ruslan_prashchurovich LIMIT 5")
test_query(
    "SELECT DISTINCT post_id, user_id FROM feed_data WHERE action = 'like' LIMIT 5"
)
