from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from datetime import datetime
from loguru import logger
import numpy as np
from dotenv import load_dotenv
from schema import PostGet


# Путь к файлу .env относительно текущего файла
dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "notebooks", ".env"
)

# Загружаем переменные окружения из .env файла
load_dotenv(dotenv_path)

# Инициализация FastAPI и SQLAlchemy
app = FastAPI(title="FastAPI")

# Создаем URL для SQLAlchemy
SQLALCHEMY_DATABASE_URL = (
    f"postgresql://"
    f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}/"
    f"{os.getenv('POSTGRES_DATABASE')}"
)

try:
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
except Exception as e:
    logger.error(f"Failed to connect to the database: {e}")
    raise


# Функция для загрузки SQL-запросов по частям
def batch_load_sql(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    try:
        with engine.connect().execution_options(stream_results=True) as conn:
            chunks = []
            for chunk in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
                chunks.append(chunk)
                logger.info(f"Got chunk: {len(chunk)}")
            return pd.concat(chunks, ignore_index=True)
    except Exception as e:
        logger.error(f"Error while loading SQL data: {e}")
        raise


# Функция для получения пути к модели
def get_model_path(path: str) -> str:
    try:
        if os.environ.get("IS_LMS") == "1":
            MODEL_PATH = "/workdir/user_input/model"
        else:
            MODEL_PATH = path
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        return MODEL_PATH
    except Exception as e:
        logger.error(f"Error while getting model path: {e}")
        raise


# Загрузка признаков из базы данных
def load_features():
    try:
        logger.info("Loading liked posts")
        liked_posts_query = """
            SELECT DISTINCT post_id, user_id
            FROM public.feed_data
            WHERE action = 'like'"""
        liked_posts = batch_load_sql(liked_posts_query)

        logger.info("Loading posts features")
        posts_features = pd.read_sql(
            """SELECT * FROM posts_info_features_ruslan_prashchurovich""", con=engine
        )

        logger.info("Loading users features")
        user_features = pd.read_sql(
            """SELECT * FROM users_info_features_ruslan_prashchurovich""", con=engine
        )

        return [liked_posts, posts_features, user_features]
    except Exception as e:
        logger.error(f"Error while loading features: {e}")
        raise


# Загрузка модели CatBoost
def load_model():
    try:
        model_path = get_model_path("catboost_model_base")
        loaded_model = CatBoostClassifier()
        loaded_model.load_model(model_path)
        logger.info("Model loaded successfully")
        return loaded_model
    except Exception as e:
        logger.error(f"Error while loading model: {e}")
        raise


# Глобальная инициализация модели и признаков
try:
    logger.info("Loading model")
    model = load_model()
    logger.info("Loading features")
    features = load_features()
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise


# Функция для получения рекомендаций
def get_recommended_feed(id: int, time: datetime, limit: int):
    try:
        # Проверка входных данных
        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        # Извлечение признаков пользователя
        user_features = features[2].loc[features[2].user_id == id]
        if user_features.empty:
            raise ValueError(f"No features found for user with ID {id}")

        user_features = user_features.drop(columns=["user_id"], axis=1)

        # Извлечение признаков постов
        posts_features = features[1].drop(columns=["text"], axis=1)
        content = features[1][["post_id", "text", "topic"]]

        # Добавление признаков пользователя к признакам постов
        add_user_features = dict(zip(user_features.columns, user_features.values[0]))
        user_post_features = posts_features.assign(**add_user_features)
        user_post_features = user_post_features.set_index("post_id")

        # Добавление временных признаков
        user_post_features["hour"] = time.hour
        user_post_features["month"] = time.month
        user_post_features["hour_sin"] = np.sin(
            2 * np.pi * user_post_features["hour"] / 24
        )
        user_post_features["hour_cos"] = np.cos(
            2 * np.pi * user_post_features["hour"] / 24
        )
        user_post_features["month_sin"] = np.sin(
            2 * np.pi * user_post_features["month"] / 12
        )
        user_post_features["month_cos"] = np.cos(
            2 * np.pi * user_post_features["month"] / 12
        )

        # Предсказание вероятностей
        predicts = model.predict_proba(user_post_features)[:, 1]
        user_post_features["predicts"] = predicts

        # Фильтрация уже лайкнутых постов
        liked_posts = features[0]
        liked_posts = liked_posts[liked_posts.user_id == id].post_id.values
        filtered_ = user_post_features[~user_post_features.index.isin(liked_posts)]

        # Сортировка и выбор топ-N постов
        recommended_posts = filtered_.sort_values("predicts", ascending=False)[
            :limit
        ].index

        # Возвращение результатов
        return [
            PostGet(
                **{
                    "id": i,
                    "text": content[content.post_id == i].text.values[0],
                    "topic": content[content.post_id == i].topic.values[0],
                }
            )
            for i in recommended_posts
        ]
    except Exception as e:
        logger.error(f"Error in get_recommended_feed: {e}")
        raise


# Обработчик FastAPI
@app.get("/post/recommendations/", response_model=List[PostGet])
async def recommended_posts(id: int, time: datetime, limit: int = 10) -> List[PostGet]:
    try:
        return get_recommended_feed(id, time, limit)
    except Exception as e:
        logger.error(f"Error in /post/recommendations/: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
