from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine
import os
import pandas as pd
import numpy as np
from typing import List
from catboost import CatBoostClassifier
from datetime import datetime
from loguru import logger
import hashlib
from dotenv import load_dotenv
from schema import PostGet


# Новый класс для возвращения ответов
class Response(BaseModel):
    exp_group: str
    recommendations: List[PostGet]


# Путь к файлу .env относительно текущего файла
dotenv_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "notebooks", ".env"
)

# Загружаем переменные окружения из .env файла
load_dotenv(dotenv_path)

# Инициализация FastAPI и SQLAlchemy
app = FastAPI(title="FastAPI")

# Определяем соль нашей рекомендации и порог
salt = "reco_salt"
threshold = 50

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


def get_model_path(model_version: str) -> str:
    """
    Возвращает путь к нужной модели.

    model_version может быть 'control' или 'test'
    """
    if model_version not in ["control", "test"]:
        raise ValueError("model_version должен быть 'control' или 'test'")

    model_name = f"model_{model_version}"

    if os.environ.get("IS_LMS") == "1":
        return f"/workdir/user_input/{model_name}"
    else:
        # Здесь можно указать локальные имена файлов для тестирования
        local_map = {
            "control": "catboost_model_base",  # локальная версия model_control
            "test": "catboost_model_advanced",  # локальная версия model_test
        }
        return local_map[model_version]


# Загрузка признаков из базы данных
def load_features():
    try:
        logger.info("Loading liked posts")
        liked_posts_query = """
            SELECT DISTINCT post_id, user_id
            FROM public.feed_data
            WHERE action = 'like'"""
        liked_posts = batch_load_sql(liked_posts_query)

        logger.info("Loading posts features for the first model")
        posts_features_control = pd.read_sql(
            """SELECT * FROM posts_info_features_ruslan_prashchurovich""", con=engine
        )

        logger.info("Loading posts features for the second model")
        posts_features_test = pd.read_sql(
            """SELECT * FROM posts_info_deep_features_ruslan_prashchurovich""",
            con=engine,
        )

        logger.info("Loading users features for the first model")
        user_features_control = pd.read_sql(
            """SELECT * FROM users_info_features_ruslan_prashchurovich""", con=engine
        )

        logger.info("Loading users features for the second model")
        user_features_test = pd.read_sql(
            """SELECT * FROM public.user_data""", con=engine
        )

        return [
            liked_posts,
            posts_features_control,
            user_features_control,
            posts_features_test,
            user_features_test,
        ]
    except Exception as e:
        logger.error(f"Error while loading features: {e}")
        raise


# Загрузка модели CatBoost
def load_models():
    try:
        control_model_path = get_model_path("control")
        test_model_path = get_model_path("test")

        model_control = CatBoostClassifier()
        model_test = CatBoostClassifier()

        model_control.load_model(control_model_path)
        logger.info("Control model loaded successfully")
        model_test.load_model(test_model_path)
        logger.info("Test model loaded successfully")

        return [model_control, model_test]

    except Exception as e:
        logger.error(f"Error while loading models: {e}")
        raise


# Глобальная инициализация модели и признаков
try:
    logger.info("Loading model")
    models = load_models()
    logger.info("Loading features")
    features = load_features()
except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise


def get_exp_group(user_id: int) -> str:
    group = int(hashlib.md5((str(user_id) + salt).encode()).hexdigest(), 16) % 100
    return "control" if group <= threshold else "test"


# Функция для получения рекомендаций контрольной версии
def get_recommended_feed_control(id: int, time: datetime, limit: int):
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

        # Поменяем порядок признаков
        model = models[0]
        order = model.feature_names_
        user_post_features = user_post_features[order]

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
        return [recommended_posts, content]

    except Exception as e:
        logger.error(f"Error in get_recommended_feed: {e}")
        raise


# Функция для получения рекомендаций тестовой модели
def get_recommended_feed_test(id: int, time: datetime, limit: int):
    try:
        # Проверка входных данных
        if limit <= 0:
            raise ValueError("Limit must be greater than zero")

        # Извлечение признаков пользователя
        user_features = features[4].loc[features[4].user_id == id]
        if user_features.empty:
            raise ValueError(f"No features found for user with ID {id}")

        user_features = user_features.drop(columns=["user_id"], axis=1)

        # Извлечение признаков постов
        posts_features = features[3].drop(columns=["text"], axis=1)
        content = features[3][["post_id", "text", "topic"]]

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

        # Поменяем порядок признаков
        model = models[1]
        order = model.feature_names_
        user_post_features = user_post_features[order]

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
        return [recommended_posts, content]

    except Exception as e:
        logger.error(f"Error in get_recommended_feed: {e}")
        raise


# Обработчик FastAPI
@app.get("/post/recommendations/", response_model=Response)
async def recommended_posts(id: int, time: datetime, limit: int = 10) -> Response:
    try:
        exp_group = get_exp_group(user_id=id)
        logger.info(f"We use {exp_group} model!")
        if exp_group == "control":
            recommended_posts, content = get_recommended_feed_control(id, time, limit)
        elif exp_group == "test":
            recommended_posts, content = get_recommended_feed_test(id, time, limit)

        # Создаем список рекомендованных постов
        recommendations = [
            PostGet(
                id=i,
                text=content[content.post_id == i].text.values[0],
                topic=content[content.post_id == i].topic.values[0],
            )
            for i in recommended_posts
        ]

        # Возвращаем ответ в соответствии с моделью Response
        return Response(exp_group=exp_group, recommendations=recommendations)

    except Exception as e:
        logger.error(f"Error in /post/recommendations/: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
