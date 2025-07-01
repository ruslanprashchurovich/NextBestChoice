from typing import List, Union
import numpy as np


def cumulative_gain(relevance: List[float], k: int) -> float:
    """Score is cumulative gain at k (CG@k)

    Parameters
    ----------
    relevance:  `List[float]`
        Relevance labels (Ranks)
    k : `int`
        Number of elements to be counted

    Returns
    -------
    score : float
    """
    if k < 0:
        raise ValueError("k must be positive")

    relevance = np.asfarray(relevance)
    cum_gain = np.sum(relevance[:k])
    return cum_gain


def discounted_cumulative_gain(
    relevance: List[float], k: int, method: str = "standard"
) -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values:
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    if method not in ("standard", "industry"):
        raise ValueError("Method must be either 'standard' or 'industry'")

    relevance = np.asfarray(relevance)

    if k > len(relevance):
        raise ValueError(
            f"k ({k}) cannot be greater than the length of relevance ({len(relevance)})"
        )

    if method == "standard":
        gain = relevance[:k]
    elif method == "industry":
        gain = np.power(2, relevance[:k]) - 1

    discounts = np.log2(
        np.arange(2, k + 2)
    )  # индексация с 2: log2(2), log2(3), ..., log2(k+1)

    dcg_k = np.sum(gain / discounts)
    return dcg_k


def normalized_dcg(relevance: List[float], k: int, method: str = "standard") -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list.
    k : `int`
        Count relevance to compute.
    method : `str`, optional
        Metric implementation method:
        `standard` - adds weight to the denominator
        `industry` - adds weights to numerator and denominator
        Default is 'standard'.

    Returns
    -------
    score : `float`
        Metric score.
    """
    if method not in ("standard", "industry"):
        raise ValueError("Method must be either 'standard' or 'industry'")

    relevance = np.asfarray(relevance)

    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if k > len(relevance):
        relevance = np.pad(
            relevance, (0, k - len(relevance)), "constant", constant_values=0
        )

    # Формируем набор релевантностей
    actual_relevance = relevance[:k]

    # Вычисляем gain для актуального списка
    if method == "standard":
        gains = actual_relevance
    elif method == "industry":
        gains = np.power(2, actual_relevance) - 1

    # Находим идеальный список релевантности (топ-k)
    sorted_rel = np.sort(relevance)[::-1]
    ideal_relevance = sorted_rel[:k]

    # Вычисляем gain для идеального списка
    if method == "standard":
        ideal_gains = ideal_relevance
    elif method == "industry":
        ideal_gains = np.power(2, ideal_relevance) - 1

    # Считаем скидки
    discounts = np.log2(np.arange(2, k + 2))  # log2(2), log2(3), ..., log2(k+1)

    # DCG и IDCG
    dcg = np.sum(gains / discounts)
    idcg = np.sum(ideal_gains / discounts)

    return dcg / idcg if idcg != 0 else 0.0


def batch_normalized_dcg(
    all_relevance: Union[List[List[float]], np.ndarray],
    k: int,
    method: str = "standard",
) -> np.ndarray:
    """
    Вычисляет nDCG для каждого списка в `all_relevance`.

    Parameters
    ----------
    all_relevance : `List[List[float]]` or `np.ndarray`
        Список списков/массивов с релевантностями
    k : `int`
        Количество элементов для оценки
    method : `str`, optional
        Метод вычисления ('standard' или 'industry')

    Returns
    -------
    scores : `np.ndarray`
        Массив с nDCG для каждого запроса
    """

    def calculate_one(rel):
        return normalized_dcg(rel, k=k, method=method)

    return np.array([calculate_one(r) for r in all_relevance])


def avg_ndcg(
    list_relevances: List[List[float]], k: int, method: str = "standard"
) -> float:
    """average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    # Проверяем, пустой ли список
    if isinstance(list_relevances, list) and len(list_relevances) == 0:
        return 0.0
    elif isinstance(list_relevances, np.ndarray) and list_relevances.size == 0:
        return 0.0

    def calculate_one(rel):
        return normalized_dcg(rel, k=k, method=method)

    scores = np.array(
        [
            calculate_one(r) if isinstance(r, (list, np.ndarray)) else 0.0
            for r in list_relevances
        ]
    )
    return float(np.mean(scores))


def hit_rate_at_k(recommended_items, relevant_items, k=5):
    """
    Вычисляет HitRate@k для рекомендательной системы

    Параметры:
    recommended_items : list of lists
        Рекомендованные айтемы для каждого пользователя (в порядке убывания релевантности)
    relevant_items : list of lists
        Релевантные (лайкнутые) айтемы для каждого пользователя
    k : int
        Количество рассматриваемых рекомендаций (топ-k)

    Возвращает:
    float : Значение HitRate@k
    """
    hits = 0
    total_users = len(recommended_items)

    for user_recs, user_relevant in zip(recommended_items, relevant_items):
        # Берем только топ-k рекомендаций
        top_k = user_recs[:k]
        # Проверяем, есть ли пересечение с релевантными айтемами
        if set(top_k) & set(user_relevant):
            hits += 1

    return hits / total_users
