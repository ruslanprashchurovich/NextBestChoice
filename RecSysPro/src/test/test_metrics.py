import pytest
import numpy as np
from metrics import (
    normalized_dcg,
    cumulative_gain,
    discounted_cumulative_gain,
    batch_normalized_dcg,
    avg_ndcg,
    hit_rate_at_k,
)

# pytest test_metrics.py -vs


# Тест корректных значений
def test_ndcg_standard_case():
    rel = [5, 4, 3, 2, 1, 0]
    assert round(normalized_dcg(rel, k=4, method="standard"), 4) == 1.0000
    assert round(normalized_dcg(rel, k=4, method="industry"), 4) == 1.0000


# Тест метода
def test_invalid_method_raises_error():
    with pytest.raises(ValueError):
        normalized_dcg([1, 2, 3], k=2, method="invalid")


# Тест при k > len(relevance)
def test_k_greater_than_length():
    rel = [3, 2]
    assert normalized_dcg(rel, k=4, method="standard") == 1.0


# Тест на нулевую релевантность
def test_all_zero_relevance():
    rel = [0, 0, 0]
    assert normalized_dcg(rel, k=3, method="standard") == 0.0


# Тест минимального значения
def test_min_value():
    rel = [0, 1, 0, 0]
    assert normalized_dcg(rel, k=4, method="standard") < 1.0


# Тест с полной идеальной релевантностью
def test_perfect_relevance():
    rel = [5, 5, 5, 5]
    assert normalized_dcg(rel, k=4, method="standard") == 1.0


# Тест с пустым списком
def test_empty_list():
    assert normalized_dcg([], k=3, method="standard") == 0.0


# Тест k = 0
def test_k_zero():
    with pytest.raises(ValueError):
        normalized_dcg([1, 2], k=0)


# Тест k = 1
def test_k_equals_1():
    assert normalized_dcg([1, 0, 0], k=1, method="standard") == 1.0


def test_avg_ndcg_all_perfect():
    assert avg_ndcg([[3, 2, 1], [2, 1, 0]], 3) == 1.0


def test_avg_ndcg_mixed():
    score = avg_ndcg([[1, 2, 3], [3, 2, 1], [0, 0, 0]], 3)
    assert 0 <= score <= 1.0


def test_avg_ndcg_k_1():
    assert avg_ndcg([[5], [1], [3]], 1) == 1.0


def test_avg_ndcg_k_2():
    score = avg_ndcg([[1, 2], [2, 1]], 2)
    assert 0 <= score <= 1.0


def test_avg_ndcg_industry_mode():
    score = avg_ndcg([[3, 2, 1], [1, 2, 3]], 3, method="industry")
    assert 0 <= score <= 1.0


def test_avg_ndcg_one_zero():
    assert avg_ndcg([[0]][[0]][[0]], 3) == 1.0


def test_avg_ndcg_different_lengths():
    score = avg_ndcg([[1, 2], [1, 2, 3], [1]], 2)
    assert 0 <= score <= 1.0


def test_avg_ndcg_empty_input_returns_zero():
    assert avg_ndcg([], 2) == 0.0


def test_avg_ndcg_invalid_k():
    try:
        avg_ndcg([[1]][[2]], 0)
    except ValueError:
        pass
    else:
        assert False, "Expected error for invalid k"


def test_avg_ndcg_method_invalid():
    try:
        avg_ndcg([[1]][[2]], 2, method="invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected error for invalid method"


def test_batch_ndcg_all_perfect():
    assert list(batch_normalized_dcg([[3, 2, 1], [2, 1, 0]], 3)) == [1.0, 1.0]


def test_batch_ndcg_mixed():
    scores = batch_normalized_dcg([[1, 2, 3], [3, 2, 1], [0, 0, 0]], 3)
    assert all(0 <= s <= 1 for s in scores)


def test_batch_ndcg_k_1():
    scores = batch_normalized_dcg([[5], [1], [3]], 1)
    assert all(s == 1.0 for s in scores)


def test_batch_ndcg_k_2():
    scores = batch_normalized_dcg([[1, 2], [2, 1]], 2)
    assert all(0 <= s <= 1 for s in scores)


def test_batch_ndcg_industry_mode():
    scores = batch_normalized_dcg([[3, 2, 1], [1, 2, 3]], 3, method="industry")
    assert all(0 <= s <= 1 for s in scores)


def test_batch_ndcg_one_zero():
    assert batch_normalized_dcg([[0]][[0]][[0]], 3)[0] == 1.0


def test_batch_ndcg_different_lengths():
    scores = batch_normalized_dcg([[1, 2], [1, 2, 3], [1]], 2)
    assert len(scores) == 3


def test_batch_ndcg_empty_input():
    assert len(batch_normalized_dcg([], 2)) == 0


def test_batch_ndcg_invalid_k():
    try:
        batch_normalized_dcg([[1]][[2]], 0)
    except ValueError:
        pass
    else:
        assert False, "Expected error for invalid k"


def test_batch_ndcg_method_invalid():
    try:
        batch_normalized_dcg([[1]][[2]], 2, method="invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected error for invalid method"


def test_cumulative_gain_full():
    assert cumulative_gain([3, 2, 1], 3) == 6


def test_cumulative_gain_k_1():
    assert cumulative_gain([5, 3, 2], 1) == 5


def test_cumulative_gain_k_0():
    assert cumulative_gain([5, 3, 2], 0) == 0


def test_cumulative_gain_negative_values():
    assert cumulative_gain([-1, 2, -3], 3) == -2


def test_cumulative_gain_partial_k():
    assert cumulative_gain([1, 1, 1, 1], 2) == 2


def test_cumulative_gain_zero_relevance():
    assert cumulative_gain([0, 0, 0], 3) == 0


def test_cumulative_gain_single_element():
    assert cumulative_gain([10], 1) == 10


def test_cumulative_gain_empty_list():
    assert cumulative_gain([], 0) == 0


def test_cumulative_gain_k_larger_than_length():
    assert cumulative_gain([1, 2], 5) == 3


def test_cumulative_gain_floats():
    assert cumulative_gain([1.5, 2.5, 3.0], 2) == 4.0


def test_dcg_standard_simple():
    assert discounted_cumulative_gain([3, 2, 1], 3, "standard") == 3 + 2 / np.log2(
        3
    ) + 1 / np.log2(4)


def test_dcg_industry_simple():
    dcg = (2**3 - 1) / np.log2(2) + (2**2 - 1) / np.log2(3) + (2**1 - 1) / np.log2(4)
    assert discounted_cumulative_gain([3, 2, 1], 3, "industry") == dcg


def test_dcg_k_equals_1():
    assert discounted_cumulative_gain([5, 3, 2], 1, "standard") == 5


def test_dcg_k_0_raises_error():
    try:
        discounted_cumulative_gain([1, 2], 0)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for k=0"


def test_dcg_k_larger_than_len_raises_error():
    try:
        discounted_cumulative_gain([1, 2], 3)
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for k > len(relevance)"


def test_dcg_all_zeros():
    assert discounted_cumulative_gain([0, 0, 0], 3, "standard") == 0


def test_dcg_industry_with_zeros():
    assert discounted_cumulative_gain([0, 0], 2, "industry") == 0


def test_dcg_industry_one_item():
    assert discounted_cumulative_gain([2], 1, "industry") == (2**2 - 1) / np.log2(2)


def test_dcg_standard_with_negatives():
    assert (
        discounted_cumulative_gain([-1, 2, -3], 3, "standard")
        == (-1 + 2 + (-3)) / np.array([np.log2(i + 1) for i in range(3)]).sum()
    )


def test_dcg_method_invalid():
    try:
        discounted_cumulative_gain([1, 2], 2, method="invalid")
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError for invalid method"


# Тесты для hit_rate_at_k
def test_hr_basic():
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [[1, 4], [5, 6]]
    assert (
        hit_rate_at_k(recommended, relevant, 1) == 0.5
    )  # first user: hit, second: miss at k=1
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0  # both hit within top-2


def test_hr_no_hits():
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [[7, 8], [9, 10]]
    assert hit_rate_at_k(recommended, relevant, 3) == 0.0


def test_hr_all_hits():
    recommended = [[1, 2], [3, 4]]
    relevant = [[1], [3]]
    assert hit_rate_at_k(recommended, relevant, 1) == 1.0


def test_hr_k_larger_than_recommended():
    recommended = [[1], [2]]
    relevant = [[1], [3]]
    assert hit_rate_at_k(recommended, relevant, 3) == 0.5


def test_hr_empty_lists():
    recommended = [[], []]
    relevant = [[1], [2]]
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0


def test_hr_different_k():
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [[3], [4]]
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0


def test_hr_multiple_relevant():
    recommended = [[1, 2, 3], [4, 5, 6]]
    relevant = [[1, 3], [5, 7]]
    assert hit_rate_at_k(recommended, relevant, 1) == 0.5  # first user hit
    assert hit_rate_at_k(recommended, relevant, 2) == 1.0  # both hit


def test_hr_single_user():
    recommended = [[1, 2, 3]]
    relevant = [[3, 4]]
    assert hit_rate_at_k(recommended, relevant, 1) == 0.0
    assert hit_rate_at_k(recommended, relevant, 3) == 1.0


def test_hr_duplicate_items():
    recommended = [[1, 1, 2]]  # Неправильный формат, но должен работать
    relevant = [[1, 2]]
    assert hit_rate_at_k(recommended, relevant, 1) == 1.0


def test_hr_k_zero():
    recommended = [[1, 2, 3]]
    relevant = [[1]]
    assert hit_rate_at_k(recommended, relevant, 0) == 0.0
