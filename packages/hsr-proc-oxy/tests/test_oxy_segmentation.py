"""Тесты сглаживания и порогового выделения."""

from __future__ import annotations

import numpy as np
import pytest
from hsr_proc_algo.segmentation import (
    LEVELS,
    gaussian_blur,
    otsu_threshold,
    to_levels,
)


def test_порог_разделяет_две_группы_яркостей() -> None:
    """Метод Отсу должен найти границу между тёмной и светлой областями."""
    levels = np.zeros((20, 20), dtype=np.uint8)
    levels[:, 10:] = 200

    threshold = otsu_threshold(levels)

    assert 0 <= threshold < 200
    assert (levels > threshold).sum() == 200


def test_однородная_картина_не_даёт_разделения() -> None:
    """Делить нечего: любая граница была бы произвольной."""
    levels = np.full((10, 10), 128, dtype=np.uint8)

    assert (levels > otsu_threshold(levels)).sum() in (0, levels.size)


def test_приведение_к_уровням_растягивает_размах() -> None:
    """Порог считается по гистограмме, поэтому размах должен занимать всю шкалу."""
    image = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)

    levels = to_levels(image)

    assert levels.min() == 0
    assert levels.max() == LEVELS - 1


def test_постоянная_карта_приводится_к_нулям() -> None:
    """Без размаха масштабирование дало бы деление на ноль."""
    assert to_levels(np.full((4, 4), 0.5, dtype=np.float32)).tolist() == [[0] * 4] * 4


def test_пропуски_не_ломают_приведение() -> None:
    """Отдельный испорченный пиксель не должен обнулять весь кадр."""
    image = np.array([[0.0, 1.0], [np.nan, 0.5]], dtype=np.float32)

    levels = to_levels(image)

    assert levels[0, 1] == LEVELS - 1
    assert levels[1, 0] == 0


def test_сглаживание_сохраняет_средний_уровень() -> None:
    """Размытие перераспределяет яркость, но не добавляет и не убирает её."""
    rng = np.random.default_rng(seed=1)
    image = rng.random((32, 32), dtype=np.float32)

    blurred = gaussian_blur(image, sigma=1.5)

    assert blurred.mean() == pytest.approx(image.mean(), abs=0.01)
    assert blurred.std() < image.std()


def test_края_не_темнеют_после_сглаживания() -> None:
    """Продление краёв не даёт тёмной рамки, которую порог примет за границу."""
    image = np.full((16, 16), 0.7, dtype=np.float32)

    blurred = gaussian_blur(image, sigma=2.0)

    assert np.allclose(blurred, 0.7, atol=1e-5)


def test_нулевое_сглаживание_ничего_не_меняет() -> None:
    """Сглаживание можно отключить, не меняя ход обработки."""
    image = np.array([[0.1, 0.9]], dtype=np.float32)

    assert np.array_equal(gaussian_blur(image, sigma=0.0), image)
