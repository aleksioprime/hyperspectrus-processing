"""Сглаживание и пороговое выделение области поражения.

Реализовано на numpy, без opencv и scikit-image: пакет описывает контракт
обработки, и тянуть ради двух функций тяжёлые зависимости в рабочее место
незачем. Результат совпадает с ``cv2.threshold(..., THRESH_OTSU)``.
"""

from __future__ import annotations

import numpy as np
from hsr_proc.models import FloatMap
from numpy.lib.stride_tricks import sliding_window_view

#: Число уровней яркости, по которым строится гистограмма для метода Отсу.
LEVELS = 256


def gaussian_blur(image: FloatMap, sigma: float) -> FloatMap:
    """Размыть карту гауссовым ядром.

    Сглаживание убирает пиксельный шум, из-за которого порог Отсу дробит
    область поражения на отдельные пятна.
    """
    if sigma <= 0:
        return image

    radius = max(1, round(3 * sigma))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()

    # Ядро разделимо: два одномерных прохода дают тот же результат, что и
    # двумерная свёртка, но заметно дешевле.
    blurred = _convolve(image.astype(np.float64), kernel, axis=1)
    blurred = _convolve(blurred, kernel, axis=0)
    return np.asarray(blurred, dtype=np.float32)


def _convolve(image: np.ndarray, kernel: np.ndarray, *, axis: int) -> np.ndarray:
    """Свернуть карту с одномерным ядром вдоль указанной оси."""
    radius = (len(kernel) - 1) // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    # Края продлеваются, а не обнуляются: иначе по периметру появляется тёмная
    # рамка, которую порог принимает за границу области.
    padded = np.pad(image, padding, mode="edge")
    windows = sliding_window_view(padded, len(kernel), axis=axis)
    return np.asarray(np.tensordot(windows, kernel, axes=([-1], [0])))


def to_levels(image: FloatMap) -> np.ndarray:
    """Привести карту к целым уровням яркости для построения гистограммы."""
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)

    low = float(np.min(image[finite]))
    high = float(np.max(image[finite]))
    # Постоянная карта не имеет размаха: любое масштабирование дало бы деление
    # на ноль, а порог на ней всё равно не имеет смысла.
    span = high - low
    if span <= 0:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = (np.nan_to_num(image, nan=low) - low) / span * (LEVELS - 1)
    return np.asarray(np.clip(scaled, 0, LEVELS - 1), dtype=np.uint8)


def otsu_threshold(levels: np.ndarray) -> int:
    """Найти порог методом Отсу: максимум межклассовой дисперсии."""
    histogram = np.bincount(levels.ravel(), minlength=LEVELS).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0

    weights_low = np.cumsum(histogram)[:-1]
    weights_high = total - weights_low
    sums = np.cumsum(histogram * np.arange(LEVELS, dtype=np.float64))
    total_sum = sums[-1]

    usable = (weights_low > 0) & (weights_high > 0)
    means_low = np.divide(sums[:-1], weights_low, out=np.zeros_like(weights_low), where=usable)
    means_high = np.divide(
        total_sum - sums[:-1], weights_high, out=np.zeros_like(weights_high), where=usable
    )

    variance = weights_low * weights_high * (means_low - means_high) ** 2
    variance[~usable] = -1.0
    return int(np.argmax(variance))
