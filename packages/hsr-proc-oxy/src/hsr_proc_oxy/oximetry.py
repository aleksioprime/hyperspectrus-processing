"""Оксигенация и полный гемоглобин кожи по светодиодным полосам.

Физика расчёта. Перенесено из ``multispec`` (``reconstruction/tissue_oximetry.py``),
где методы сравнивались на пробе с окклюзией предплечья 30.09.2026; оттуда же
числа в пояснениях ниже. Только numpy.

Цепочка для одного кадра:

1. яркости переводятся в линейный свет: RAW уже линеен, JPG проходит обратную
   тоновую кривую и снятие усиления краёв, которое камера добавляет сама;
2. свет делится на белый эталон прибора - иначе в карте доминирует рисунок
   подсветки восьми светодиодов, а не кожа;
3. сглаживание без тонких тёмных структур кожи (складки, волосы): сигнал на
   красных полосах - десятки отсчётов, а складки модель приняла бы за
   дезоксигемоглобин;
4. по оптической плотности на 671-939 нм решается модифицированный закон
   Бугера: OD(λ) = ε_HbO2(λ)·[HbO2]L + ε_Hb(λ)·[Hb]L + сдвиг.

Почему только 671-939 нм и только постоянный фон. На 450/517 нм свет
проникает много мельче, и модель с одним путём на все полосы даёт неверный
знак изменения HbO2. А наклон фона по λ на 671-939 нм почти неотличим от
спектра HbO2 (R² = 0,97): со свободным наклоном HbO2 не определяется.

Единицы: ε - десятичные, М⁻¹·см⁻¹, как в справочнике рабочего места; длина
пути в коже неизвестна, поэтому концентрации - произведение c·L в мкМ·см.
Сатурация HbO2 / (HbO2 + Hb) от L не зависит.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

#: Нижняя граница отражения перед логарифмом.
MIN_REFLECTANCE = 1e-4

#: Полосы модифицированного закона Бугера.
FIT_BANDS_NM = (671, 775, 803, 851, 888, 939)

#: Полосы, по которым ищутся складки кожи: в ближнем ИК кровь почти прозрачна.
LINE_BANDS_NM = (888, 939)


# ── Сглаживание ──────────────────────────────────────────────────────────────


def gaussian_blur(image: Array, sigma: float) -> Array:
    """Гауссово сглаживание по двум последним осям через FFT.

    Края зеркалятся на 4σ, чтобы свёртка не замыкала кадр в тор. Свёртка
    скользящим окном, как в ``segmentation.gaussian_blur``, на кадре 1280x720
    при σ ≈ 13 держала бы в памяти ~0,6 ГБ на одну полосу.
    """
    result = np.asarray(image, dtype=np.float64)
    if sigma <= 0:
        return result
    for axis in (-2, -1):
        result = _blur_axis(result, sigma, axis)
    return result


def _blur_axis(values: Array, sigma: float, axis: int) -> Array:
    """Сгладить вдоль одной оси."""
    length = values.shape[axis]
    pad = min(int(np.ceil(4 * sigma)), length - 1)
    widths = [(0, 0)] * values.ndim
    widths[axis] = (pad, pad)
    padded = np.pad(values, widths, mode="reflect") if pad > 0 else values
    size = padded.shape[axis]
    frequencies = np.fft.rfftfreq(size)
    transfer = np.exp(-2.0 * (np.pi * sigma * frequencies) ** 2)
    shape = [1] * values.ndim
    shape[axis] = transfer.size
    spectrum = np.fft.rfft(padded, axis=axis) * transfer.reshape(shape)
    blurred = np.fft.irfft(spectrum, n=size, axis=axis)
    return np.take(blurred, np.arange(pad, pad + length), axis=axis)


def dark_line_weights(reflectance: Array, sigma: float, threshold: float) -> Array:
    """Вес 0 для тонких тёмных структур кожи, 1 - для остального.

    Складки и волосы темнят все полосы почти одинаково (ΔOD ≈ 0,017 на 450 нм
    и ≈ 0,008 в ИК) - это рельеф, а не кровь. Пиксель исключается, если он
    темнее среднего по окрестности ``sigma`` больше чем на (1 − threshold).
    """
    contrast = gaussian_blur(reflectance, 1.0) / np.maximum(gaussian_blur(reflectance, sigma), 1e-9)
    return np.asarray(contrast >= threshold, dtype=np.float64)


def weighted_blur(cube: Array, weights: Array, sigma: float) -> Array:
    """Сглаживание только по пикселям с весом (нормированная свёртка).

    Исключённые пиксели получают среднее соседей, а не свои значения, поэтому
    складки не размазываются по карте.
    """
    numerator = gaussian_blur(np.asarray(cube, dtype=np.float64) * weights, sigma)
    denominator = gaussian_blur(weights, sigma)
    return numerator / np.maximum(denominator, 1e-3)


# ── Линейный свет и белый эталон ─────────────────────────────────────────────


def polynomial_design(height: int, width: int, degree: int) -> Array:
    """Одночлены по нормированным координатам кадра, (термы, H*W).

    Координаты центров пикселей отображаются в [-1, 1], поэтому одни и те же
    коэффициенты годятся для кадра RAW (768x432) и JPG (1280x720).
    """
    y = (np.arange(height) + 0.5) / height * 2.0 - 1.0
    x = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    terms = [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]
    return np.stack([(xx**i * yy**j).ravel() for i, j in terms])


def evaluate_polynomial(coefficients: Array, height: int, width: int, degree: int) -> Array:
    """exp(полином) на сетке кадра, (полосы, H, W)."""
    design = polynomial_design(height, width, degree)
    values = np.exp(np.asarray(coefficients, dtype=np.float64) @ design)
    return np.asarray(values, dtype=np.float64).reshape(-1, height, width)


def linearize_jpeg(cube: Array, tables: Array, shading: Array, degree: int) -> Array:
    """Кадры JPG рабочего места (0..1 = уровень/255) в линейный свет.

    Камера применяет тоновую кривую (≈ raw^0,5), а до неё поднимает края кадра
    на 20-50% (коррекция виньетирования). Таблица возвращает свет по уровню,
    полином ``shading`` снимает усиление краёв.
    """
    levels = np.clip(np.rint(np.asarray(cube) * 255.0), 0, tables.shape[1] - 1).astype(np.intp)
    linear = np.stack([tables[band][levels[band]] for band in range(levels.shape[0])])
    return linear / evaluate_polynomial(shading, levels.shape[1], levels.shape[2], degree)


def optical_density(linear: Array, white: Array) -> Array:
    """OD = −log10(отражение), отражение = линейный свет / белый эталон."""
    reflectance = np.asarray(linear, dtype=np.float64) / np.maximum(white, 1e-9)
    return -np.log10(np.maximum(reflectance, MIN_REFLECTANCE))


# ── Модифицированный закон Бугера ────────────────────────────────────────────


@dataclass(frozen=True)
class HemoglobinFit:
    """Решение по всем пикселям сразу; концентрации - c·L в мкМ·см."""

    hbo2: Array
    hb: Array

    @property
    def thb(self) -> Array:
        """Полный гемоглобин."""
        return self.hbo2 + self.hb

    def saturation(self, floor: float = 1e-6) -> Array:
        """HbO2 / (HbO2 + Hb); где гемоглобина нет - NaN."""
        total = self.thb
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(total > floor, self.hbo2 / total, np.nan)


def solve_hemoglobin(od: Array, extinction: Array) -> HemoglobinFit:
    """HbO2 и Hb ≥ 0 по столбцам ``od`` (полосы, N) с постоянным фоном без знака.

    ``extinction`` - (полосы, 2): столбцы HbO2 и Hb в М⁻¹·см⁻¹. Решение точное:
    у двух ограниченных столбцов четыре активных множества, решаются все, и
    берётся допустимое с наименьшей невязкой - это оптимум выпуклой задачи
    (условия Каруша-Куна-Таккера). Каждое множество - одна псевдообратная
    матрица на все пиксели.
    """
    values = np.asarray(od, dtype=np.float64)
    columns = np.hstack(
        [np.asarray(extinction, dtype=np.float64) * 1e-6, np.ones((values.shape[0], 1))]
    )
    best = np.zeros((3, values.shape[1]))
    best_error = np.full(values.shape[1], np.inf)
    for support in ((0, 1), (0,), (1,), ()):
        chosen = [*support, 2]
        matrix = columns[:, chosen]
        solution = np.linalg.pinv(matrix) @ values
        residual = matrix @ solution - values
        error = np.einsum("bn,bn->n", residual, residual)
        feasible = (solution[: len(support)] >= 0).all(axis=0)
        better = feasible & (error < best_error)
        if not better.any():
            continue
        best[:, better] = 0.0
        best[np.ix_(chosen, better)] = solution[:, better]
        best_error[better] = error[better]
    return HemoglobinFit(hbo2=best[0], hb=best[1])


def display_index(
    saturation: Array,
    anchors: tuple[float, float],
    *,
    values: tuple[float, float] = (85.0, 98.0),
    limits: tuple[float, float] = (80.0, 100.0),
) -> Array:
    """Перевести сатурацию в условную шкалу показа.

    Линейное отображение по двум якорям: ``anchors[0]`` → ``values[0]``,
    ``anchors[1]`` → ``values[1]``; обрезка по ``limits``. Распределение и
    порядок снимков задаёт физика, якоря - только где на шкале оказаться.
    """
    low, high = anchors
    scaled = values[0] + (np.asarray(saturation) - low) * (values[1] - values[0]) / (high - low)
    return np.clip(scaled, *limits)
