"""Синтетические наборы данных для проверок.

Задача обратима, и это даёт проверку без единого реального снимка: по
известным концентрациям считается оптическая плотность, по ней - яркости, и
алгоритм обязан вернуть исходные концентрации. Ошибиться в такой проверке
нельзя: правильный ответ известен заранее.

Наборы здесь заведомо простые. Настоящие снимки они не заменяют - для них
есть `datasets/` и команда `hsr-proc-run`, - но неверный алгоритм спотыкается
уже на них.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hsr_proc.models import (
    Chromophore,
    OverlapMatrix,
    ProcessingParams,
    ProcessingRequest,
    SpectralCube,
)

#: Длины волн синтетических серий. Взяты из рабочего набора прибора.
WAVELENGTHS: tuple[int, ...] = (450, 517, 671, 939)

#: Коэффициенты перекрытия: оксигемоглобин сильнее поглощает на коротких
#: волнах, дезоксигемоглобин - на длинных.
COEFFICIENTS = np.array(
    [[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]],
    dtype=np.float32,
)

#: Справочник хромофоров, который рабочее место передаёт в обычном случае.
HEMOGLOBIN: tuple[Chromophore, ...] = (
    Chromophore(symbol="HbO2", name="Оксигемоглобин"),
    Chromophore(symbol="Hb", name="Дезоксигемоглобин"),
)


@dataclass(frozen=True)
class Case:
    """Набор данных с заранее известным правильным ответом."""

    name: str
    """Короткое имя для отчёта."""

    description: str
    """Что этот набор проверяет."""

    request: ProcessingRequest

    concentrations: np.ndarray | None = None
    """Эталонные карты концентраций, если они известны."""

    lesion: slice | None = None
    """Границы очага по обеим осям, если он есть."""


def overlap(
    chromophores: tuple[Chromophore, ...] = HEMOGLOBIN,
    *,
    wavelengths: tuple[int, ...] = WAVELENGTHS,
    coefficients: np.ndarray = COEFFICIENTS,
) -> OverlapMatrix:
    """Собрать матрицу коэффициентов перекрытия."""
    return OverlapMatrix(
        wavelengths_nm=wavelengths,
        chromophores=chromophores,
        values=np.asarray(coefficients, dtype=np.float32),
    )


def make_concentrations(
    size: int = 40,
    lesion: slice | None = slice(14, 26),
    *,
    skin: tuple[float, float] = (0.2, 0.1),
    lesion_values: tuple[float, float] = (0.6, 0.3),
) -> np.ndarray:
    """Построить эталонные карты концентраций: однородная кожа и очаг в центре."""
    maps = np.zeros((2, size, size), dtype=np.float32)
    maps[0, :, :] = skin[0]
    maps[1, :, :] = skin[1]
    if lesion is not None:
        maps[0, lesion, lesion] = lesion_values[0]
        maps[1, lesion, lesion] = lesion_values[1]
    return maps


def make_cube(
    concentrations: np.ndarray,
    *,
    wavelengths: tuple[int, ...] = WAVELENGTHS,
    coefficients: np.ndarray = COEFFICIENTS,
) -> SpectralCube:
    """Собрать куб яркостей, соответствующий заданным концентрациям.

    Прямая задача обратна тому, что делает алгоритм: по концентрациям считается
    оптическая плотность, а по ней - яркость. Значит, восстановленные
    концентрации должны совпасть с исходными.
    """
    optical_density = np.einsum("sc,cyx->syx", coefficients, concentrations)
    intensities = np.power(10.0, -optical_density)
    return SpectralCube(
        wavelengths_nm=wavelengths,
        data=np.asarray(intensities, dtype=np.float32),
    )


def make_request(
    concentrations: np.ndarray,
    *,
    matrix: OverlapMatrix | None = None,
    gaussian_sigma: float = 1.0,
) -> ProcessingRequest:
    """Собрать полный запрос обработки по эталонным концентрациям."""
    return ProcessingRequest(
        cube=make_cube(concentrations),
        overlap=matrix if matrix is not None else overlap(),
        params=ProcessingParams(gaussian_sigma=gaussian_sigma),
    )


def reference_case() -> Case:
    """Основной набор: однородная кожа и втрое более насыщенный очаг в центре."""
    lesion = slice(14, 26)
    concentrations = make_concentrations(lesion=lesion)
    return Case(
        name="эталон",
        description="очаг в центре кадра, отношение средних THb равно 3",
        request=make_request(concentrations),
        concentrations=concentrations,
        lesion=lesion,
    )


def uniform_case(level: tuple[float, float] = (0.2, 0.1)) -> Case:
    """Кадр без очага: выделять нечего, и делить на ноль нельзя."""
    concentrations = make_concentrations(size=20, lesion=None, skin=level)
    return Case(
        name=f"однородный кадр {level[0]}/{level[1]}",
        description="поражения нет; показатели обязаны остаться числами",
        request=make_request(concentrations),
        concentrations=concentrations,
    )


def without_hemoglobin_case() -> Case:
    """Справочник, в котором гемоглобина нет вовсе.

    Справочник ведёт пользователь рабочего места, и такие встречаются. Отказ
    обрабатывать серию - нормальный исход, но молча посчитать THb неизвестно
    из чего нельзя.
    """
    concentrations = make_concentrations()
    matrix = overlap((Chromophore(symbol="Mel"), Chromophore(symbol="H2O")))
    return Case(
        name="справочник без гемоглобина",
        description="в справочнике ни HbO2, ни Hb",
        request=make_request(concentrations, matrix=matrix),
        concentrations=concentrations,
    )


def without_deoxy_case() -> Case:
    """Справочник без дезоксигемоглобина: неполный, но пригодный."""
    concentrations = make_concentrations()
    matrix = overlap((Chromophore(symbol="HbO2"), Chromophore(symbol="Mel")))
    return Case(
        name="справочник без Hb",
        description="дезоксигемоглобина в справочнике нет",
        request=make_request(concentrations, matrix=matrix),
        concentrations=concentrations,
    )


def frame_case(height: int = 720, width: int = 1280) -> Case:
    """Кадр натурального размера: на нём меряется время обработки."""
    concentrations = np.zeros((2, height, width), dtype=np.float32)
    concentrations[0, :, :] = 0.2
    concentrations[1, :, :] = 0.1
    lesion_rows = slice(height // 3, 2 * height // 3)
    lesion_columns = slice(width // 3, 2 * width // 3)
    concentrations[0, lesion_rows, lesion_columns] = 0.6
    concentrations[1, lesion_rows, lesion_columns] = 0.3
    return Case(
        name=f"кадр {width}×{height}",
        description="размер кадра прибора: проверяется время и память",
        request=make_request(concentrations),
        concentrations=concentrations,
    )
