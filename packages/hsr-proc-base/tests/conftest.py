"""Общие приспособления для тестов обработки."""

from __future__ import annotations

import numpy as np
import pytest
from hsr_proc.models import Chromophore, OverlapMatrix, SpectralCube

#: Длины волн, на которых сняты кадры в тестовых сериях.
WAVELENGTHS = (450, 517, 671, 939)

#: Коэффициенты перекрытия: оксигемоглобин сильнее поглощает на коротких
#: волнах, дезоксигемоглобин - на длинных.
COEFFICIENTS = np.array(
    [[0.9, 0.1], [0.7, 0.3], [0.3, 0.7], [0.1, 0.9]],
    dtype=np.float32,
)


@pytest.fixture
def chromophores() -> tuple[Chromophore, ...]:
    """Хромофоры тестового справочника."""
    return (
        Chromophore(symbol="HbO2", name="Оксигемоглобин"),
        Chromophore(symbol="Hb", name="Дезоксигемоглобин"),
    )


@pytest.fixture
def overlap(chromophores: tuple[Chromophore, ...]) -> OverlapMatrix:
    """Матрица коэффициентов перекрытия."""
    return OverlapMatrix(
        wavelengths_nm=WAVELENGTHS,
        chromophores=chromophores,
        values=COEFFICIENTS,
    )


def make_concentrations(
    size: int = 40,
    lesion: slice = slice(14, 26),
    *,
    skin: tuple[float, float] = (0.2, 0.1),
    lesion_values: tuple[float, float] = (0.6, 0.3),
) -> np.ndarray:
    """Построить эталонные карты концентраций: однородная кожа и очаг в центре."""
    maps = np.zeros((2, size, size), dtype=np.float32)
    maps[0, :, :] = skin[0]
    maps[1, :, :] = skin[1]
    maps[0, lesion, lesion] = lesion_values[0]
    maps[1, lesion, lesion] = lesion_values[1]
    return maps


def make_cube(concentrations: np.ndarray) -> SpectralCube:
    """Собрать куб яркостей, соответствующий заданным концентрациям.

    Прямая задача обратна тому, что делает алгоритм: по концентрациям
    считается оптическая плотность, а по ней - яркость. Значит, восстановленные
    концентрации должны совпасть с исходными.
    """
    optical_density = np.einsum("sc,cyx->syx", COEFFICIENTS, concentrations)
    intensities = np.power(10.0, -optical_density)
    return SpectralCube(
        wavelengths_nm=WAVELENGTHS,
        data=np.asarray(intensities, dtype=np.float32),
    )
