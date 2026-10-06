"""Приспособления для тестов расчёта.

Задача обратима, и это даёт проверку без реальных снимков: по известным
концентрациям считаются оптические плотности, по ним - яркости, и алгоритм
обязан вернуть исходные концентрации. Собственной реализации стоит
проверяться так же - файл можно скопировать вместе с пакетом.
"""

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


# ── Серии для физического расчёта ────────────────────────────────────────────

#: Полосы прибора.
DEVICE_WAVELENGTHS = (450, 517, 671, 775, 803, 851, 888, 939)

#: Справочник HSR-EXAMPLE: HbO2, Hb в М⁻¹·см⁻¹ (как в references/hsr-example.json).
DEVICE_EXTINCTION = np.array(
    [
        [62816, 103292],
        [26630, 32780],
        [291, 2795],
        [690, 1190],
        [830, 758],
        [1058, 693],
        [1128, 735],
        [1214, 693],
    ],
    dtype=np.float32,
)


@pytest.fixture
def device_overlap(chromophores: tuple[Chromophore, ...]) -> OverlapMatrix:
    """Справочник прибора на восемь полос."""
    return OverlapMatrix(
        wavelengths_nm=DEVICE_WAVELENGTHS, chromophores=chromophores, values=DEVICE_EXTINCTION
    )


def make_skin_cube(
    hbo2: np.ndarray, hb: np.ndarray, *, offset: float = 0.3, gain: float = 1.0
) -> SpectralCube:
    """Линейный куб RAW, каким его увидел бы прибор на коже с заданным гемоглобином.

    Отражение строится по модели расчёта (c·L в мкМ·см, постоянный фон в OD) и
    умножается на белый эталон прибора - подсветка в кадре настоящая,
    неравномерная. ``gain`` - равномерное изменение яркости всего кадра.
    """
    from hsr_proc_algo.calibration import DEVICE
    from hsr_proc_algo.oximetry import evaluate_polynomial

    height, width = hbo2.shape
    extinction = DEVICE_EXTINCTION.astype(np.float64)
    od = 1e-6 * np.einsum("bc,cyx->byx", extinction, np.stack([hbo2, hb]))
    reflectance = np.power(10.0, -(od + offset))
    white = evaluate_polynomial(
        DEVICE.flat_field[DEVICE.rows(DEVICE_WAVELENGTHS)], height, width, DEVICE.degree
    )
    return SpectralCube(
        wavelengths_nm=DEVICE_WAVELENGTHS,
        data=np.asarray(np.clip(gain * reflectance * white, 0, 1), dtype=np.float32),
        linear=True,
    )
