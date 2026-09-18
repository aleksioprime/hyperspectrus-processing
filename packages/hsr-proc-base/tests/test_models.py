"""Тесты структур данных обработки."""

from __future__ import annotations

import numpy as np
import pytest
from hsr_proc.errors import InvalidInput
from hsr_proc.models import (
    Chromophore,
    OverlapMatrix,
    ProcessingParams,
    SpectralCube,
)


def test_куб_знает_свои_размеры() -> None:
    """Размеры кадра берутся из массива, а не задаются отдельно."""
    cube = SpectralCube(wavelengths_nm=(450, 517), data=np.zeros((2, 4, 6), dtype=np.float32))

    assert (cube.count, cube.height, cube.width) == (2, 4, 6)


@pytest.mark.parametrize(
    ("wavelengths", "shape"),
    [
        ((450, 517), (3, 4, 4)),
        ((450, 517), (2, 4)),
        ((), (0, 4, 4)),
        ((450, 450), (2, 4, 4)),
    ],
)
def test_несогласованный_куб_отклоняется(
    wavelengths: tuple[int, ...], shape: tuple[int, ...]
) -> None:
    """Ошибка обнаруживается при сборке куба, а не в глубине алгоритма."""
    with pytest.raises(InvalidInput):
        SpectralCube(wavelengths_nm=wavelengths, data=np.zeros(shape, dtype=np.float32))


def test_матрица_переупорядочивается_под_порядок_куба(
    chromophores: tuple[Chromophore, ...],
) -> None:
    """Порядок кадров задаёт устройство, порядок строк матрицы - справочник.

    Несовпадение исказило бы все концентрации сразу, поэтому строки
    выстраиваются явно.
    """
    matrix = OverlapMatrix(
        wavelengths_nm=(517, 450),
        chromophores=chromophores,
        values=np.array([[0.7, 0.3], [0.9, 0.1]], dtype=np.float32),
    )
    cube = SpectralCube(wavelengths_nm=(450, 517), data=np.zeros((2, 2, 2), dtype=np.float32))

    aligned = matrix.aligned_to(cube)

    assert np.allclose(aligned, [[0.9, 0.1], [0.7, 0.3]])


def test_недостающая_длина_волны_отклоняется(
    chromophores: tuple[Chromophore, ...],
) -> None:
    """Молча подставлять ноль нельзя: это тихо испортило бы результат."""
    matrix = OverlapMatrix(
        wavelengths_nm=(450,),
        chromophores=chromophores,
        values=np.array([[0.9, 0.1]], dtype=np.float32),
    )
    cube = SpectralCube(wavelengths_nm=(450, 517), data=np.zeros((2, 2, 2), dtype=np.float32))

    with pytest.raises(InvalidInput, match="517"):
        matrix.aligned_to(cube)


def test_размер_матрицы_проверяется(chromophores: tuple[Chromophore, ...]) -> None:
    """Матрица должна соответствовать подписям строк и столбцов."""
    with pytest.raises(InvalidInput):
        OverlapMatrix(
            wavelengths_nm=(450, 517),
            chromophores=chromophores,
            values=np.zeros((2, 3), dtype=np.float32),
        )


def test_дополнительные_параметры_доступны_по_имени() -> None:
    """Алгоритм со своими настройками подключается без правки контракта."""
    params = ProcessingParams(extra={"метод": "unmixing"})

    assert params.option("метод") == "unmixing"
    assert params.option("порог", 0.5) == 0.5
