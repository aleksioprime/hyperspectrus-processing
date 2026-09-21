"""Тесты расчёта.

Проверка построена на обратимости задачи: по известным концентрациям строятся
яркости, алгоритм восстанавливает концентрации обратно, и они должны совпасть.
"""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_concentrations, make_cube
from hsr_proc import registry
from hsr_proc.errors import InvalidInput
from hsr_proc.models import (
    Chromophore,
    OverlapMatrix,
    ProcessingParams,
    ProcessingRequest,
)
from hsr_proc_algo import AlgoProcessor


def test_реализация_подключается_точкой_входа() -> None:
    """Установка пакета - единственное, что нужно для подключения.

    Проверяется настоящая точка входа из ``pyproject.toml``, а не регистрация
    из кода. Реализация запрашивается по имени: рядом может стоять второй
    пакет с другим расчётом, и выбор по умолчанию тогда не делается вовсе.
    """
    registry.refresh()

    assert registry.get_processor("algo").name == "algo"


def test_концентрации_восстанавливаются_точно(overlap: OverlapMatrix) -> None:
    """Главная проверка: алгоритм решает ту самую систему, которую должен."""
    expected = make_concentrations()
    request = ProcessingRequest(cube=make_cube(expected), overlap=overlap)

    result = AlgoProcessor().process(request)

    assert np.allclose(result.concentration("HbO2"), expected[0], atol=1e-4)
    assert np.allclose(result.concentration("Hb"), expected[1], atol=1e-4)


def test_карта_thb_складывается_из_гемоглобинов(overlap: OverlapMatrix) -> None:
    """THb - сумма модулей концентраций оксигемоглобина и дезоксигемоглобина."""
    expected = make_concentrations()
    request = ProcessingRequest(cube=make_cube(expected), overlap=overlap)

    result = AlgoProcessor().process(request)

    assert np.allclose(result.thb_map, expected[0] + expected[1], atol=1e-4)
    assert result.notes == "THb = |HbO2| + |Hb|"


def test_очаг_поражения_выделяется(overlap: OverlapMatrix) -> None:
    """Порог должен отделить очаг от здоровой кожи, а не разрезать кадр наугад."""
    lesion = slice(14, 26)
    expected = make_concentrations(lesion=lesion)
    request = ProcessingRequest(cube=make_cube(expected), overlap=overlap)

    result = AlgoProcessor().process(request)

    # Сглаживание размывает границу, поэтому сверяются середина очага и
    # заведомо здоровая область у края кадра.
    assert result.lesion_mask[18:22, 18:22].all()
    assert not result.lesion_mask[0:6, 0:6].any()


def test_показатели_соответствуют_концентрациям(overlap: OverlapMatrix) -> None:
    """Коэффициент s - отношение средних THb в очаге и в коже."""
    expected = make_concentrations(skin=(0.2, 0.1), lesion_values=(0.6, 0.3))
    request = ProcessingRequest(cube=make_cube(expected), overlap=overlap)

    result = AlgoProcessor().process(request)

    assert result.metrics.mean_lesion_thb == pytest.approx(0.9, abs=0.05)
    assert result.metrics.mean_skin_thb == pytest.approx(0.3, abs=0.05)
    assert result.metrics.s_coefficient == pytest.approx(3.0, abs=0.2)


def test_однородный_кадр_не_даёт_ложного_очага(overlap: OverlapMatrix) -> None:
    """На коже без поражения выделять нечего, и деления на ноль быть не должно."""
    uniform = np.zeros((2, 20, 20), dtype=np.float32)
    uniform[0, :, :] = 0.2
    uniform[1, :, :] = 0.1
    request = ProcessingRequest(cube=make_cube(uniform), overlap=overlap)

    result = AlgoProcessor().process(request)

    assert not result.lesion_mask.any()
    assert result.metrics.s_coefficient == 0.0


def test_справочник_без_гемоглобина_не_срывает_обработку(
    overlap: OverlapMatrix,
) -> None:
    """Справочник ведёт пользователь, но врач должен видеть, из чего счёт."""
    matrix = OverlapMatrix(
        wavelengths_nm=overlap.wavelengths_nm,
        chromophores=(Chromophore(symbol="Mel"), Chromophore(symbol="H2O")),
        values=overlap.values,
    )
    request = ProcessingRequest(cube=make_cube(make_concentrations()), overlap=matrix)

    result = AlgoProcessor().process(request)

    assert "гемоглобин в справочнике не найден" in result.notes
    assert result.thb_map.shape == (40, 40)


def test_отсутствие_дезоксигемоглобина_отмечается(overlap: OverlapMatrix) -> None:
    """Неполный справочник - повод для пометки, а не для отказа."""
    matrix = OverlapMatrix(
        wavelengths_nm=overlap.wavelengths_nm,
        chromophores=(Chromophore(symbol="HbO2"), Chromophore(symbol="Mel")),
        values=overlap.values,
    )
    request = ProcessingRequest(cube=make_cube(make_concentrations()), overlap=matrix)

    result = AlgoProcessor().process(request)

    assert "Hb в справочнике не найден" in result.notes


def test_ход_обработки_сообщается(overlap: OverlapMatrix) -> None:
    """Обработка занимает секунды: врач должен видеть, что происходит."""
    steps: list[tuple[int, str]] = []
    request = ProcessingRequest(cube=make_cube(make_concentrations()), overlap=overlap)

    AlgoProcessor().process(request, lambda percent, message: steps.append((percent, message)))

    assert steps[0][0] > 0
    assert steps[-1][0] == 100
    assert [percent for percent, _ in steps] == sorted(percent for percent, _ in steps)


def test_сглаживание_настраивается(overlap: OverlapMatrix) -> None:
    """Параметр сглаживания попадает в описание сегментации без изменений."""
    request = ProcessingRequest(
        cube=make_cube(make_concentrations()),
        overlap=overlap,
        params=ProcessingParams(gaussian_sigma=2.5),
    )

    result = AlgoProcessor().process(request)

    assert result.segmentation.gaussian_sigma == 2.5
    assert result.segmentation.method == "otsu"


def test_реализация_подписывает_результат(overlap: OverlapMatrix) -> None:
    """По результату должно быть видно, какой алгоритм его получил."""
    request = ProcessingRequest(cube=make_cube(make_concentrations()), overlap=overlap)

    result = AlgoProcessor().process(request)

    assert result.processor == "algo 1.0"


def test_пропуски_в_матрице_отклоняются(
    overlap: OverlapMatrix, chromophores: tuple[Chromophore, ...]
) -> None:
    """Незаполненный коэффициент должен быть виден сразу, а не портить результат.

    В прототипе отсутствующий коэффициент молча заменялся нулём, и понять по
    результату, что справочник неполон, было невозможно.
    """
    with pytest.raises(InvalidInput):
        OverlapMatrix(
            wavelengths_nm=overlap.wavelengths_nm,
            chromophores=chromophores,
            values=np.full((4, 2), np.nan, dtype=np.float32),
        )
