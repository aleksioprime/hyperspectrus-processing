"""Проверки на подопытных с внесёнными нарушениями.

Каждая проверка набора существует ради конкретной беды. Здесь эта беда
воспроизводится нарочно, и проверка обязана её увидеть. Без этого набор
разрастается утверждениями, которые ничего не ловят, и обнаруживается это
только на пакете, прошедшем приёмку и сломавшем рабочее место.
"""

from __future__ import annotations

from typing import Any

import pytest
from hsr_proc.contract import Processor
from hsr_proc_conformance import ProcessorConformance, Settings, data
from reference_processor import (
    БезВыравнивания,
    Молчаливый,
    ПеретирающийВход,
    ПишущийФайл,
    ПравильнаяРеализация,
    Случайный,
    СПропусками,
    СЧужойПодписью,
)


def проверка(processor: Processor, name: str, *args: Any) -> None:
    """Выполнить одну проверку набора над указанной реализацией."""
    подопытный = type("Проверки", (ProcessorConformance,), {"processor": processor})()
    getattr(подопытный, name)(*args)


@pytest.mark.parametrize(
    ("подопытный", "проверяет"),
    [
        (ПеретирающийВход(), "test_вход_не_изменяется"),
        (БезВыравнивания(), "test_порядок_строк_матрицы_не_влияет"),
        (ПишущийФайл(), "test_обработка_не_пишет_файлов_и_не_ходит_в_сеть"),
        (Случайный(), "test_повторный_прогон_даёт_тот_же_результат"),
        (Молчаливый(), "test_ход_обработки_сообщается"),
    ],
    ids=lambda value: value if isinstance(value, str) else type(value).__name__,
)
def test_нарушение_замечено(подопытный: Processor, проверяет: str) -> None:
    """Проверка обязана упасть на реализации с внесённым нарушением."""
    with pytest.raises(AssertionError):
        проверка(подопытный, проверяет, подопытный)


def test_пропуск_в_карте_замечен() -> None:
    """NaN в карте дошёл бы до карточки сеанса молча."""
    подопытный = СПропусками()
    case = data.reference_case()
    result = подопытный.process(case.request)

    with pytest.raises(AssertionError):
        проверка(подопытный, "test_результат_заполнен_целиком", result, case)


def test_чужая_подпись_замечена() -> None:
    """Подпись чужим именем сделала бы сеанс неотличимым от посчитанного другим алгоритмом."""
    подопытный = СЧужойПодписью()
    result = подопытный.process(data.reference_case().request)

    with pytest.raises(AssertionError):
        проверка(подопытный, "test_результат_подписан_именем_и_версией", подопытный, result)


def test_тихое_послабление_допуска_замечено() -> None:
    """Ослабленный допуск без объяснения приёмку не проходит."""
    подопытный = ПравильнаяРеализация()
    settings = Settings(concentration_atol=1e-2)

    assert settings.relaxed(), "ослабленный допуск не распознан"
    with pytest.raises(AssertionError):
        проверка(подопытный, "test_послабления_объяснены", settings)


def test_объяснённое_послабление_принимается() -> None:
    """Объявленное отступление - повод для разговора, а не для отказа."""
    подопытный = ПравильнаяРеализация()
    settings = Settings(
        profile="custom",
        concentration_atol=1e-2,
        deviations=("Модель учитывает рассеяние",),
    )

    проверка(подопытный, "test_послабления_объяснены", settings)
