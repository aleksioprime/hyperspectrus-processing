"""Тесты отображения карт."""

from __future__ import annotations

import numpy as np
import pytest
from hsr_proc.display import MISSING, PALETTES, colorize, draw_region, legend
from hsr_proc.models import DisplayScale

SCALE = DisplayScale(80.0, 100.0, "индекс оксигенации, усл. ед.")


def test_края_шкалы_получают_крайние_цвета() -> None:
    """Значения за пределами шкалы показываются крайними цветами, а не исчезают."""
    picture = colorize(np.array([[70.0, 80.0, 100.0, 120.0]]), SCALE)

    lightest, darkest = PALETTES["oxygenation"][0], PALETTES["oxygenation"][-1]
    assert tuple(picture[0, 0]) == lightest == tuple(picture[0, 1])
    assert tuple(picture[0, 2]) == darkest == tuple(picture[0, 3])


def test_шкала_фиксирована_а_не_растянута() -> None:
    """Одинаковое значение на двух разных картах - одинаковый цвет."""
    first = colorize(np.array([[90.0, 85.0]]), SCALE)
    second = colorize(np.array([[90.0, 99.0]]), SCALE)

    assert tuple(first[0, 0]) == tuple(second[0, 0])


def test_больше_значение_темнее() -> None:
    """Палитра идёт от светлого к тёмному: глаз читает тёмное как «больше»."""
    picture = colorize(np.linspace(80, 100, 21)[None, :], SCALE).astype(int)

    assert (np.diff(picture.sum(axis=2)[0]) <= 0).all()


def test_пропуск_серый() -> None:
    """Пиксель без значения не притворяется ни низким, ни высоким."""
    picture = colorize(np.array([[np.nan]]), SCALE)

    assert tuple(picture[0, 0]) == MISSING


def test_неизвестная_палитра_отклоняется() -> None:
    """Опечатка в имени палитры не должна молча давать чужие цвета."""
    with pytest.raises(ValueError, match="палитры"):
        colorize(np.zeros((2, 2)), SCALE, "радуга")


def test_рамка_рисуется_только_по_краю_области() -> None:
    """Внутри области карта не меняется: рамка не должна закрывать измерение."""
    picture = colorize(np.full((40, 60), 90.0), SCALE)

    framed = draw_region(picture, (10, 15, 30, 45), width=1, dash=4)

    changed = np.any(framed != picture, axis=2)
    assert changed.any()
    assert not changed[12:28, 17:43].any()
    assert not changed[:9].any() and not changed[31:].any()


def test_легенда_нужной_высоты() -> None:
    """Легенда встаёт рядом с картой той же высоты."""
    bar = legend(SCALE, height=200, font_size=12)

    assert bar.dtype == np.uint8 and bar.shape[2] == 3
    assert bar.shape[0] == 200 + 2 * 12
