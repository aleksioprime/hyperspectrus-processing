"""Тесты демонстрационной ленты карт."""

from __future__ import annotations

from hsr_proc_conformance import data
from hsr_proc_conformance.demo import build_strip
from reference_processor import ПравильнаяРеализация, СОксигенацией


def test_лента_собирается_из_серий_пробы() -> None:
    """Две строки (оксигенация и THb), по карте на каждую серию."""
    request = data.reference_case().request
    results = [(label, СОксигенацией().process(request)) for label in ("до", "1 мин", "3 мин")]

    strip = build_strip(results, title="Проба")

    single = build_strip(results[:1])
    assert strip.width > single.width
    assert strip.height > 2 * request.cube.height


def test_без_оксигенации_остаётся_одна_строка() -> None:
    """Расчёт без карты оксигенации даёт ленту из одной строки THb."""
    request = data.reference_case().request
    with_oxygenation = build_strip([("до", СОксигенацией().process(request))])
    without = build_strip([("до", ПравильнаяРеализация().process(request))])

    assert without.height < with_oxygenation.height
