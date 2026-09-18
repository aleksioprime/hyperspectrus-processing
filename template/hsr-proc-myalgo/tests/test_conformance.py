"""Приёмочные проверки алгоритма.

Весь набор достаётся наследованием: свои проверки к нему добавляются обычными
методами класса, а общие обновляются вместе с пакетом ``hsr-proc-conformance``.
"""

from __future__ import annotations

from hsr_proc_conformance import ProcessorConformance
from hsr_proc_myalgo import MyProcessor


class TestМойАлгоритм(ProcessorConformance):
    """Проверки реализации перед передачей пакета."""

    processor = MyProcessor()
    distribution = "hsr-proc-myalgo"
