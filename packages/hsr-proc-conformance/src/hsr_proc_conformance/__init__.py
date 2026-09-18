"""Приёмочные проверки пакетов с алгоритмами обработки HyperSpectRus.

Набор решает две задачи сразу. Разработчику алгоритма он отвечает на вопрос
«готов ли мой пакет», не требуя ни рабочего места, ни снимков пациентов.
Принимающему - даёт один и тот же ответ на одних и тех же данных, вместо
разбирательства с каждым пакетом заново.

Подключение к своему пакету - наследованием:

    from hsr_proc_conformance import ProcessorConformance
    from hsr_proc_myalgo import MyProcessor


    class TestМойАлгоритм(ProcessorConformance):
        processor = MyProcessor()
        distribution = "hsr-proc-myalgo"

Проверка установленного пакета целиком - командой ``hsr-proc-check``.
Прогон на настоящих кадрах - командой ``hsr-proc-run``.
"""

from __future__ import annotations

from . import data
from .config import CUSTOM, REFERENCE, Settings, load
from .packaging import Finding, inspect
from .report import Outcome, Report
from .suite import ProcessorConformance

__version__ = "0.1.0"

__all__ = [
    "CUSTOM",
    "REFERENCE",
    "Finding",
    "Outcome",
    "ProcessorConformance",
    "Report",
    "Settings",
    "data",
    "inspect",
    "load",
]
