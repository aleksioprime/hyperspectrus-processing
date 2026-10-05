"""Приёмочные проверки алгоритма."""

from __future__ import annotations

from hsr_proc_conformance import ProcessorConformance
from hsr_proc_oxy import OxyProcessor


class TestОксигенация(ProcessorConformance):
    """Проверки реализации перед передачей пакета."""

    processor = OxyProcessor()
    distribution = "hsr-proc-oxy"
