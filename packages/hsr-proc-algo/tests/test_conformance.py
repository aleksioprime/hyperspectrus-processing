"""Приёмочные проверки алгоритма."""

from __future__ import annotations

from hsr_proc_algo import AlgoProcessor
from hsr_proc_conformance import ProcessorConformance


class TestАлгоритм(ProcessorConformance):
    """Проверки реализации перед передачей пакета."""

    processor = AlgoProcessor()
    distribution = "hsr-proc-algo"
