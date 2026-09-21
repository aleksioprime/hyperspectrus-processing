"""Обработка мультиспектральных серий: контракт, реестр и эталонная реализация.

Сам алгоритм поставляется отдельным пакетом и подключается через реестр:
рабочее место выбирает реализацию по имени, а не импортирует конкретный
модуль. Благодаря этому замена алгоритма сводится к установке другого пакета,
а проверять алгоритм можно отдельно от приложения.

Пакет не зависит ни от интерфейса, ни от базы данных: на вход поступают
массивы и коэффициенты, на выход - карты и показатели.

Как оформить и подключить пакет с алгоритмом - ``README.md``.
"""

from __future__ import annotations

from .contract import Processor, ProgressCallback, describe, report
from .errors import (
    InvalidInput,
    ProcessingError,
    ProcessingFailed,
    ProcessorNotFound,
)
from .loading import discover_frames, load_cube, load_session_set
from .models import (
    BoolMask,
    Chromophore,
    FloatMap,
    OverlapMatrix,
    ProcessingMetrics,
    ProcessingParams,
    ProcessingRequest,
    ProcessingResult,
    SegmentationInfo,
    SpectralCube,
)
from .registry import (
    ENTRY_POINT_GROUP,
    PROCESSOR_ENV,
    available,
    get_processor,
    refresh,
    register,
    unregister,
)

__version__ = "0.1.0"

__all__ = [
    "ENTRY_POINT_GROUP",
    "PROCESSOR_ENV",
    "BoolMask",
    "Chromophore",
    "FloatMap",
    "InvalidInput",
    "OverlapMatrix",
    "ProcessingError",
    "ProcessingFailed",
    "ProcessingMetrics",
    "ProcessingParams",
    "ProcessingRequest",
    "ProcessingResult",
    "Processor",
    "ProcessorNotFound",
    "ProgressCallback",
    "SegmentationInfo",
    "SpectralCube",
    "available",
    "describe",
    "discover_frames",
    "get_processor",
    "load_cube",
    "load_session_set",
    "refresh",
    "register",
    "report",
    "unregister",
]
