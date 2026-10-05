"""Константы прибора: белый эталон, линеаризация JPG и якоря шкалы показа.

Лежат в ``device_208.npz`` рядом с кодом и едут в колесе вместе с пакетом.
Собираются в репозитории ``multispec`` (``scripts/export_hsr_constants.py``)
из калибровки прибора в 208 и пробы с окклюзией 30.09.2026. Файл читается
один раз при импорте; во время обработки диск не трогается.

Константы привязаны к настройкам съёмки: экспозиция, скважности светодиодов и
обработка JPG в камере. Сменились настройки - калибровку нужно пересобрать.
"""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from typing import Any

import numpy as np
from numpy.typing import NDArray

#: Файл констант прибора внутри пакета.
DEVICE_FILE = "device_208.npz"


@dataclass(frozen=True)
class DeviceCalibration:
    """Всё, что расчёт знает о приборе."""

    wavelengths_nm: tuple[int, ...]
    degree: int
    flat_field: NDArray[Any]
    """Полином log белого эталона на полосу, (полосы, термы); свет в долях размаха сенсора."""

    jpeg_tables: NDArray[Any]
    """Уровень JPG → линейный свет, (полосы, 256)."""

    jpeg_shading: NDArray[Any]
    """Полином log усиления краёв кадра в JPG, (полосы, термы)."""

    anchors_raw: tuple[float, float]
    """Сатурация, которая показывается как 85 и как 98, при входе RAW."""

    anchors_jpeg: tuple[float, float]
    """То же при входе JPG."""

    thb_scale: tuple[float, float]
    """Пределы показа THb, c·L мкМ·см."""

    smoothing_fraction: float
    """σ сглаживания в долях ширины кадра."""

    line_threshold: float
    """Порог местного ИК-контраста, ниже которого пиксель - складка кожи."""

    source: str
    """Откуда взяты константы, для пояснения врачу и журнала."""

    def rows(self, wavelengths: Sequence[int]) -> list[int]:
        """Строки констант для перечисленных длин волн, в их порядке."""
        return [self.wavelengths_nm.index(wavelength) for wavelength in wavelengths]


def load(name: str = DEVICE_FILE) -> DeviceCalibration:
    """Прочитать константы прибора из данных пакета."""
    payload = resources.files(__package__).joinpath(name).read_bytes()
    with np.load(io.BytesIO(payload)) as data:
        source = json.loads(str(data["source"]))
        return DeviceCalibration(
            wavelengths_nm=tuple(int(value) for value in data["wavelengths_nm"]),
            degree=int(data["polynomial_degree"]),
            flat_field=np.array(data["flat_field"], dtype=np.float64),
            jpeg_tables=np.array(data["jpeg_tables"], dtype=np.float64),
            jpeg_shading=np.array(data["jpeg_shading"], dtype=np.float64),
            anchors_raw=_pair(data["anchors_raw"]),
            anchors_jpeg=_pair(data["anchors_jpeg"]),
            thb_scale=_pair(data["thb_scale"]),
            smoothing_fraction=float(data["smoothing_fraction"]),
            line_threshold=float(data["line_threshold"]),
            source="; ".join(
                f"{key}: {value}" for key, value in source.items() if not key.startswith("якоря")
            ),
        )


def _pair(values: NDArray[Any]) -> tuple[float, float]:
    """Пара чисел из массива констант."""
    first, second = (float(value) for value in values)
    return first, second


#: Константы прибора, с которым работает рабочее место.
DEVICE = load()
