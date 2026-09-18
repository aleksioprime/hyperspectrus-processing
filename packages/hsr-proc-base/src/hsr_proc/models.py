"""Данные, которыми обмениваются рабочее место и алгоритм обработки.

Структуры здесь - граница между приложением и алгоритмом. Приложение собирает
их из базы и файлов, алгоритм получает только массивы и коэффициенты и ничего
не знает ни про базу, ни про интерфейс. Поэтому алгоритм можно заменить, не
трогая рабочее место, и проверять его отдельно.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .errors import InvalidInput

#: Карта величины по пикселям кадра.
FloatMap = NDArray[np.float32]

#: Двоичная маска области кадра.
BoolMask = NDArray[np.bool_]


@dataclass(frozen=True)
class Chromophore:
    """Вещество, концентрацию которого восстанавливают по спектрам."""

    symbol: str
    """Обозначение, например ``HbO2``. Служит ключом в результатах."""

    name: str = ""
    """Название для интерфейса."""


@dataclass(frozen=True)
class SpectralCube:
    """Серия кадров одного участка, снятых в разных спектрах.

    Слои идут в том же порядке, что и ``wavelengths_nm``. Значения - яркости,
    приведённые к диапазону от 0 до 1: приводить их обязано приложение, потому
    что разрядность зависит от формата снимка.
    """

    wavelengths_nm: tuple[int, ...]
    data: FloatMap

    def __post_init__(self) -> None:
        """Проверить согласованность формы куба и списка длин волн."""
        if self.data.ndim != 3:
            raise InvalidInput(
                f"куб должен быть трёхмерным (спектры, высота, ширина), получено {self.data.ndim}"
            )
        if self.data.shape[0] != len(self.wavelengths_nm):
            raise InvalidInput(
                f"слоёв {self.data.shape[0]}, а длин волн {len(self.wavelengths_nm)}"
            )
        if not self.wavelengths_nm:
            raise InvalidInput("куб не содержит ни одного спектра")
        if len(set(self.wavelengths_nm)) != len(self.wavelengths_nm):
            raise InvalidInput(f"длины волн повторяются: {self.wavelengths_nm}")
        if not np.isfinite(self.data).all():
            # Пропущенное значение в одном слое исказило бы концентрации во
            # всех хромофорах этого пикселя, причём молча.
            raise InvalidInput("в кубе есть значения NaN или бесконечности")

    @property
    def count(self) -> int:
        """Число спектров в серии."""
        return len(self.wavelengths_nm)

    @property
    def height(self) -> int:
        """Высота кадра в пикселях."""
        return int(self.data.shape[1])

    @property
    def width(self) -> int:
        """Ширина кадра в пикселях."""
        return int(self.data.shape[2])


@dataclass(frozen=True)
class OverlapMatrix:
    """Коэффициенты перекрытия спектров и хромофоров.

    Строки - длины волн, столбцы - хромофоры. Значение показывает, насколько
    хромофор поглощает на этой длине волны; по этой матрице и решается
    обратная задача восстановления концентраций.
    """

    wavelengths_nm: tuple[int, ...]
    chromophores: tuple[Chromophore, ...]
    values: FloatMap

    def __post_init__(self) -> None:
        """Проверить, что размеры матрицы соответствуют подписям строк и столбцов."""
        if self.values.ndim != 2:
            raise InvalidInput(
                f"матрица должна быть двумерной, получено измерений: {self.values.ndim}"
            )
        expected = (len(self.wavelengths_nm), len(self.chromophores))
        if self.values.shape != expected:
            raise InvalidInput(
                f"размер матрицы {self.values.shape} не совпадает с ожидаемым {expected}"
            )
        if not self.chromophores:
            raise InvalidInput("матрица не содержит ни одного хромофора")
        if not np.isfinite(self.values).all():
            raise InvalidInput("в матрице коэффициентов есть значения NaN или бесконечности")

    @property
    def symbols(self) -> tuple[str, ...]:
        """Обозначения хромофоров в порядке столбцов."""
        return tuple(chromophore.symbol for chromophore in self.chromophores)

    def aligned_to(self, cube: SpectralCube) -> FloatMap:
        """Вернуть матрицу со строками в порядке спектров куба.

        Порядок кадров в серии задаёт устройство, а порядок строк матрицы -
        справочник рабочего места. Совпадать они не обязаны, и молчаливое
        несовпадение исказило бы все концентрации сразу.
        """
        missing = set(cube.wavelengths_nm) - set(self.wavelengths_nm)
        if missing:
            raise InvalidInput(f"в матрице нет коэффициентов для длин волн: {sorted(missing)}")
        order = [self.wavelengths_nm.index(wavelength) for wavelength in cube.wavelengths_nm]
        return np.asarray(self.values[order, :], dtype=np.float32)


@dataclass(frozen=True)
class ProcessingParams:
    """Параметры обработки.

    Общие для всех реализаций поля объявлены явно; всё, что нужно конкретному
    алгоритму, передаётся в ``extra`` - это позволяет подключить алгоритм с
    собственными настройками, не меняя контракт.
    """

    gaussian_sigma: float = 1.0
    """Сглаживание перед выделением области поражения."""

    extra: Mapping[str, Any] = field(default_factory=dict)

    def option(self, key: str, default: Any = None) -> Any:
        """Вернуть значение дополнительного параметра."""
        return self.extra.get(key, default)


@dataclass(frozen=True)
class ProcessingRequest:
    """Полный набор входных данных для обработки."""

    cube: SpectralCube
    overlap: OverlapMatrix
    params: ProcessingParams = field(default_factory=ProcessingParams)


@dataclass(frozen=True)
class ProcessingMetrics:
    """Числовые показатели, которые видит врач."""

    s_coefficient: float
    """Отношение средней концентрации THb в поражении к концентрации в коже."""

    mean_lesion_thb: float
    mean_skin_thb: float


@dataclass(frozen=True)
class SegmentationInfo:
    """Как была выделена область поражения."""

    method: str
    threshold: float
    gaussian_sigma: float


@dataclass(frozen=True)
class ProcessingResult:
    """Итог обработки серии."""

    concentrations: Mapping[str, FloatMap]
    """Карты концентраций по обозначению хромофора."""

    thb_map: FloatMap
    """Карта общей концентрации гемоглобина."""

    lesion_mask: BoolMask
    """Маска области поражения."""

    metrics: ProcessingMetrics
    segmentation: SegmentationInfo

    processor: str = ""
    """Имя и версия реализации, выполнившей обработку."""

    notes: str = ""

    def concentration(self, symbol: str) -> FloatMap:
        """Вернуть карту концентраций хромофора по его обозначению."""
        try:
            return self.concentrations[symbol]
        except KeyError as exc:
            known: Sequence[str] = sorted(self.concentrations)
            raise InvalidInput(f"нет карты для {symbol!r}, есть: {known}") from exc
