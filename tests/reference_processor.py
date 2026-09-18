"""Реализации для проверки самого набора приёмки.

Набор проверок - такой же код, как и всё остальное, и ошибаться он может в обе
стороны: пропустить неверный алгоритм или забраковать верный. Поэтому здесь
живут две вещи: заведомо правильная реализация, на которой весь набор обязан
проходить, и подопытные с одним нарушением каждый - на них обязана падать
ровно та проверка, ради которой нарушение внесено.

Эталонной реализацией комплекса это не является: перед вами учебное решение
той же задачи, годное только для проверки проверок.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from hsr_proc import (
    BoolMask,
    FloatMap,
    ProcessingMetrics,
    ProcessingRequest,
    ProcessingResult,
    ProgressCallback,
    SegmentationInfo,
    report,
)

#: Нижняя граница яркости: логарифм нуля обратился бы в бесконечность.
MIN_INTENSITY = 1e-6

#: Ниже этого значения средняя концентрация в коже считается нулевой.
MIN_MEAN = 1e-6


class ПравильнаяРеализация:
    """Линейное обращение задачи и порог посередине размаха THb."""

    name = "reference"
    version = "1.0"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию."""
        cube = request.cube
        matrix = request.overlap.aligned_to(cube)
        report(progress, 10, "Перевод яркостей в оптическую плотность")

        density = -np.log10(np.clip(cube.data, MIN_INTENSITY, 1.0))
        flat = density.reshape(cube.count, cube.height * cube.width)
        solution, *_ = np.linalg.lstsq(
            matrix.astype(np.float64), flat.astype(np.float64), rcond=None
        )
        maps = np.asarray(
            solution.reshape(matrix.shape[1], cube.height, cube.width), dtype=np.float32
        )
        concentrations = dict(zip(request.overlap.symbols, maps, strict=True))
        report(progress, 50, "Восстановление концентраций")

        thb, note = _total(concentrations)
        threshold = float((thb.min() + thb.max()) / 2.0)
        lesion: BoolMask = thb > threshold
        report(progress, 80, "Выделение области поражения")

        metrics = _metrics(thb, lesion)
        report(progress, 100, "Обработка завершена")
        return ProcessingResult(
            concentrations=concentrations,
            thb_map=thb,
            lesion_mask=lesion,
            metrics=metrics,
            segmentation=SegmentationInfo(
                method="середина размаха",
                threshold=threshold,
                gaussian_sigma=request.params.gaussian_sigma,
            ),
            processor=f"{self.name} {self.version}",
            notes=note,
        )


class ПеретирающийВход(ПравильнаяРеализация):
    """Считает поверх входного куба и портит его следующему участку."""

    name = "перетирающий"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию, испортив вход."""
        request.cube.data[0, 0, 0] = 0.5
        return super().process(request, progress)


class БезВыравнивания(ПравильнаяРеализация):
    """Берёт матрицу как есть, не сверяя порядок строк с порядком кадров."""

    name = "без-выравнивания"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию без выравнивания матрицы."""
        matrix = np.asarray(request.overlap.values, dtype=np.float32)
        cube = request.cube
        density = -np.log10(np.clip(cube.data, MIN_INTENSITY, 1.0))
        flat = density.reshape(cube.count, cube.height * cube.width)
        solution, *_ = np.linalg.lstsq(
            matrix.astype(np.float64), flat.astype(np.float64), rcond=None
        )
        maps = np.asarray(
            solution.reshape(matrix.shape[1], cube.height, cube.width), dtype=np.float32
        )
        concentrations = dict(zip(request.overlap.symbols, maps, strict=True))
        thb, note = _total(concentrations)
        threshold = float((thb.min() + thb.max()) / 2.0)
        lesion: BoolMask = thb > threshold
        report(progress, 100, "Обработка завершена")
        return ProcessingResult(
            concentrations=concentrations,
            thb_map=thb,
            lesion_mask=lesion,
            metrics=_metrics(thb, lesion),
            segmentation=SegmentationInfo("середина размаха", threshold, 1.0),
            processor=f"{self.name} {self.version}",
            notes=note,
        )


class ПишущийФайл(ПравильнаяРеализация):
    """Складывает промежуточный результат рядом с собой."""

    name = "пишущий"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию, сохранив промежуточные данные."""
        result = super().process(request, progress)
        path = Path(tempfile.gettempdir()) / "hsr-проверка-записи.npy"
        with path.open("wb") as stream:
            np.save(stream, result.thb_map)
        path.unlink(missing_ok=True)
        return result


class Случайный(ПравильнаяРеализация):
    """Подмешивает шум и на тех же данных даёт разные числа."""

    name = "случайный"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию со случайной добавкой."""
        result = super().process(request, progress)
        noise = np.random.default_rng().normal(0.0, 0.01, result.thb_map.shape)
        return ProcessingResult(
            concentrations=result.concentrations,
            thb_map=np.asarray(result.thb_map + noise, dtype=np.float32),
            lesion_mask=result.lesion_mask,
            metrics=result.metrics,
            segmentation=result.segmentation,
            processor=result.processor,
            notes=result.notes,
        )


class Молчаливый(ПравильнаяРеализация):
    """Ничего не сообщает о ходе работы."""

    name = "молчаливый"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию, не сообщая о ходе."""
        return super().process(request, None)


class СПропусками(ПравильнаяРеализация):
    """Оставляет NaN в карте THb."""

    name = "с-пропусками"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию, оставив пропуск в карте."""
        result = super().process(request, progress)
        thb = result.thb_map.copy()
        thb[0, 0] = np.nan
        return ProcessingResult(
            concentrations=result.concentrations,
            thb_map=thb,
            lesion_mask=result.lesion_mask,
            metrics=result.metrics,
            segmentation=result.segmentation,
            processor=result.processor,
            notes=result.notes,
        )


class СЧужойПодписью(ПравильнаяРеализация):
    """Подписывает результат не своим именем."""

    name = "чужая-подпись"

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработать серию, подписав результат чужим именем."""
        result = super().process(request, progress)
        return ProcessingResult(
            concentrations=result.concentrations,
            thb_map=result.thb_map,
            lesion_mask=result.lesion_mask,
            metrics=result.metrics,
            segmentation=result.segmentation,
            processor="algo 1.0",
            notes=result.notes,
        )


def _total(concentrations: dict[str, FloatMap]) -> tuple[FloatMap, str]:
    """Собрать карту THb и пояснение к ней."""
    lookup = {symbol.lower(): symbol for symbol in concentrations}
    oxy = lookup.get("hbo2")
    deoxy = lookup.get("hb")
    if oxy is not None and deoxy is not None:
        total = np.abs(concentrations[oxy]) + np.abs(concentrations[deoxy])
        return np.asarray(total, dtype=np.float32), f"THb = |{oxy}| + |{deoxy}|"

    single = oxy or deoxy or next(iter(concentrations))
    return (
        np.asarray(np.abs(concentrations[single]), dtype=np.float32),
        f"THb = |{single}|; полного набора гемоглобина в справочнике нет",
    )


def _metrics(thb: FloatMap, lesion: BoolMask) -> ProcessingMetrics:
    """Посчитать средние концентрации и их отношение."""
    skin = ~lesion
    mean_lesion = float(np.nanmean(thb[lesion])) if lesion.any() else 0.0
    mean_skin = float(np.nanmean(thb[skin])) if skin.any() else 0.0
    return ProcessingMetrics(
        s_coefficient=mean_lesion / mean_skin if mean_skin > MIN_MEAN else 0.0,
        mean_lesion_thb=mean_lesion,
        mean_skin_thb=mean_skin,
    )
