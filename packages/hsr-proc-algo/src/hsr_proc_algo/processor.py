"""Реализация обработки.

Исходный код возвращает пустые карты правильного вида: проверки уровня
``contract`` проходят, а числовые — нет. Ищите пометки ЗДЕСЬ.
"""

from __future__ import annotations

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


class AlgoProcessor:
    """Восстановление концентраций хромофоров."""

    #: Согласовано с рабочим местом и между версиями не меняется.
    name = "algo"

    #: Поднимается при любом изменении расчёта.
    version = "0.1"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты концентраций и показатели."""
        cube = request.cube

        # Порядок кадров задаёт прибор, порядок строк матрицы — справочник.
        matrix = request.overlap.aligned_to(cube)
        report(progress, 10, "Подготовка данных")

        # ЗДЕСЬ: перевод яркостей в оптическую плотность и решение системы.
        shape = (cube.height, cube.width)
        empty: FloatMap = np.zeros(shape, dtype=np.float32)
        concentrations = {symbol: empty.copy() for symbol in request.overlap.symbols}
        report(progress, 50, "Расчёт концентраций")

        # ЗДЕСЬ: карта общей концентрации гемоглобина.
        thb_map: FloatMap = np.zeros(shape, dtype=np.float32)

        # ЗДЕСЬ: выделение области поражения по карте THb.
        lesion: BoolMask = np.zeros(shape, dtype=np.bool_)
        report(progress, 80, "Выделение области поражения")

        # ЗДЕСЬ: средние концентрации и их отношение.
        metrics = ProcessingMetrics(
            s_coefficient=0.0,
            mean_lesion_thb=0.0,
            mean_skin_thb=0.0,
        )

        report(progress, 100, "Обработка завершена")
        return ProcessingResult(
            concentrations=concentrations,
            thb_map=thb_map,
            lesion_mask=lesion,
            metrics=metrics,
            segmentation=SegmentationInfo(
                method="не реализовано",
                threshold=0.0,
                gaussian_sigma=request.params.gaussian_sigma,
            ),
            processor=f"{self.name} {self.version}",
            notes=f"Расчёт не реализован ({matrix.shape[1]} хромофоров)",
        )
