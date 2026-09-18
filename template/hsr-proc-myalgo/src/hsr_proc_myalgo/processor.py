"""Реализация обработки.

Заготовка возвращает пустые карты правильного вида: проверки уровня
``contract`` на ней проходят, а числовые - нет, и так и должно быть, пока
расчёт не написан. Ищите пометки ЗДЕСЬ.
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


class MyProcessor:
    """Восстановление концентраций хромофоров."""

    #: Имя, по которому реализацию выбирают в настройках рабочего места.
    #: Между версиями не меняется: по нему отличают, чем посчитан сеанс.
    name = "myalgo"

    #: Версия алгоритма. Поднимается при любом изменении расчёта - она
    #: попадает в карточку сеанса, и по ней сравнивают результаты.
    version = "0.1"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты концентраций и показатели."""
        cube = request.cube

        # Порядок кадров задаёт прибор, порядок строк матрицы - справочник
        # рабочего места. Совпадать они не обязаны, и без этой строки
        # концентрации окажутся перепутаны молча.
        matrix = request.overlap.aligned_to(cube)
        report(progress, 10, "Подготовка данных")

        # ЗДЕСЬ: перевод яркостей в оптическую плотность и решение системы.
        # Заготовка возвращает нули, чтобы сразу было видно, что расчёта нет.
        shape = (cube.height, cube.width)
        empty: FloatMap = np.zeros(shape, dtype=np.float32)
        concentrations = {symbol: empty.copy() for symbol in request.overlap.symbols}
        report(progress, 50, "Расчёт концентраций")

        # ЗДЕСЬ: карта общей концентрации гемоглобина.
        thb_map: FloatMap = np.zeros(shape, dtype=np.float32)

        # ЗДЕСЬ: выделение области поражения по карте THb.
        lesion: BoolMask = np.zeros(shape, dtype=np.bool_)
        report(progress, 80, "Выделение области поражения")

        # ЗДЕСЬ: средние концентрации и их отношение. Деления на ноль быть не
        # должно: на однородном кадре кожи может не оказаться вовсе.
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
            notes=f"Расчёт не реализован: заготовка пакета ({matrix.shape[1]} хромофоров)",
        )
