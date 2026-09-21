"""Восстановление концентраций хромофоров.

Действующий расчёт рабочего места и отправная точка исследования:

1. яркости переводятся в оптическую плотность;
2. для каждого пикселя решается система уравнений по матрице коэффициентов
   перекрытия - получаются карты концентраций хромофоров;
3. по картам гемоглобина строится карта общей концентрации THb;
4. область поражения выделяется порогом Отсу после сглаживания;
5. считаются средние THb в поражении и в коже и их отношение.

В отличие от прототипа система решается сразу для всех пикселей одним вызовом:
поэлементный цикл по кадру 1280×720 - это почти миллион вызовов ``lstsq``, на
что уходили минуты.

Менять здесь можно всё, кроме имени ``name``: по нему рабочее место выбирает
расчёт. Каждый шаг проверяется приёмкой отдельно, поэтому переписывать сразу
весь алгоритм не требуется - можно заменить сегментацию, оставив восстановление
концентраций, или наоборот.
"""

from __future__ import annotations

import numpy as np
from hsr_proc.contract import ProgressCallback, report
from hsr_proc.errors import ProcessingFailed
from hsr_proc.models import (
    BoolMask,
    FloatMap,
    ProcessingMetrics,
    ProcessingRequest,
    ProcessingResult,
    SegmentationInfo,
)

from .segmentation import gaussian_blur, otsu_threshold, to_levels

#: Нижняя граница яркости при переводе в оптическую плотность.
#: Логарифм нуля обратился бы в бесконечность и испортил бы весь столбец.
MIN_INTENSITY = 1e-6

#: Обозначения гемоглобина. Прототип допускал написание через ноль, и такие
#: справочники встречаются у пользователей до сих пор.
OXYHEMOGLOBIN = ("hbo2", "hb02")
DEOXYHEMOGLOBIN = ("hb",)

#: Ниже этого значения средняя концентрация в коже считается нулевой.
MIN_MEAN_THB = 1e-6


class AlgoProcessor:
    """Восстановление концентраций хромофоров по спектральному кубу."""

    name = "algo"
    version = "1.0"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты концентраций и показатели."""
        cube = request.cube
        matrix = request.overlap.aligned_to(cube)

        report(progress, 10, "Перевод яркостей в оптическую плотность")
        optical_density = self._optical_density(cube.data)

        report(progress, 30, "Восстановление концентраций хромофоров")
        maps = self._solve(matrix, optical_density, cube.height, cube.width)
        concentrations = dict(zip(request.overlap.symbols, maps, strict=True))

        report(progress, 60, "Расчёт общей концентрации гемоглобина")
        thb_map, note = self._total_hemoglobin(concentrations)

        report(progress, 75, "Выделение области поражения")
        sigma = request.params.gaussian_sigma
        levels = to_levels(gaussian_blur(thb_map, sigma))
        threshold = otsu_threshold(levels)
        lesion: BoolMask = levels > threshold

        report(progress, 90, "Расчёт показателей")
        metrics = self._metrics(thb_map, lesion)

        report(progress, 100, "Обработка завершена")
        return ProcessingResult(
            concentrations=concentrations,
            thb_map=thb_map,
            lesion_mask=lesion,
            metrics=metrics,
            segmentation=SegmentationInfo(
                method="otsu", threshold=float(threshold), gaussian_sigma=sigma
            ),
            processor=f"{self.name} {self.version}",
            notes=note,
        )

    @staticmethod
    def _optical_density(data: FloatMap) -> FloatMap:
        """Перевести яркости в оптическую плотность.

        Оптическая плотность связана с концентрацией поглотителя линейно -
        именно это позволяет решать задачу как систему линейных уравнений.
        """
        clipped = np.clip(data, MIN_INTENSITY, 1.0)
        return np.asarray(-np.log10(clipped), dtype=np.float32)

    @staticmethod
    def _solve(matrix: FloatMap, optical_density: FloatMap, height: int, width: int) -> FloatMap:
        """Решить систему для всех пикселей сразу методом наименьших квадратов."""
        spectra = optical_density.shape[0]
        # Кадр разворачивается в один столбец на пиксель: lstsq решает систему
        # для всех столбцов за один вызов.
        flat = optical_density.reshape(spectra, height * width)
        try:
            solution, *_ = np.linalg.lstsq(
                matrix.astype(np.float64), flat.astype(np.float64), rcond=None
            )
        except np.linalg.LinAlgError as exc:
            raise ProcessingFailed(f"не удалось восстановить концентрации: {exc}") from exc

        chromophores = matrix.shape[1]
        return np.asarray(solution.reshape(chromophores, height, width), dtype=np.float32)

    @staticmethod
    def _total_hemoglobin(concentrations: dict[str, FloatMap]) -> tuple[FloatMap, str]:
        """Собрать карту общей концентрации гемоглобина и пояснение к ней.

        Справочник хромофоров ведёт пользователь, и в нём может не оказаться
        нужных обозначений. Отказывать в обработке из-за этого нельзя, но врач
        должен видеть, из чего посчитан показатель.
        """
        lookup = {symbol.lower(): symbol for symbol in concentrations}
        oxy = next((lookup[key] for key in OXYHEMOGLOBIN if key in lookup), None)
        deoxy = next((lookup[key] for key in DEOXYHEMOGLOBIN if key in lookup), None)

        if oxy is not None and deoxy is not None:
            total = np.abs(concentrations[oxy]) + np.abs(concentrations[deoxy])
            return np.asarray(total, dtype=np.float32), f"THb = |{oxy}| + |{deoxy}|"

        single = oxy or deoxy
        if single is not None:
            missing = "Hb" if oxy is not None else "HbO2"
            return (
                np.asarray(np.abs(concentrations[single]), dtype=np.float32),
                f"THb = |{single}|; хромофор {missing} в справочнике не найден",
            )

        if not concentrations:
            raise ProcessingFailed("в справочнике нет ни одного хромофора")

        fallback = next(iter(concentrations))
        return (
            np.asarray(np.abs(concentrations[fallback]), dtype=np.float32),
            f"THb = |{fallback}|; гемоглобин в справочнике не найден",
        )

    @staticmethod
    def _metrics(thb_map: FloatMap, lesion: BoolMask) -> ProcessingMetrics:
        """Посчитать средние концентрации и их отношение."""
        skin = ~lesion
        mean_lesion = float(np.nanmean(thb_map[lesion])) if lesion.any() else 0.0
        mean_skin = float(np.nanmean(thb_map[skin])) if skin.any() else 0.0
        coefficient = mean_lesion / mean_skin if mean_skin > MIN_MEAN_THB else 0.0
        return ProcessingMetrics(
            s_coefficient=coefficient,
            mean_lesion_thb=mean_lesion,
            mean_skin_thb=mean_skin,
        )
