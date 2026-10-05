"""Оксигенация и полный гемоглобин кожи по спектральному кубу.

Расчёт для демонстрации прибора на пробе с окклюзией: вместо карт, которые
повторяют подсветку прибора, - карта индекса оксигенации и карта THb, по
которым видно, как кожа теряет и возвращает кислород.

1. Свет восстанавливается линейным: кадры RAW (``cube.linear``) используются
   как есть, кадры JPG проходят обратную тоновую кривую камеры.
2. Свет делится на белый эталон прибора и сглаживается без складок кожи.
3. На полосах 671-939 нм решается модифицированный закон Бугера с постоянным
   фоном; HbO2 и Hb неотрицательны.
4. THb = HbO2 + Hb; оксигенация - HbO2 / THb в условной шкале 80-100.
5. Очаг выделяется порогом Отсу по THb, как в действующем расчёте; средние по
   центру кадра возвращаются отдельно.

Физике нужны полосы 671-939 нм и справочник ровно из HbO2 и Hb. Если чего-то
нет (синтетические серии приёмки, чужой справочник), считается так же, как в
действующем расчёте ``algo 1.0``, без карты оксигенации и с пояснением в
``notes``.

Почему не действующий расчёт: на снимках окклюзии 30.09.2026 его карта
совпадает с рисунком подсветки прибора (r = 0,95), первая минута окклюзии
неотличима от исходной, а кадр «до» с на 20% меньшим светом (светодиоды
просели, рука чуть дальше) он показывает ниже трёхминутной окклюзии. Здесь
равномерное ослабление света уходит в фоновый член модели - это закреплено
тестом ``test_равномерное_затемнение_не_меняет_оксигенацию``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from hsr_proc.contract import ProgressCallback, report
from hsr_proc.errors import InvalidInput, ProcessingFailed
from hsr_proc.models import (
    BoolMask,
    DisplayScale,
    FloatMap,
    ProcessingMetrics,
    ProcessingParams,
    ProcessingRequest,
    ProcessingResult,
    RegionMeans,
    SegmentationInfo,
    SpectralCube,
)
from numpy.typing import NDArray

from . import oximetry
from .calibration import DEVICE
from .segmentation import gaussian_blur, otsu_threshold, to_levels

#: Нижняя граница яркости при переводе в оптическую плотность в линейном расчёте.
MIN_INTENSITY = 1e-6

#: Обозначения гемоглобина. Прототип допускал написание через ноль.
OXYHEMOGLOBIN = ("hbo2", "hb02")
DEOXYHEMOGLOBIN = ("hb",)

#: Ниже этого значения средняя концентрация в коже считается нулевой.
MIN_MEAN_THB = 1e-6

#: Шкала показа оксигенации: якоря переводятся в 85 и 98, края - 80 и 100.
OXYGENATION_SCALE = DisplayScale(80.0, 100.0, "индекс оксигенации, усл. ед.")

#: Доля стороны кадра, которую занимает центральная область по умолчанию.
CENTRE_FRACTION = 0.5

# Дополнительные параметры обработки (``params.extra``).
#: σ сглаживания в долях ширины кадра.
SMOOTHING_OPTION = "smoothing"
#: Исключать ли складки кожи из сглаживания.
SKIN_LINES_OPTION = "skin_lines"
#: Пара значений сатурации, которые показываются как 85 и 98.
ANCHORS_OPTION = "anchors"
#: Доля стороны кадра для центральной области.
CENTRE_OPTION = "centre_fraction"


@dataclass
class _Maps:
    """Карты, посчитанные одним из двух путей, и пояснения к ним."""

    concentrations: dict[str, FloatMap]
    thb: FloatMap
    oxygenation: FloatMap | None = None
    notes: list[str] = field(default_factory=list)


class OxyProcessor:
    """Оксигенация и полный гемоглобин по спектральному кубу."""

    name = "oxy"
    version = "0.1"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты, оксигенацию и показатели."""
        cube = request.cube
        matrix = request.overlap.aligned_to(cube)
        params = request.params

        report(progress, 10, "Проверка серии и справочника")
        symbols = request.overlap.symbols
        reason = _physics_unavailable(cube, symbols)
        if reason is None:
            maps = _physics(cube, matrix, symbols, params, progress)
        else:
            maps = _linear(cube, matrix, symbols, progress)
            maps.notes.append(f"оксигенация не считается: {reason}")

        report(progress, 80, "Выделение области поражения")
        sigma = params.gaussian_sigma
        levels = to_levels(gaussian_blur(maps.thb, sigma))
        threshold = otsu_threshold(levels)
        lesion: BoolMask = levels > threshold

        report(progress, 90, "Расчёт показателей")
        centre = _centre_means(maps, _option_fraction(params, CENTRE_OPTION, CENTRE_FRACTION))
        if centre.oxygenation is not None:
            maps.notes.append(
                f"центр кадра: оксигенация {centre.oxygenation:.1f}, THb {centre.thb:.0f}"
            )
        metrics = _metrics(maps.thb, lesion, centre)

        report(progress, 100, "Обработка завершена")
        physics = maps.oxygenation is not None
        return ProcessingResult(
            concentrations=maps.concentrations,
            thb_map=maps.thb,
            lesion_mask=lesion,
            metrics=metrics,
            segmentation=SegmentationInfo(
                method="otsu", threshold=float(threshold), gaussian_sigma=sigma
            ),
            processor=f"{self.name} {self.version}",
            notes="; ".join(maps.notes),
            oxygenation=maps.oxygenation,
            oxygenation_scale=OXYGENATION_SCALE if physics else None,
            thb_scale=DisplayScale(*DEVICE.thb_scale, "THb, c·L мкМ·см") if physics else None,
        )


# ── Физический расчёт ────────────────────────────────────────────────────────


def _physics_unavailable(cube: SpectralCube, symbols: tuple[str, ...]) -> str | None:
    """Почему физический расчёт невозможен, или None, если возможен."""
    needed = set(oximetry.FIT_BANDS_NM)
    missing = sorted(needed - set(cube.wavelengths_nm))
    if missing:
        return f"в серии нет полос {missing} нм"
    uncalibrated = sorted(needed - set(DEVICE.wavelengths_nm))
    if uncalibrated:
        return f"в калибровке прибора нет полос {uncalibrated} нм"
    lowered = {symbol.lower() for symbol in symbols}
    has_oxy = any(name in lowered for name in OXYHEMOGLOBIN)
    has_deoxy = any(name in lowered for name in DEOXYHEMOGLOBIN)
    if len(symbols) != 2 or not (has_oxy and has_deoxy):
        return f"справочник должен состоять из HbO2 и Hb, а в нём {list(symbols)}"
    return None


def _physics(
    cube: SpectralCube,
    matrix: FloatMap,
    symbols: tuple[str, ...],
    params: ProcessingParams,
    progress: ProgressCallback | None,
) -> _Maps:
    """Модифицированный закон Бугера на 671-939 нм по линейному свету."""
    bands = oximetry.FIT_BANDS_NM
    rows = [cube.wavelengths_nm.index(wavelength) for wavelength in bands]
    constants = DEVICE.rows(bands)
    height, width = cube.height, cube.width
    data = np.asarray(cube.data[rows], dtype=np.float64)

    report(progress, 25, "Перевод в линейный свет и деление на белый эталон")
    if cube.linear:
        linear = data
        anchors = DEVICE.anchors_raw
        source = "RAW"
    else:
        linear = oximetry.linearize_jpeg(
            data, DEVICE.jpeg_tables[constants], DEVICE.jpeg_shading[constants], DEVICE.degree
        )
        anchors = DEVICE.anchors_jpeg
        source = "JPG (свет восстановлен по тоновой кривой камеры; точнее считать по RAW)"
    anchors = _option_anchors(params, anchors)
    white = oximetry.evaluate_polynomial(DEVICE.flat_field[constants], height, width, DEVICE.degree)

    # Делить на белое нужно до сглаживания: сглаженный свет, делённый на
    # несглаженное белое, переносит кривизну подсветки в карту, и у краёв кадра
    # одинаковая кожа выглядела бы разной.
    reflectance = linear / np.maximum(white, 1e-9)

    report(progress, 40, "Сглаживание без складок кожи")
    sigma = _option_fraction(params, SMOOTHING_OPTION, DEVICE.smoothing_fraction) * width
    if params.option(SKIN_LINES_OPTION, True):
        nir = [bands.index(wavelength) for wavelength in oximetry.LINE_BANDS_NM]
        weights = oximetry.dark_line_weights(
            np.mean(reflectance[nir], axis=0), sigma, DEVICE.line_threshold
        )
    else:
        weights = np.ones((height, width))
    smoothed = oximetry.weighted_blur(reflectance, weights[None], sigma)

    report(progress, 55, "Восстановление HbO2 и Hb")
    optical_density = oximetry.optical_density(smoothed, np.ones_like(smoothed))
    optical_density = optical_density.reshape(len(bands), -1)
    lowered = [symbol.lower() for symbol in symbols]
    oxy = next(i for i, symbol in enumerate(lowered) if symbol in OXYHEMOGLOBIN)
    deoxy = next(i for i, symbol in enumerate(lowered) if symbol in DEOXYHEMOGLOBIN)
    extinction = np.asarray(matrix[rows][:, [oxy, deoxy]], dtype=np.float64)
    try:
        fit = oximetry.solve_hemoglobin(optical_density, extinction)
    except np.linalg.LinAlgError as exc:
        raise ProcessingFailed(f"не удалось восстановить концентрации: {exc}") from exc

    report(progress, 70, "Оксигенация и полный гемоглобин")
    saturation = fit.saturation()
    undefined = ~np.isfinite(saturation)
    index = oximetry.display_index(np.where(undefined, anchors[0], saturation), anchors)
    index[undefined] = OXYGENATION_SCALE.low

    def frame(values: NDArray[Any]) -> FloatMap:
        return np.asarray(values.reshape(height, width), dtype=np.float32)

    notes = [
        f"вход {source}",
        "HbO2 и Hb - модифицированный закон Бугера на 671-939 нм с белым эталоном прибора, "
        "c·L в мкМ·см; THb = HbO2 + Hb",
        f"оксигенация - индекс 80-100 усл. ед.: сатурация {anchors[0]:.3f} показывается как 85, "
        f"{anchors[1]:.3f} - как 98; это не SpO2",
        f"калибровка: {DEVICE.source}",
    ]
    if undefined.any():
        notes.append(f"без гемоглобина {100 * undefined.mean():.1f}% пикселей, показаны как 80")
    return _Maps(
        concentrations={symbols[oxy]: frame(fit.hbo2), symbols[deoxy]: frame(fit.hb)},
        thb=frame(fit.thb),
        oxygenation=frame(index),
        notes=notes,
    )


# ── Линейный расчёт (как в algo 1.0) ─────────────────────────────────────────


def _linear(
    cube: SpectralCube,
    matrix: FloatMap,
    symbols: tuple[str, ...],
    progress: ProgressCallback | None,
) -> _Maps:
    """Решение системы по матрице коэффициентов, как в действующем расчёте."""
    report(progress, 30, "Восстановление концентраций хромофоров")
    density = -np.log10(np.clip(cube.data, MIN_INTENSITY, 1.0))
    flat = density.reshape(cube.count, cube.height * cube.width)
    try:
        solution, *_ = np.linalg.lstsq(
            matrix.astype(np.float64), flat.astype(np.float64), rcond=None
        )
    except np.linalg.LinAlgError as exc:
        raise ProcessingFailed(f"не удалось восстановить концентрации: {exc}") from exc
    maps = np.asarray(solution.reshape(matrix.shape[1], cube.height, cube.width), dtype=np.float32)
    concentrations = dict(zip(symbols, maps, strict=True))

    report(progress, 60, "Расчёт общей концентрации гемоглобина")
    thb, note = _total_hemoglobin(concentrations)
    return _Maps(concentrations=concentrations, thb=thb, notes=[note])


def _total_hemoglobin(concentrations: dict[str, FloatMap]) -> tuple[FloatMap, str]:
    """Собрать карту THb из модулей концентраций, как действующий расчёт."""
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


# ── Показатели ───────────────────────────────────────────────────────────────


def _centre_means(maps: _Maps, fraction: float) -> RegionMeans:
    """Средние THb и оксигенации по центральному прямоугольнику кадра."""
    height, width = maps.thb.shape
    margin_y = round(height * (1.0 - fraction) / 2.0)
    margin_x = round(width * (1.0 - fraction) / 2.0)
    top, left = margin_y, margin_x
    bottom, right = max(height - margin_y, top + 1), max(width - margin_x, left + 1)
    region = (slice(top, bottom), slice(left, right))
    oxygenation = None if maps.oxygenation is None else float(np.mean(maps.oxygenation[region]))
    return RegionMeans(
        box=(top, left, bottom, right),
        thb=float(np.mean(maps.thb[region])),
        oxygenation=oxygenation,
    )


def _metrics(thb_map: FloatMap, lesion: BoolMask, centre: RegionMeans) -> ProcessingMetrics:
    """Средние THb в очаге и в коже, их отношение и средние по центру."""
    skin = ~lesion
    mean_lesion = float(np.nanmean(thb_map[lesion])) if lesion.any() else 0.0
    mean_skin = float(np.nanmean(thb_map[skin])) if skin.any() else 0.0
    coefficient = mean_lesion / mean_skin if mean_skin > MIN_MEAN_THB else 0.0
    return ProcessingMetrics(
        s_coefficient=coefficient,
        mean_lesion_thb=mean_lesion,
        mean_skin_thb=mean_skin,
        centre=centre,
    )


# ── Дополнительные параметры ─────────────────────────────────────────────────


def _option_fraction(params: ProcessingParams, key: str, default: float) -> float:
    """Доля от 0 до 1 из ``params.extra``."""
    value: Any = params.option(key, default)
    try:
        fraction = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"параметр {key} должен быть числом, получено {value!r}") from exc
    if not 0.0 < fraction <= 1.0:
        raise InvalidInput(f"параметр {key} должен лежать в (0, 1], получено {fraction}")
    return fraction


def _option_anchors(params: ProcessingParams, default: tuple[float, float]) -> tuple[float, float]:
    """Якоря шкалы оксигенации: заданные на репетиции или из калибровки."""
    value: Any = params.option(ANCHORS_OPTION)
    if value is None:
        return default
    try:
        low, high = (float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise InvalidInput(f"параметр {ANCHORS_OPTION} - пара чисел, получено {value!r}") from exc
    if not (np.isfinite(low) and np.isfinite(high) and low < high):
        raise InvalidInput(f"якоря шкалы должны идти по возрастанию: {low}, {high}")
    return low, high
