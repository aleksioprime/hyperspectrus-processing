"""Тесты расчёта оксигенации.

Физический путь проверяется на кубах, построенных по той же модели, что и
расчёт, но с настоящей неравномерной подсветкой прибора: по известным HbO2 и
Hb строится отражение, оно умножается на белый эталон, и расчёт обязан вернуть
исходные концентрации и одинаковую по кадру оксигенацию. Запасной линейный
путь проверяется на синтетике приёмки.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from conftest import (
    DEVICE_WAVELENGTHS,
    WAVELENGTHS,
    make_concentrations,
    make_cube,
    make_skin_cube,
)
from hsr_proc import registry
from hsr_proc.errors import InvalidInput
from hsr_proc.loading import discover_raw_frames, load_raw_cube
from hsr_proc.models import (
    Chromophore,
    OverlapMatrix,
    ProcessingParams,
    ProcessingRequest,
    ProcessingResult,
    SpectralCube,
)
from hsr_proc_algo import AlgoProcessor
from hsr_proc_algo.calibration import DEVICE
from hsr_proc_algo.oximetry import FIT_BANDS_NM, evaluate_polynomial

SHAPE = (72, 128)


def uniform(value: float) -> np.ndarray:
    """Однородная карта размера тестового кадра."""
    return np.full(SHAPE, value)


def skin(saturation: float, total: float = 230.0) -> tuple[np.ndarray, np.ndarray]:
    """HbO2 и Hb с заданной сатурацией и полным гемоглобином, c·L мкМ·см."""
    return uniform(saturation * total), uniform((1 - saturation) * total)


def run(cube: SpectralCube, overlap: OverlapMatrix, **extra: Any) -> ProcessingResult:
    """Обработать куб с дополнительными параметрами."""
    params = ProcessingParams(extra=extra)
    return AlgoProcessor().process(ProcessingRequest(cube=cube, overlap=overlap, params=params))


def test_реализация_подключается_точкой_входа() -> None:
    """Рабочее место находит расчёт по имени из точки входа."""
    registry.refresh()

    assert registry.get_processor("algo").name == "algo"


def test_концентрации_восстанавливаются_сквозь_подсветку(device_overlap: OverlapMatrix) -> None:
    """HbO2 и Hb возвращаются точно, хотя кадр освещён неравномерно."""
    hbo2, hb = skin(0.68)

    result = run(make_skin_cube(hbo2, hb), device_overlap)

    assert np.allclose(result.concentration("HbO2"), hbo2, rtol=1e-3)
    assert np.allclose(result.concentration("Hb"), hb, rtol=1e-3)
    assert np.allclose(result.thb_map, hbo2 + hb, rtol=1e-3)


def test_оксигенация_не_повторяет_подсветку(device_overlap: OverlapMatrix) -> None:
    """Главное отличие от действующего расчёта: одинаковая кожа - одинаковое число по кадру."""
    result = run(make_skin_cube(*skin(0.68)), device_overlap)

    assert result.oxygenation is not None
    assert float(np.ptp(result.oxygenation)) < 0.05


@pytest.mark.parametrize(("which", "expected"), [(0, 85.0), (1, 98.0)])
def test_якоря_шкалы_переводятся_в_85_и_98(
    device_overlap: OverlapMatrix, which: int, expected: float
) -> None:
    """Сатурация якоря показывается ровно своим значением шкалы."""
    result = run(make_skin_cube(*skin(DEVICE.anchors_raw[which])), device_overlap)

    assert result.oxygenation is not None
    assert np.allclose(result.oxygenation, expected, atol=0.05)
    assert result.oxygenation_scale is not None
    assert (result.oxygenation_scale.low, result.oxygenation_scale.high) == (80.0, 100.0)
    assert result.thb_scale is not None


def test_равномерное_затемнение_не_меняет_оксигенацию(device_overlap: OverlapMatrix) -> None:
    """Светодиоды просели или рука чуть дальше - это не окклюзия."""
    hbo2, hb = skin(0.68)

    bright = run(make_skin_cube(hbo2, hb), device_overlap)
    dim = run(make_skin_cube(hbo2, hb, gain=0.8), device_overlap)

    assert bright.oxygenation is not None and dim.oxygenation is not None
    assert np.allclose(bright.oxygenation, dim.oxygenation, atol=1e-3)
    assert np.allclose(bright.thb_map, dim.thb_map, rtol=1e-3)


def test_окклюзия_снижает_оксигенацию_а_не_thb(device_overlap: OverlapMatrix) -> None:
    """Кислород уходит при том же объёме крови: падает оксигенация, THb стоит."""
    before = run(make_skin_cube(*skin(0.70)), device_overlap)
    occluded = run(make_skin_cube(*skin(0.65)), device_overlap)

    assert before.oxygenation is not None and occluded.oxygenation is not None
    assert float(occluded.oxygenation.mean()) < float(before.oxygenation.mean()) - 10
    assert np.allclose(occluded.thb_map, before.thb_map, rtol=1e-3)


def test_средние_по_центру_возвращаются(device_overlap: OverlapMatrix) -> None:
    """Центральная область - половина стороны кадра, средние совпадают с картами."""
    hbo2, hb = skin(0.68)
    hbo2[:, : SHAPE[1] // 2] *= 1.2

    result = run(make_skin_cube(hbo2, hb), device_overlap)

    centre = result.metrics.centre
    assert centre is not None and result.oxygenation is not None
    top, left, bottom, right = centre.box
    assert (top, left, bottom, right) == (18, 32, 54, 96)
    assert centre.thb == pytest.approx(float(result.thb_map[top:bottom, left:right].mean()))
    assert centre.oxygenation == pytest.approx(
        float(result.oxygenation[top:bottom, left:right].mean())
    )
    assert "центр кадра" in result.notes


def test_размер_центральной_области_настраивается(device_overlap: OverlapMatrix) -> None:
    """Область задаётся долей стороны кадра."""
    result = run(make_skin_cube(*skin(0.68)), device_overlap, centre_fraction=0.25)

    assert result.metrics.centre is not None
    assert result.metrics.centre.box == (27, 48, 45, 80)


def test_якоря_задаются_на_репетиции(device_overlap: OverlapMatrix) -> None:
    """Исходная сатурация человека другая - шкалу подстраивают без пересборки."""
    result = run(make_skin_cube(*skin(0.72)), device_overlap, anchors=(0.67, 0.72))

    assert result.oxygenation is not None
    assert np.allclose(result.oxygenation, 98.0, atol=0.05)


@pytest.mark.parametrize(
    "extra",
    [
        {"anchors": (0.7, 0.6)},
        {"anchors": "0.6"},
        {"centre_fraction": 0.0},
        {"smoothing": "много"},
    ],
)
def test_неверные_параметры_отклоняются(
    device_overlap: OverlapMatrix, extra: dict[str, Any]
) -> None:
    """Ошибка в настройках должна дойти до врача понятным сообщением."""
    with pytest.raises(InvalidInput):
        run(make_skin_cube(*skin(0.68)), device_overlap, **extra)


def test_вход_jpg_считается_через_тоновую_кривую(device_overlap: OverlapMatrix) -> None:
    """Пока рабочее место присылает JPG, расчёт возвращает свет по таблице камеры."""
    linear = make_skin_cube(*skin(0.68)).data.astype(np.float64)
    rows = DEVICE.rows(DEVICE_WAVELENGTHS)
    gain = evaluate_polynomial(DEVICE.jpeg_shading[rows], *SHAPE, DEVICE.degree)
    levels = np.stack(
        [
            np.clip(np.searchsorted(DEVICE.jpeg_tables[row], linear[band] * gain[band]), 0, 255)
            for band, row in enumerate(rows)
        ]
    )
    cube = SpectralCube(DEVICE_WAVELENGTHS, np.asarray(levels / 255.0, dtype=np.float32))

    result = run(cube, device_overlap)
    reference = run(make_skin_cube(*skin(0.68)), device_overlap)

    assert result.oxygenation is not None and reference.oxygenation is not None
    assert np.isfinite(result.oxygenation).all()
    assert "JPG" in result.notes
    assert abs(float(np.median(result.oxygenation)) - float(np.median(reference.oxygenation))) < 3


def test_складки_можно_не_исключать(device_overlap: OverlapMatrix) -> None:
    """Выключатель исключения складок работает и не ломает расчёт."""
    result = run(make_skin_cube(*skin(0.68)), device_overlap, skin_lines=False)

    assert result.oxygenation is not None


def test_без_нужных_полос_считается_как_прежде(overlap: OverlapMatrix) -> None:
    """Синтетика приёмки: физике не хватает полос, расчёт линейный, причина в пояснении."""
    expected = make_concentrations()

    result = run(make_cube(expected), overlap)

    assert result.oxygenation is None and result.oxygenation_scale is None
    assert result.metrics.centre is not None and result.metrics.centre.oxygenation is None
    assert np.allclose(result.concentration("HbO2"), expected[0], atol=1e-4)
    assert "оксигенация не считается" in result.notes
    assert str(sorted(set(FIT_BANDS_NM) - set(WAVELENGTHS))) in result.notes


def test_чужой_справочник_не_ломает_обработку(device_overlap: OverlapMatrix) -> None:
    """Справочник с меланином физике не подходит; карты всё равно есть по каждому хромофору."""
    chromophores = (Chromophore("HbO2"), Chromophore("Hb"), Chromophore("Mel"))
    melanin = np.linspace(1.0, 0.2, 8, dtype=np.float32)[:, None]
    overlap = OverlapMatrix(
        DEVICE_WAVELENGTHS, chromophores, np.hstack([device_overlap.values, melanin])
    )

    result = run(make_skin_cube(*skin(0.68)), overlap)

    assert set(result.concentrations) == {"HbO2", "Hb", "Mel"}
    assert result.oxygenation is None
    assert "справочник" in result.notes


def test_кадр_raw_укладывается_во_время(device_overlap: OverlapMatrix) -> None:
    """Кадр RAW прибора (768x432) считается за секунды."""
    cube = make_skin_cube(np.full((432, 768), 156.0), np.full((432, 768), 74.0))

    started = time.perf_counter()
    run(cube, device_overlap)

    assert time.perf_counter() - started < 10.0


#: Кадры пробы с окклюзией 30.09.2026 из репозитория multispec. В этот
#: репозиторий снимки не попадают; тест пропускается, если их нет рядом.
OCCLUSION_FRAMES = Path(
    os.environ.get(
        "HSR_OCCLUSION_FRAMES",
        str(Path(__file__).resolve().parents[4] / "multispec/data/occlusion_20260930/frames"),
    )
)


@pytest.mark.skipif(not OCCLUSION_FRAMES.is_dir(), reason="кадров пробы с окклюзией рядом нет")
def test_проба_с_окклюзией_по_raw(device_overlap: OverlapMatrix) -> None:
    """Настоящие снимки: оксигенация в центре падает при окклюзии и возвращается."""
    centre = {}
    for folder in sorted(OCCLUSION_FRAMES.iterdir()):
        result = run(load_raw_cube(discover_raw_frames(folder)), device_overlap)
        means = result.metrics.centre
        assert means is not None and means.oxygenation is not None
        centre[folder.name.split("_", 1)[1]] = means.oxygenation

    before, after = centre["до_окклюзии"], centre["2_мин_после_окклюзии"]
    one, three = centre["1_мин_окклюзии"], centre["3_мин_окклюзии"]
    assert before > one > three, centre
    assert after > one, centre
