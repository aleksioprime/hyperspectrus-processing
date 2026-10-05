"""Сборка спектрального куба из файлов серии.

Устройство раскладывает снимки по наборам и форматам, а имя файла содержит
длину волны - см. ``apps/device/README.md``. Здесь эта раскладка превращается
в куб, пригодный для обработки.

Кадры JPG, PNG, TIFF и BMP читаются Pillow. Кадры DNG (RAW) читаются через
``rawpy``: он нужен только тем, кто считает по RAW, и ставится отдельно -
``hsr-proc-base[raw]``. Разбор RAW живёт здесь, а не в приложении и не в
алгоритме, чтобы обе стороны получали один и тот же куб.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import InvalidInput
from .models import SpectralCube

#: Имя кадра: длина волны в нанометрах и расширение, например ``450nm.jpg``.
FRAME_PATTERN = re.compile(r"^(?P<wavelength>\d+)nm$", re.IGNORECASE)

#: Расширения, которые умеет читать Pillow.
READABLE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

#: Расширения кадров RAW.
RAW_SUFFIXES = (".dng",)

#: Максимальное значение яркости восьмибитного снимка.
MAX_LEVEL = 255.0


def discover_frames(directory: Path) -> dict[int, Path]:
    """Найти кадры в каталоге и сопоставить их с длинами волн."""
    return _discover(directory, READABLE_SUFFIXES, "450nm.jpg")


def discover_raw_frames(directory: Path) -> dict[int, Path]:
    """Найти кадры RAW (DNG) в каталоге и сопоставить их с длинами волн."""
    return _discover(directory, RAW_SUFFIXES, "450nm.dng")


def _discover(directory: Path, suffixes: tuple[str, ...], example: str) -> dict[int, Path]:
    if not directory.is_dir():
        raise InvalidInput(f"каталог с кадрами не найден: {directory}")

    frames: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in suffixes:
            continue
        match = FRAME_PATTERN.match(path.stem)
        if match is None:
            continue
        wavelength = int(match.group("wavelength"))
        if wavelength in frames:
            raise InvalidInput(f"в каталоге {directory} несколько кадров для {wavelength} нм")
        frames[wavelength] = path

    if not frames:
        raise InvalidInput(f"в каталоге {directory} нет кадров вида «{example}»")
    return frames


def load_cube(frames: Mapping[int, Path]) -> SpectralCube:
    """Собрать куб из кадров, упорядочив слои по возрастанию длины волны.

    Порядок задан явно, потому что от него зависит соответствие строкам
    матрицы коэффициентов, а порядок обхода каталога от него не зависит.
    """
    return _stack(frames, _load_layer, linear=False)


def load_raw_cube(frames: Mapping[int, Path]) -> SpectralCube:
    """Собрать линейный куб из кадров RAW (DNG).

    Из каждого кадра вычитается чёрный уровень своего байеровского сайта,
    отсчёты делятся на размах до белого уровня, а квадрат 2x2 усредняется в
    один пиксель: под узкополосным светодиодом все сайты видят одно и то же
    отражение, и демозаика не нужна. Кадр получается вдвое меньше по каждой
    стороне - 768x432 для сенсора 1536x864.
    """
    return _stack(frames, _load_raw_layer, linear=True)


def _stack(
    frames: Mapping[int, Path], reader: Callable[[Path], np.ndarray], *, linear: bool
) -> SpectralCube:
    if not frames:
        raise InvalidInput("не передано ни одного кадра")

    wavelengths = tuple(sorted(frames))
    layers = [reader(frames[wavelength]) for wavelength in wavelengths]

    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise InvalidInput(f"кадры серии имеют разный размер: {sorted(shapes)}")

    return SpectralCube(
        wavelengths_nm=wavelengths,
        data=np.asarray(np.stack(layers, axis=0), dtype=np.float32),
        linear=linear,
    )


def load_session_set(session_dir: Path, set_number: int = 1, kind: str = "jpeg") -> SpectralCube:
    """Собрать куб из набора кадров серии, снятой устройством.

    ``kind="raw"`` берёт кадры DNG из ``raw/<набор>/`` и даёт линейный куб.
    """
    directory = session_dir / kind / str(set_number)
    if kind == "raw":
        return load_raw_cube(discover_raw_frames(directory))
    return load_cube(discover_frames(directory))


def _load_layer(path: Path) -> np.ndarray:
    """Прочитать кадр в градациях серого и привести яркости к диапазону 0…1."""
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            values = np.asarray(grayscale, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidInput(f"не удалось прочитать кадр {path}: {exc}") from exc

    return np.asarray(values / MAX_LEVEL, dtype=np.float32)


def _load_raw_layer(path: Path) -> np.ndarray:
    """Прочитать кадр DNG: линейные отсчёты 0…1, байеровский квадрат усреднён."""
    try:
        import rawpy
    except ImportError as exc:
        raise InvalidInput(
            "для кадров RAW нужен пакет rawpy: установите hsr-proc-base[raw]"
        ) from exc

    # Класс ошибки LibRaw rawpy не экспортирует явно; без него ловится OSError.
    library_error = getattr(rawpy, "LibRawError", OSError)
    try:
        with rawpy.imread(str(path)) as raw:
            mosaic = raw.raw_image_visible.astype(np.float32)
            black = np.asarray(raw.black_level_per_channel, dtype=np.float32)[
                raw.raw_colors_visible
            ]
            white = float(raw.white_level)
    except (OSError, library_error) as exc:
        raise InvalidInput(f"не удалось прочитать кадр {path}: {exc}") from exc

    span = white - float(black.mean())
    if span <= 0:
        raise InvalidInput(f"в кадре {path} белый уровень не выше чёрного")
    height, width = mosaic.shape
    height, width = height - height % 2, width - width % 2
    linear = (mosaic[:height, :width] - black[:height, :width]) / span
    binned = linear.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
    return np.asarray(np.clip(binned, 0.0, 1.0), dtype=np.float32)
