"""Сборка спектрального куба из файлов серии.

Устройство раскладывает снимки по наборам и форматам, а имя файла содержит
длину волны - см. ``apps/device/README.md``. Здесь эта раскладка превращается
в куб, пригодный для обработки.

Читаются только форматы, понятные Pillow. Файлы DNG остаются на диске: их
обработка требует собственных зависимостей, и заниматься ею должен пакет с
алгоритмом, а не контракт.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import InvalidInput
from .models import SpectralCube

#: Имя кадра: длина волны в нанометрах и расширение, например ``450nm.jpg``.
FRAME_PATTERN = re.compile(r"^(?P<wavelength>\d+)nm$", re.IGNORECASE)

#: Расширения, которые умеет читать Pillow.
READABLE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

#: Максимальное значение яркости восьмибитного снимка.
MAX_LEVEL = 255.0


def discover_frames(directory: Path) -> dict[int, Path]:
    """Найти кадры в каталоге и сопоставить их с длинами волн."""
    if not directory.is_dir():
        raise InvalidInput(f"каталог с кадрами не найден: {directory}")

    frames: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in READABLE_SUFFIXES:
            continue
        match = FRAME_PATTERN.match(path.stem)
        if match is None:
            continue
        wavelength = int(match.group("wavelength"))
        if wavelength in frames:
            raise InvalidInput(f"в каталоге {directory} несколько кадров для {wavelength} нм")
        frames[wavelength] = path

    if not frames:
        raise InvalidInput(f"в каталоге {directory} нет кадров вида «450nm.jpg»")
    return frames


def load_cube(frames: Mapping[int, Path]) -> SpectralCube:
    """Собрать куб из кадров, упорядочив слои по возрастанию длины волны.

    Порядок задан явно, потому что от него зависит соответствие строкам
    матрицы коэффициентов, а порядок обхода каталога от него не зависит.
    """
    if not frames:
        raise InvalidInput("не передано ни одного кадра")

    wavelengths = tuple(sorted(frames))
    layers = [_load_layer(frames[wavelength]) for wavelength in wavelengths]

    shapes = {layer.shape for layer in layers}
    if len(shapes) != 1:
        raise InvalidInput(f"кадры серии имеют разный размер: {sorted(shapes)}")

    return SpectralCube(
        wavelengths_nm=wavelengths,
        data=np.asarray(np.stack(layers, axis=0), dtype=np.float32),
    )


def load_session_set(session_dir: Path, set_number: int = 1, kind: str = "jpeg") -> SpectralCube:
    """Собрать куб из набора кадров серии, снятой устройством."""
    return load_cube(discover_frames(session_dir / kind / str(set_number)))


def _load_layer(path: Path) -> np.ndarray:
    """Прочитать кадр в градациях серого и привести яркости к диапазону 0…1."""
    try:
        with Image.open(path) as image:
            grayscale = image.convert("L")
            values = np.asarray(grayscale, dtype=np.float32)
    except (OSError, UnidentifiedImageError) as exc:
        raise InvalidInput(f"не удалось прочитать кадр {path}: {exc}") from exc

    return np.asarray(values / MAX_LEVEL, dtype=np.float32)
