"""Как показывать карты: цвет по фиксированной шкале, область центра, легенда.

Рисование живёт в контракте, а не в приложении и не в алгоритме, чтобы
рабочее место, прогон ``hsr-proc-run`` и демонстрационная лента показывали
одни и те же карты одинаково. Главное правило - шкала фиксированная: её
задаёт алгоритм (``DisplayScale``), и одинаковое значение на двух снимках
получает одинаковый цвет. Растяжение каждой карты по её собственному размаху
сделало бы снимки «до» и «3 мин окклюзии» одинаково яркими.

Только numpy и Pillow - те же зависимости, что у остального контракта.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from .models import DisplayScale

RGB = tuple[int, int, int]
Picture = NDArray[np.uint8]

#: Палитры: одна гамма от светлого (мало) к тёмному (много).
PALETTES: dict[str, tuple[RGB, ...]] = {
    # Оксигенация: светлое - мало кислорода, тёмно-синее - много.
    "oxygenation": (
        (205, 226, 251),
        (134, 182, 239),
        (57, 135, 229),
        (37, 106, 191),
        (24, 79, 149),
        (13, 54, 107),
    ),
    # Полный гемоглобин: светлое - мало крови, тёмно-оранжевое - много.
    "thb": (
        (251, 227, 215),
        (244, 173, 140),
        (235, 104, 52),
        (189, 74, 28),
        (122, 44, 12),
    ),
}

#: Цвет пикселей без значения (NaN) - нейтральный серый.
MISSING: RGB = (232, 231, 226)

#: Цвет рамки центральной области и подписей на светлом фоне.
INK: RGB = (21, 25, 29)
FRAME: RGB = (255, 255, 255)


def colorize(values: NDArray[Any], scale: DisplayScale, palette: str = "oxygenation") -> Picture:
    """Раскрасить карту по фиксированной шкале: (H, W) → (H, W, 3) uint8.

    Значения ниже ``scale.low`` получают самый светлый цвет палитры, выше
    ``scale.high`` - самый тёмный; NaN - серый.
    """
    stops = np.asarray(_palette(palette), dtype=np.float64)
    data = np.asarray(values, dtype=np.float64)
    position = np.clip((data - scale.low) / (scale.high - scale.low), 0.0, 1.0)
    position = np.nan_to_num(position, nan=0.0) * (len(stops) - 1)
    lower = np.clip(np.floor(position).astype(np.intp), 0, len(stops) - 2)
    weight = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - weight) + stops[lower + 1] * weight
    rgb[~np.isfinite(data)] = MISSING
    return np.asarray(np.rint(rgb), dtype=np.uint8)


def draw_region(
    picture: Picture,
    box: tuple[int, int, int, int],
    *,
    color: RGB = FRAME,
    width: int = 2,
    dash: int = 8,
) -> Picture:
    """Нарисовать пунктирную рамку области (верх, лево, низ, право) поверх карты."""
    image = Image.fromarray(np.asarray(picture, dtype=np.uint8))
    draw = ImageDraw.Draw(image)
    top, left, bottom, right = box
    bottom, right = bottom - 1, right - 1
    for start, end in (
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ):
        _dashed_line(draw, start, end, color, width, dash)
    return np.asarray(image, dtype=np.uint8)


def legend(
    scale: DisplayScale,
    palette: str = "oxygenation",
    *,
    height: int = 240,
    width: int = 18,
    ticks: int = 5,
    font_size: int = 14,
) -> Picture:
    """Вертикальная шкала-легенда с подписями значений, светлый фон."""
    font = _font(font_size)
    labels = [scale.low + (scale.high - scale.low) * i / (ticks - 1) for i in range(ticks)]
    texts = [_format(value, scale.high - scale.low) for value in labels]
    text_width = max(int(font.getlength(text)) for text in texts)
    margin = font_size
    canvas = Image.new("RGB", (width + 8 + text_width + 4, height + 2 * margin), (252, 252, 251))
    column = np.linspace(scale.high, scale.low, height)[:, None] * np.ones((1, width))
    canvas.paste(Image.fromarray(colorize(column, scale, palette)), (0, margin))
    draw = ImageDraw.Draw(canvas)
    for value, text in zip(labels, texts, strict=True):
        y = margin + round((scale.high - value) / (scale.high - scale.low) * (height - 1))
        draw.line([(width, y), (width + 4, y)], fill=INK, width=1)
        draw.text((width + 8, y), text, fill=INK, font=font, anchor="lm")
    return np.asarray(canvas, dtype=np.uint8)


def label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Шрифт с кириллицей для подписей; встроенный, если системного нет."""
    return _font(size)


def _palette(name: str) -> tuple[RGB, ...]:
    try:
        return PALETTES[name]
    except KeyError as exc:
        raise ValueError(f"нет палитры {name!r}, есть: {sorted(PALETTES)}") from exc


def _dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: RGB,
    width: int,
    dash: int,
) -> None:
    (x0, y0), (x1, y1) = start, end
    length = max(abs(x1 - x0), abs(y1 - y0))
    for offset in range(0, length + 1, 2 * dash):
        a = offset / length if length else 0.0
        b = min(offset + dash, length) / length if length else 0.0
        draw.line(
            [(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)],
            fill=color,
            width=width,
        )


def _format(value: float, span: float) -> str:
    """Подпись деления: целые при широкой шкале, десятые - при узкой."""
    return f"{value:.0f}" if span >= 10 or float(value).is_integer() else f"{value:.1f}"


#: Шрифты с кириллицей: Windows, Linux, macOS.
FONT_CANDIDATES = ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc")


@cache
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)
