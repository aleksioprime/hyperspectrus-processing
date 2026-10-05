"""Команда ``hsr-proc-demo``: лента карт по нескольким сериям пробы - для показа.

Проба с окклюзией - это несколько серий подряд: до, во время, после. Команда
прогоняет алгоритм на каждой и собирает одну картинку: строка карт
оксигенации, строка карт THb, под каждой картой - среднее по центру кадра.
Шкалы фиксированные, их задаёт алгоритм, поэтому цвет между снимками
сравним. Это запасной путь для демонстрации, пока рабочее место не умеет
показывать карты оксигенации само, и образец того, как их показывать.

    uv run hsr-proc-demo "до=путь/к/серии" "1 мин=..." "3 мин=..." --processor oxy --out лента.png

Серия - каталог с кадрами в раскладке прибора (``raw/1``, ``jpeg/1``) или
плоский каталог, как ``source/`` зоны в хранилище рабочего места. Кадры RAW
берутся, если они есть.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from hsr_proc.display import INK, colorize, draw_region, label_font, legend
from hsr_proc.errors import ProcessingError
from hsr_proc.models import DisplayScale, ProcessingRequest, ProcessingResult
from hsr_proc.registry import get_processor
from PIL import Image, ImageDraw

from .runner import _matrix, load_series

#: Ширина одной карты в ленте, пикселей.
TILE_WIDTH = 360

#: Фон ленты.
BACKGROUND = (252, 252, 251)

#: Серый фон карты, которую алгоритм не посчитал.
EMPTY = (232, 231, 226)


def build_strip(results: Sequence[tuple[str, ProcessingResult]], *, title: str = "") -> Image.Image:
    """Собрать ленту: строка оксигенации (если есть) и строка THb.

    ``results`` - пары «подпись момента пробы, результат обработки» в порядке
    протокола.
    """
    if not results:
        raise ValueError("нет ни одной серии")
    first = results[0][1]
    height = round(TILE_WIDTH * first.thb_map.shape[0] / first.thb_map.shape[1])
    rows = []
    if any(result.oxygenation is not None for _, result in results):
        rows.append("oxygenation")
    rows.append("thb")

    label_size, number_size, gap = 18, 30, 12
    header = 34 if title else 0
    caption = label_size + 10
    number = number_size + 14
    row_height = caption + height + number
    legend_width = 90
    width = gap + len(results) * (TILE_WIDTH + gap) + legend_width
    canvas = Image.new("RGB", (width, header + len(rows) * (row_height + gap) + gap), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((gap, gap), title, fill=INK, font=label_font(20))

    for row_index, row in enumerate(rows):
        top = header + gap + row_index * (row_height + gap)
        scale = None
        for column, (label, result) in enumerate(results):
            left = gap + column * (TILE_WIDTH + gap)
            draw.text((left, top), label, fill=INK, font=label_font(label_size))
            tile, value, scale = _tile(result, row, (TILE_WIDTH, height), scale)
            canvas.paste(tile, (left, top + caption))
            if value is not None:
                draw.text(
                    (left + TILE_WIDTH // 2, top + caption + height + number // 2),
                    f"{value:.0f}",
                    fill=INK,
                    font=label_font(number_size),
                    anchor="mm",
                )
        if scale is not None:
            bar = Image.fromarray(legend(scale, row, height=height - 2 * 14, font_size=13))
            canvas.paste(bar, (width - legend_width + 6, top + caption))
            draw.text(
                (width - legend_width + 6, top),
                scale.label.rsplit(",", 1)[-1].strip(),
                fill=INK,
                font=label_font(14),
            )
    return canvas


def _tile(
    result: ProcessingResult, row: str, size: tuple[int, int], scale: DisplayScale | None
) -> tuple[Image.Image, float | None, DisplayScale | None]:
    """Карта одной серии в цвете, среднее по центру и шкала строки."""
    centre = result.metrics.centre
    if row == "oxygenation":
        values, scale = result.oxygenation, result.oxygenation_scale or scale
        value = None if centre is None else centre.oxygenation
    else:
        values, scale = result.thb_map, result.thb_scale or scale
        value = None if centre is None else centre.thb
    if values is None or scale is None:
        return Image.new("RGB", size, EMPTY), None, scale
    picture = colorize(values, scale, row)
    if centre is not None:
        picture = draw_region(picture, centre.box, width=max(2, values.shape[1] // 200))
    return Image.fromarray(picture).resize(size, Image.Resampling.BILINEAR), value, scale


def _parse(item: str) -> tuple[str, Path]:
    """Разобрать «подпись=путь» или просто путь (подписью станет имя каталога)."""
    if "=" in item:
        label, location = item.split("=", 1)
        return label.strip(), Path(location.strip())
    directory = Path(item)
    return directory.name, directory


def main(argv: list[str] | None = None) -> int:
    """Прогнать алгоритм на сериях пробы и собрать ленту карт."""
    parser = argparse.ArgumentParser(
        prog="hsr-proc-demo", description="Лента карт оксигенации и THb по сериям пробы"
    )
    parser.add_argument("series", nargs="+", help="серии в порядке пробы: «подпись=каталог»")
    parser.add_argument("--processor", help="имя реализации, если подключено несколько")
    parser.add_argument("--reference", type=Path, help="файл справочника")
    parser.add_argument("--format", choices=["auto", "raw", "jpeg"], default="auto")
    parser.add_argument("--title", default="Индекс оксигенации и полный гемоглобин кожи")
    parser.add_argument("--out", type=Path, default=Path("лента.png"), help="куда сохранить PNG")
    args = parser.parse_args(argv)

    try:
        processor = get_processor(args.processor)
        results = []
        matrix = None
        for item in args.series:
            label, path = _parse(item)
            cube = load_series(path, args.format)
            # Справочник один на все серии пробы: прибор и набор полос те же.
            matrix = matrix if matrix is not None else _matrix(args.reference, cube)
            result = processor.process(ProcessingRequest(cube=cube, overlap=matrix))
            centre = result.metrics.centre
            summary = (
                ""
                if centre is None or centre.oxygenation is None
                else (f"оксигенация в центре {centre.oxygenation:.1f}, THb {centre.thb:.0f}")
            )
            kind = "RAW" if cube.linear else "JPG"
            print(f"{label}: {kind}, {cube.width}×{cube.height}; {summary}")
            results.append((label, result))
    except (ProcessingError, OSError, ValueError) as error:
        print(f"Не удалось обработать серию: {error}", file=sys.stderr)
        return 1

    strip = build_strip(results, title=args.title)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    strip.save(args.out)
    print(f"Лента: {args.out} ({strip.width}×{strip.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
