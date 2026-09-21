"""Команда ``hsr-proc-run``: прогон алгоритма на кадрах без рабочего места.

Проверки говорят, соответствует ли алгоритм контракту. Глазами результат
смотрят здесь: команда собирает куб из каталога с кадрами, зовёт алгоритм и
кладёт рядом карты в PNG и показатели в JSON.

    uv run hsr-proc-run --synthetic --out out
    uv run hsr-proc-run ../datasets/real/jpeg --reference ../references/hsr-example.json

Кадры раскладываются так же, как их пишет прибор: ``<каталог>/jpeg/1/450nm.jpg``
или просто ``<каталог>/450nm.jpg``. Файлы DNG не читаются - для них алгоритму
нужны собственные зависимости.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from hsr_proc.errors import ProcessingError
from hsr_proc.loading import discover_frames, load_cube
from hsr_proc.models import (
    Chromophore,
    OverlapMatrix,
    ProcessingParams,
    ProcessingRequest,
    ProcessingResult,
    SpectralCube,
)
from hsr_proc.registry import get_processor
from PIL import Image

from . import data


def main(argv: list[str] | None = None) -> int:
    """Прогнать алгоритм на кадрах и разложить результат по файлам."""
    parser = argparse.ArgumentParser(
        prog="hsr-proc-run",
        description="Прогон алгоритма обработки на серии кадров",
    )
    parser.add_argument("series", nargs="?", type=Path, help="каталог с кадрами серии")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="взять синтетическую серию с известным ответом вместо каталога",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        help="файл справочника: длины волн, хромофоры и коэффициенты перекрытия",
    )
    parser.add_argument("--out", type=Path, default=Path("out"), help="куда сложить результат")
    parser.add_argument("--sigma", type=float, default=1.0, help="сглаживание перед выделением")
    parser.add_argument("--processor", help="имя реализации, если подключено несколько")
    args = parser.parse_args(argv)

    if not args.synthetic and args.series is None:
        parser.error("укажите каталог с кадрами или --synthetic")

    try:
        cube = _cube(args)
        matrix = _matrix(args.reference, cube)
        processor = get_processor(args.processor)
    except (ProcessingError, OSError, ValueError) as error:
        print(f"Не удалось подготовить данные: {error}", file=sys.stderr)
        return 1

    print(f"Реализация: {processor.name} {processor.version}")
    print(f"Серия: {cube.count} спектров, кадр {cube.width}×{cube.height}")

    request = ProcessingRequest(
        cube=cube, overlap=matrix, params=ProcessingParams(gaussian_sigma=args.sigma)
    )
    started = time.perf_counter()
    try:
        result = processor.process(request, _show)
    except ProcessingError as error:
        print(f"\n{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started

    _save(result, args.out, elapsed)
    print(f"\nГотово за {elapsed:.1f} с. Результат: {args.out}")
    return 0


def _show(percent: int, message: str) -> None:
    """Показать ход обработки одной строкой."""
    print(f"\r{percent:3d}% {message:<50}", end="", flush=True)


def _cube(args: argparse.Namespace) -> SpectralCube:
    """Собрать куб из каталога с кадрами или из синтетической серии."""
    if args.synthetic:
        print("Серия синтетическая: концентрации известны заранее")
        return data.reference_case().request.cube

    series: Path = args.series
    nested = series / "jpeg" / "1"
    return load_cube(discover_frames(nested if nested.is_dir() else series))


def _matrix(path: Path | None, cube: SpectralCube) -> OverlapMatrix:
    """Прочитать справочник или взять синтетический."""
    if path is None:
        print("Справочник не задан: взяты синтетические коэффициенты из набора проверок")
        return data.overlap(wavelengths=cube.wavelengths_nm[: len(data.WAVELENGTHS)])

    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("format") != "hyperspectrus-reference":
        raise ValueError("это не файл справочника HyperSpectRus")

    models = document.get("models")
    if not isinstance(models, list) or len(models) != 1:
        raise ValueError("файл справочника должен содержать ровно одну модель прибора")

    chromophores = tuple(
        Chromophore(symbol=item["symbol"], name=item.get("name", ""))
        for item in document["chromophores"]
    )
    spectra = models[0]["spectra"]
    wavelengths = tuple(int(item["wavelength_nm"]) for item in spectra)
    try:
        values = [
            [float(item["overlaps"][chromophore.symbol]) for chromophore in chromophores]
            for item in spectra
        ]
    except (KeyError, TypeError) as error:
        raise ValueError("матрица перекрытий в справочнике заполнена не полностью") from error

    missing = sorted(set(cube.wavelengths_nm) - set(wavelengths))
    if missing:
        raise ValueError(f"в справочнике нет длин волн из серии: {missing}")

    return OverlapMatrix(
        wavelengths_nm=wavelengths,
        chromophores=chromophores,
        values=np.asarray(values, dtype=np.float32),
    )


def _save(result: ProcessingResult, out: Path, elapsed: float) -> None:
    """Разложить результат по файлам: карты в PNG, показатели в JSON."""
    out.mkdir(parents=True, exist_ok=True)

    for symbol, values in result.concentrations.items():
        _save_map(values, out / f"{symbol}.png")
    _save_map(result.thb_map, out / "thb.png")
    Image.fromarray((result.lesion_mask * 255).astype(np.uint8)).save(out / "mask.png")

    summary = {
        "реализация": result.processor,
        "секунд": round(elapsed, 2),
        "показатели": {
            "s": result.metrics.s_coefficient,
            "средний THb в поражении": result.metrics.mean_lesion_thb,
            "средний THb в коже": result.metrics.mean_skin_thb,
        },
        "выделение": {
            "способ": result.segmentation.method,
            "порог": result.segmentation.threshold,
            "сглаживание": result.segmentation.gaussian_sigma,
        },
        "пояснение": result.notes,
        "карты": sorted(result.concentrations),
    }
    (out / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _save_map(values: np.ndarray, path: Path) -> None:
    """Сохранить карту в оттенках серого, растянув значения на весь диапазон.

    Растяжение нужно для глаза: абсолютные значения концентраций малы, и без
    него карта выглядит чёрным квадратом. Числа берутся из ``metrics.json``.
    """
    finite = values[np.isfinite(values)]
    low = float(finite.min()) if finite.size else 0.0
    high = float(finite.max()) if finite.size else 1.0
    span = high - low if high > low else 1.0
    levels = np.clip((values - low) / span, 0.0, 1.0) * 255.0
    Image.fromarray(levels.astype(np.uint8)).save(path)


if __name__ == "__main__":
    sys.exit(main())
