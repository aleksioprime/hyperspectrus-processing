"""Передать рабочую конфигурацию спектров в репозиторий HyperSpectRus.

Конфигурацию подбирают вместе с алгоритмом, поэтому её исходник хранится в
этом репозитории. В монорепозиторий приложения попадает точная копия:

    uv run python tools/sync_reference.py ../hyperspectrus
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REFERENCE = Path("references/hsr-example.json")

#: Прежние имена файла. Остаться в приложении они не должны: рабочее место
#: загрузит любой справочник из каталога, и два файла одной модели - это два
#: набора коэффициентов, между которыми никто не выбирал.
LEGACY_REFERENCES = ("hsr-test-8.json", "hsr-main-reference.json")


def main() -> int:
    """Проверить конфигурацию и скопировать её в репозиторий приложения."""
    parser = argparse.ArgumentParser(description="Передать основную конфигурацию спектров")
    parser.add_argument("target", type=Path, help="путь к репозиторию hyperspectrus")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    source = root / REFERENCE
    target_root = args.target.expanduser().resolve()
    destination_directory = target_root / "references"
    if not destination_directory.is_dir():
        print(f"не найден каталог справочников: {destination_directory}", file=sys.stderr)
        return 1

    document = json.loads(source.read_text(encoding="utf-8"))
    if document.get("format") != "hyperspectrus-reference":
        print(f"неверный формат конфигурации: {source}", file=sys.stderr)
        return 1

    destination = destination_directory / source.name
    shutil.copyfile(source, destination)
    for name in LEGACY_REFERENCES:
        legacy = destination_directory / name
        if legacy != destination and legacy.is_file():
            legacy.unlink()
    print(f"конфигурация обновлена: {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
