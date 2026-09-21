"""Передать рабочую конфигурацию спектров в репозиторий HyperSpectRus.

Конфигурацию подбирают вместе с алгоритмом, поэтому её исходник хранится в
этом репозитории. В монорепозиторий приложения попадает точная копия:

    uv run python tools/sync_reference.py

Путь к приложению берётся из переменной ``HYPERSPECTRUS_PATH`` или считается
соседним каталогом ``../hyperspectrus``; аргументом можно передать любой другой.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from mirror import target_argument

REFERENCE = Path("references/hsr-example.json")

#: Прежние имена файла. Остаться в приложении они не должны: рабочее место
#: загрузит любой справочник из каталога, и два файла одной модели - это два
#: набора коэффициентов, между которыми никто не выбирал.
LEGACY_REFERENCES = ("hsr-test-8.json", "hsr-main-reference.json")


def main() -> int:
    """Проверить конфигурацию и скопировать её в репозиторий приложения."""
    target_root = target_argument("Передать основную конфигурацию спектров")
    source = Path(__file__).resolve().parent.parent / REFERENCE
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
