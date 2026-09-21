"""Передача пакета в репозиторий приложения зеркалом.

Контракт и алгоритм разрабатываются здесь, а работают в ``hyperspectrus``.
Копия делается скриптом, а не руками: расхождение между тем, по чему пишут
алгоритм, и тем, что стоит в приложении, обнаружилось бы только на сборке
рабочего места, а то и позже.

Рядом с копией пишется ``MIRROR.json``: коммит источника и контрольные суммы
файлов. Тест приложения сверяет их и не даёт править зеркало на месте.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

#: Имя файла с описанием зеркала.
MIRROR_FILE = "MIRROR.json"

#: Следы работы инструментов, которые в зеркало не попадают.
EXCLUDED_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def target_argument(description: str) -> Path:
    """Разобрать единственный аргумент команд передачи - путь к приложению."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("target", type=Path, help="путь к репозиторию hyperspectrus")
    target: Path = parser.parse_args().target
    return target.expanduser().resolve()


def transfer(package: Path, target_root: Path) -> int:
    """Собрать зеркало пакета в репозитории приложения.

    Возвращает код возврата команды: содержательные ошибки печатаются здесь,
    чтобы вызывающий скрипт оставался в три строки.
    """
    source_root = Path(__file__).resolve().parent.parent
    source = source_root / package
    destination = target_root / package

    if not source.is_dir():
        print(f"не найден исходный пакет: {source}", file=sys.stderr)
        return 1
    if not (target_root / "pyproject.toml").is_file() or not destination.parent.is_dir():
        print(f"это не репозиторий hyperspectrus: {target_root}", file=sys.stderr)
        return 1

    # Зеркало собирается рядом и подменяет прежнее одним движением: прерванная
    # передача не должна оставить приложение с половиной пакета.
    with tempfile.TemporaryDirectory(prefix=".hsr-mirror-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        _copy(source, staged)

        description = {
            "источник": f"hyperspectrus-processing/{package.as_posix()}",
            "коммит": _commit(source_root, package),
            "собрано": datetime.now(UTC).strftime("%Y-%m-%d"),
            "файлы": _checksums(staged),
        }
        (staged / MIRROR_FILE).write_text(
            json.dumps(description, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(staged), str(destination))

    print(f"передано: {destination} ({len(description['файлы'])} файлов)")
    return 0


def _copy(source: Path, destination: Path) -> None:
    """Скопировать пакет без кэшей и без описания прежнего зеркала."""
    shutil.copytree(
        source,
        destination,
        ignore=lambda _directory, names: {
            name for name in names if name in EXCLUDED_DIRS or name == MIRROR_FILE
        },
    )


def _checksums(directory: Path) -> dict[str, str]:
    """Посчитать суммы всех файлов зеркала, кроме его описания."""
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != MIRROR_FILE
    }


def _commit(repository: Path, package: Path) -> str:
    """Вернуть коммит источника и отметить несохранённые правки пакета.

    Зеркало собирается из рабочего каталога, а не из коммита. Записать голый
    номер коммита при несохранённых правках значило бы соврать: по нему потом
    не восстановить, что именно передано.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", str(package)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "неизвестен"

    commit = head.stdout.strip()
    return f"{commit} + несохранённые правки" if dirty.stdout.strip() else commit
