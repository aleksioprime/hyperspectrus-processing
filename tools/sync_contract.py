"""Обновить зеркало контракта из монорепозитория HyperSpectRus.

Контракт `hsr-proc-base` разрабатывается в монорепозитории комплекса, а сюда
попадает копией: иначе сторонний разработчик алгоритма не смог бы даже
импортировать `hsr_proc`. Копия делается скриптом, а не руками, потому что
расхождение контракта, по которому пишут алгоритм, и контракта, по которому
работает рабочее место, обнаружилось бы только на приёмке готового пакета.

    uv run python tools/sync_contract.py ../hyperspectrus

Скрипт записывает `MIRROR.json` с коммитом-источником и суммами файлов. Тест
`tests/test_mirror.py` сверяет суммы: правка зеркала прямо здесь сразу красит
проверку, потому что вносить её нужно в монорепозиторий.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

#: Путь пакета контракта внутри обоих репозиториев.
PACKAGE = Path("packages/hsr-proc-base")

#: Файл с описанием зеркала.
MIRROR_FILE = "MIRROR.json"

#: Что не копируется: следы работы инструментов.
EXCLUDED_DIRS = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}

#: Ссылки на документацию монорепозитория, которых здесь нет.
LINK_REWRITES = (
    (
        "[docs/processing.md](../../docs/processing.md)",
        "[README.md](../../README.md#контракт-обработки)",
    ),
    ("``docs/processing.md``", "``README.md``"),
    ("`docs/processing.md`", "`README.md`"),
    (
        "Расчёт живёт в `packages/hsr-proc-algo` и подключается по тем же правилам, "
        "что и любая другая реализация. Сейчас там эталонная.\n\n",
        "",
    ),
)

#: Врезка, которую скрипт ставит в README зеркала после заголовка.
NOTICE = (
    "> **Зеркало.** Каталог собран скриптом `tools/sync_contract.py` из монорепозитория\n"
    "> комплекса. Правки вносятся там; всё, что изменено здесь, будет затёрто при\n"
    "> следующем обновлении, а до того уронит проверку `tests/test_mirror.py`.\n"
)


def main() -> int:
    """Собрать зеркало и записать его описание."""
    parser = argparse.ArgumentParser(description="Обновить зеркало контракта обработки")
    parser.add_argument("source", type=Path, help="путь к монорепозиторию hyperspectrus")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    target = Path(__file__).resolve().parent.parent
    source_package = source / PACKAGE

    if not source_package.is_dir():
        print(f"не найден пакет контракта: {source_package}", file=sys.stderr)
        return 1

    destination = target / PACKAGE
    if destination.exists():
        shutil.rmtree(destination)
    _copy(source_package, destination)
    _rewrite_links(destination)
    _add_notice(destination / "README.md")

    mirror = {
        "источник": "packages/hsr-proc-base",
        "коммит": _commit(source),
        "собрано": datetime.now(UTC).strftime("%Y-%m-%d"),
        "файлы": _checksums(destination),
    }
    (destination / MIRROR_FILE).write_text(
        json.dumps(mirror, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"зеркало обновлено: {len(mirror['файлы'])} файлов из {mirror['коммит']}")
    return 0


def _copy(source: Path, destination: Path) -> None:
    """Скопировать пакет, пропустив каталоги инструментов."""
    shutil.copytree(
        source,
        destination,
        ignore=lambda _directory, names: {name for name in names if name in EXCLUDED_DIRS},
    )


def _rewrite_links(destination: Path) -> None:
    """Заменить ссылки на документацию монорепозитория на здешнюю."""
    for path in sorted(destination.rglob("*")):
        if path.suffix not in {".py", ".md", ".toml"} or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text
        for old, new in LINK_REWRITES:
            updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")


def _add_notice(readme: Path) -> None:
    """Поставить в README зеркала врезку о том, где правится оригинал."""
    if not readme.is_file():
        return
    lines = readme.read_text(encoding="utf-8").splitlines(keepends=True)
    head = 1 if lines and lines[0].startswith("# ") else 0
    body = "".join(lines[head:]).lstrip("\n")
    readme.write_text("".join(lines[:head]) + "\n" + NOTICE + "\n" + body, encoding="utf-8")


def _checksums(destination: Path) -> dict[str, str]:
    """Посчитать суммы файлов зеркала."""
    sums: dict[str, str] = {}
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.name == MIRROR_FILE:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums[path.relative_to(destination).as_posix()] = digest
    return sums


def _commit(source: Path) -> str:
    """Вернуть коммит монорепозитория, из которого собрано зеркало."""
    try:
        result = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        # Зеркало собирается из рабочего каталога, а не из коммита. Если в нём
        # есть несохранённые правки, записать голый номер коммита значило бы
        # соврать: по нему потом не восстановить, что именно скопировано.
        dirty = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", str(PACKAGE)],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "неизвестен"

    commit = result.stdout.strip()
    return f"{commit} + несохранённые правки" if dirty.stdout.strip() else commit


if __name__ == "__main__":
    sys.exit(main())
