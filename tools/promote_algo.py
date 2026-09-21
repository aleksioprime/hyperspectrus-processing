"""Сделать расчёт из соседнего пакета основным.

Расчёты сравнивают рядом: `tools/new_algo.py` заводит второй пакет, и работа
идёт в нём. Когда он выигрывает, основным становится он - но уезжает в
приложение по-прежнему `hsr-proc-algo`, потому что имя `algo` записано в
настройках рабочего места.

    uv run python tools/promote_algo.py newalgo --version 1.1

Команда переносит расчёт, тесты, зависимости и допуски приёмки в
`hsr-proc-algo`, возвращает имя класса и реализации, ставит новую версию и
убирает пакет-источник. Версия называется явно: она попадает в карточку сеанса,
и выбирать её должен человек.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

#: Пакет, который уезжает в приложение.
TARGET = Path("packages/hsr-proc-algo")

#: Имена в основном пакете. Они согласованы с рабочим местом.
TARGET_MODULE = "hsr_proc_algo"
TARGET_CLASS = "AlgoProcessor"
TARGET_NAME = "algo"

#: Версия расчёта: две или три цифры через точку.
VERSION = re.compile(r"^\d+\.\d+(\.\d+)?$")

#: Следы работы инструментов.
EXCLUDED = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def main(argv: list[str] | None = None) -> int:
    """Перенести расчёт из названного пакета в основной."""
    parser = argparse.ArgumentParser(
        prog="promote_algo.py", description="Сделать расчёт из соседнего пакета основным"
    )
    parser.add_argument("name", help="имя расчёта-победителя, например newalgo")
    parser.add_argument(
        "--version",
        required=True,
        help="версия основного расчёта после замены, например 1.1",
    )
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    try:
        _require_clean(root)
        promote(args.name, version=args.version, root=root)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"расчёт {args.name} перенесён в hsr-proc-algo {args.version}")
    print("дальше:")
    print("  uv sync")
    print("  uv run hsr-proc-check hsr-proc-algo --report приёмка.md")
    print("  git add -A && git commit")
    return 0


def promote(name: str, *, version: str, root: Path) -> None:
    """Заменить содержимое основного пакета расчётом из пакета ``name``."""
    _check_version(version, root=root)
    module = f"hsr_proc_{name.replace('-', '_')}"
    source = root / "packages" / f"hsr-proc-{name}"
    target = root / TARGET
    if not (source / "src" / module).is_dir():
        raise ValueError(f"не найден пакет расчёта: {source}")
    if source == target:
        raise ValueError("этот расчёт уже основной")

    cls = _class_name(source / "src" / module / "processor.py")
    was = _current_version(source / "src" / module / "processor.py")
    _replace_sources(source / "src" / module, target / "src" / TARGET_MODULE, cls=cls, name=name)
    _replace_tests(
        source / "tests",
        target / "tests",
        cls=cls,
        name=name,
        module=module,
        was=was,
        version=version,
    )
    _merge_pyproject(source / "pyproject.toml", target / "pyproject.toml", version=version)
    _set_version(target / "src" / TARGET_MODULE / "processor.py", version=version)

    shutil.rmtree(source)
    _disconnect(root / "pyproject.toml", name=name)


def _require_clean(root: Path) -> None:
    """Отказаться работать, если есть незафиксированные правки в пакетах.

    Замена перезаписывает и удаляет файлы: без чистого дерева отличить её
    результат от собственных правок и откатить будет нечем. Смотрим только на
    то, что команда трогает: `uv.lock` пересобирается сам, и требовать его
    чистоты значило бы отказывать после любого `uv run`.
    """
    try:
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--", "packages", "pyproject.toml"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return
    if status.strip():
        raise ValueError(
            "в пакетах есть незафиксированные изменения: зафиксируйте их, "
            "иначе замену нельзя будет ни разглядеть, ни откатить"
        )


def _check_version(version: str, *, root: Path) -> None:
    """Проверить, что версия названа верно и больше нынешней."""
    if not VERSION.match(version):
        raise ValueError(f"версия {version!r} записывается как 1.1 или 1.1.0")

    current = _current_version(root / TARGET / "src" / TARGET_MODULE / "processor.py")
    if _digits(version) <= _digits(current):
        raise ValueError(
            f"версия {version} не больше нынешней {current}: по версии различают выпуски "
            "расчёта в карточке сеанса, и назад она не идёт"
        )


def _digits(version: str) -> tuple[int, ...]:
    """Разложить версию на числа для сравнения."""
    return tuple(int(part) for part in version.split("."))


def _current_version(processor: Path) -> str:
    """Прочитать версию нынешнего основного расчёта."""
    found = re.search(r'^    version = "([^"]+)"', processor.read_text(encoding="utf-8"), re.M)
    return found.group(1) if found else "0"


def _class_name(processor: Path) -> str:
    """Найти имя класса реализации в пакете-источнике."""
    found = re.search(r"^class (\w+):", processor.read_text(encoding="utf-8"), re.M)
    if found is None:
        raise ValueError(f"в {processor} не найден класс реализации")
    return found.group(1)


def _rewrite(text: str, *, cls: str, name: str, module: str | None = None) -> str:
    """Вернуть имена основного пакета на место."""
    if module is not None:
        text = text.replace(module, TARGET_MODULE)
    text = text.replace(cls, TARGET_CLASS)
    text = text.replace(f'name = "{name}"', f'name = "{TARGET_NAME}"')
    return text.replace(f"hsr-proc-{name}", "hsr-proc-algo")


def _replace_sources(source: Path, target: Path, *, cls: str, name: str) -> None:
    """Перенести модуль расчёта целиком."""
    shutil.rmtree(target)
    shutil.copytree(
        source, target, ignore=lambda _directory, names: {n for n in names if n in EXCLUDED}
    )
    for path in sorted(target.rglob("*.py")):
        path.write_text(_rewrite(path.read_text(encoding="utf-8"), cls=cls, name=name), "utf-8")


def _replace_tests(
    source: Path,
    target: Path,
    *,
    cls: str,
    name: str,
    module: str,
    was: str,
    version: str,
) -> None:
    """Перенести тесты, вернув им прежние имена файлов.

    Тесты прежнего расчёта проверяют то, чего больше нет, поэтому каталог
    заменяется целиком. Имя и версия в них тоже становятся основными: иначе
    проверка подписи результата останется требовать прежние.
    """
    shutil.rmtree(target)
    target.mkdir(parents=True)
    prefix = f"test_{name.replace('-', '_')}_"
    for path in sorted(source.iterdir()):
        if path.name in EXCLUDED or not path.is_file():
            continue
        renamed = f"test_{path.name[len(prefix) :]}" if path.name.startswith(prefix) else path.name
        text = _rewrite(path.read_text(encoding="utf-8"), cls=cls, name=name, module=module)
        short = ".".join(version.split(".")[:2])
        text = text.replace(f'get_processor("{name}")', f'get_processor("{TARGET_NAME}")')
        text = text.replace(f'"{name} {was}"', f'"{TARGET_NAME} {short}"')
        (target / renamed).write_text(text, encoding="utf-8")


def _merge_pyproject(source: Path, target: Path, *, version: str) -> None:
    """Перенести зависимости расчёта и поставить новую версию пакета."""
    block = re.compile(r"^dependencies = \[.*?^\]", re.M | re.S)
    dependencies = block.search(source.read_text(encoding="utf-8"))
    text = target.read_text(encoding="utf-8")
    if dependencies is not None:
        text = block.sub(dependencies.group(0), text, count=1)
    package = version if version.count(".") == 2 else f"{version}.0"
    text = re.sub(r'^version = "[^"]+"', f'version = "{package}"', text, count=1, flags=re.M)
    target.write_text(text, encoding="utf-8")


def _set_version(processor: Path, *, version: str) -> None:
    """Поставить версию реализации в классе."""
    short = ".".join(version.split(".")[:2])
    text = processor.read_text(encoding="utf-8")
    text = re.sub(r'^    version = "[^"]+"', f'    version = "{short}"', text, count=1, flags=re.M)
    processor.write_text(text, encoding="utf-8")


def _disconnect(path: Path, *, name: str) -> None:
    """Убрать пакет-источник из workspace."""
    package = f"hsr-proc-{name}"
    text = path.read_text(encoding="utf-8")
    text = text.replace(f'    "{package}",\n', "")
    text = text.replace(f"{package} = {{ workspace = true }}\n", "")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
