"""Создать пакет для ещё одного расчёта.

Сравнивать два расчёта удобнее, когда подключены оба. Новый пакет делается из
действующего, а не из отдельной заготовки: заготовка отстала бы от контракта
молча, и обнаружилось бы это на первом же пакете, собранном по ней.

    uv run python tools/new_algo.py newalgo            # копия действующего расчёта
    uv run python tools/new_algo.py newalgo --blank    # пустой расчёт с пометками ЗДЕСЬ

Команда переименовывает пакет, модуль, класс и точку входа, разводит имена
файлов тестов и дописывает две строки в корневой ``pyproject.toml``. Остаётся
выполнить ``uv sync``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

#: Пакет, из которого делается новый.
SOURCE = Path("packages/hsr-proc-algo")

#: Имя расчёта: строчные латинские буквы, цифры и дефис.
NAME = re.compile(r"^[a-z][a-z0-9-]*$")

#: Следы работы инструментов, которые в новый пакет не копируются.
EXCLUDED = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", "MIRROR.json"}

#: Тесты действующего расчёта: проверяют именно его, новому не подходят.
CALCULATION_TESTS = ("test_processor.py", "test_segmentation.py")

#: Пустой расчёт: возвращает карты правильного вида и ничего не считает.
BLANK = '''"""Реализация обработки.

Пустой расчёт возвращает карты правильного вида: проверки уровня ``contract``
проходят, числовые - нет, и так и должно быть, пока расчёта нет. Ищите пометки
ЗДЕСЬ.
"""

from __future__ import annotations

import numpy as np
from hsr_proc import (
    BoolMask,
    FloatMap,
    ProcessingMetrics,
    ProcessingRequest,
    ProcessingResult,
    ProgressCallback,
    SegmentationInfo,
    report,
)


class {cls}:
    """Восстановление концентраций хромофоров."""

    #: Имя, по которому реализацию выбирают при прогоне и в приёмке.
    name = "{name}"

    #: Поднимается при любом изменении расчёта.
    version = "0.1"

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты концентраций и показатели."""
        cube = request.cube

        # Порядок кадров задаёт прибор, порядок строк матрицы - справочник.
        matrix = request.overlap.aligned_to(cube)
        report(progress, 10, "Подготовка данных")

        # ЗДЕСЬ: перевод яркостей в оптическую плотность и решение системы.
        shape = (cube.height, cube.width)
        empty: FloatMap = np.zeros(shape, dtype=np.float32)
        concentrations = {{symbol: empty.copy() for symbol in request.overlap.symbols}}
        report(progress, 50, "Расчёт концентраций")

        # ЗДЕСЬ: карта общей концентрации гемоглобина.
        thb_map: FloatMap = np.zeros(shape, dtype=np.float32)

        # ЗДЕСЬ: выделение области поражения по карте THb.
        lesion: BoolMask = np.zeros(shape, dtype=np.bool_)
        report(progress, 80, "Выделение области поражения")

        # ЗДЕСЬ: средние концентрации и их отношение. Деления на ноль быть не
        # должно: на однородном кадре кожи может не оказаться вовсе.
        metrics = ProcessingMetrics(
            s_coefficient=0.0,
            mean_lesion_thb=0.0,
            mean_skin_thb=0.0,
        )

        report(progress, 100, "Обработка завершена")
        return ProcessingResult(
            concentrations=concentrations,
            thb_map=thb_map,
            lesion_mask=lesion,
            metrics=metrics,
            segmentation=SegmentationInfo(
                method="не реализовано",
                threshold=0.0,
                gaussian_sigma=request.params.gaussian_sigma,
            ),
            processor=f"{{self.name}} {{self.version}}",
            notes=f"Расчёт не реализован ({{matrix.shape[1]}} хромофоров)",
        )
'''


def main(argv: list[str] | None = None) -> int:
    """Создать пакет нового расчёта рядом с действующим."""
    parser = argparse.ArgumentParser(
        prog="new_algo.py", description="Создать пакет для ещё одного расчёта"
    )
    parser.add_argument("name", help="имя расчёта, например newalgo")
    parser.add_argument(
        "--blank",
        action="store_true",
        help="пустой расчёт вместо копии действующего",
    )
    args = parser.parse_args(argv)

    if not NAME.match(args.name):
        print(
            f"имя {args.name!r} не годится: строчные латинские буквы, цифры и дефис",
            file=sys.stderr,
        )
        return 1

    root = Path(__file__).resolve().parent.parent
    try:
        package = create(args.name, root=root, blank=args.blank)
    except FileExistsError as error:
        print(error, file=sys.stderr)
        return 1

    print(f"создан пакет: {package.relative_to(root)}")
    print("дальше:")
    print("  uv sync")
    print(f"  uv run hsr-proc-check hsr-proc-{args.name}")
    if args.blank:
        print()
        print("Расчёта в пакете нет, поэтому числовые проверки на нём падают.")
        print(f"Пока он не написан: uv run pytest packages/hsr-proc-{args.name} -m contract")
    return 0


def create(name: str, *, root: Path, blank: bool) -> Path:
    """Собрать пакет нового расчёта и подключить его к workspace."""
    module = f"hsr_proc_{name.replace('-', '_')}"
    cls = "".join(part.capitalize() for part in name.split("-")) + "Processor"
    package = root / "packages" / f"hsr-proc-{name}"
    if package.exists():
        raise FileExistsError(f"каталог уже существует: {package}")

    shutil.copytree(
        root / SOURCE,
        package,
        ignore=lambda _directory, names: {item for item in names if item in EXCLUDED},
    )
    (package / "src" / "hsr_proc_algo").rename(package / "src" / module)

    _write_pyproject(package / "pyproject.toml", name=name, module=module, cls=cls)
    _write_sources(package, name=name, module=module, cls=cls, blank=blank)
    _write_tests(package, name=name, module=module, cls=cls, blank=blank)
    _connect(root / "pyproject.toml", name=name)
    return package


def _write_pyproject(path: Path, *, name: str, module: str, cls: str) -> None:
    """Переписать описание пакета под новое имя."""
    text = path.read_text(encoding="utf-8")
    text = text.replace('name = "hsr-proc-algo"', f'name = "hsr-proc-{name}"')
    text = text.replace(
        'algo = "hsr_proc_algo:AlgoProcessor"',
        f'{name.replace("-", "_")} = "{module}:{cls}"',
    )
    text = text.replace('packages = ["src/hsr_proc_algo"]', f'packages = ["src/{module}"]')
    # Версия начинает собственный счёт: это другой расчёт, а не выпуск прежнего.
    text = re.sub(r'^version = "[^"]+"', 'version = "0.1.0"', text, count=1, flags=re.MULTILINE)
    path.write_text(text, encoding="utf-8")


def _write_sources(package: Path, *, name: str, module: str, cls: str, blank: bool) -> None:
    """Переименовать реализацию и, если просили, очистить расчёт."""
    source = package / "src" / module
    processor = source / "processor.py"
    if blank:
        processor.write_text(BLANK.format(cls=cls, name=name), encoding="utf-8")
        (source / "segmentation.py").unlink(missing_ok=True)
    else:
        text = processor.read_text(encoding="utf-8")
        text = text.replace("class AlgoProcessor:", f"class {cls}:")
        text = text.replace('    name = "algo"', f'    name = "{name}"')
        text = re.sub(r'^    version = "[^"]+"', '    version = "0.1"', text, flags=re.MULTILINE)
        processor.write_text(text, encoding="utf-8")

    init = source / "__init__.py"
    text = init.read_text(encoding="utf-8")
    text = text.replace("from .processor import AlgoProcessor", f"from .processor import {cls}")
    text = text.replace('__all__ = ["AlgoProcessor"]', f'__all__ = ["{cls}"]')
    init.write_text(text, encoding="utf-8")


def _write_tests(package: Path, *, name: str, module: str, cls: str, blank: bool) -> None:
    """Развести имена тестов и убрать проверки чужого расчёта.

    pytest импортирует тесты как модули верхнего уровня, и два ``test_processor``
    в разных пакетах не соберутся вместе.
    """
    tests = package / "tests"
    for file_name in CALCULATION_TESTS:
        path = tests / file_name
        if not path.is_file():
            continue
        if blank:
            path.unlink()
            continue
        renamed = tests / f"test_{name.replace('-', '_')}_{file_name.removeprefix('test_')}"
        text = path.read_text(encoding="utf-8")
        text = _rename(text, module=module, cls=cls)
        text = text.replace("AlgoProcessor()", f"{cls}()")
        text = text.replace('get_processor("algo")', f'get_processor("{name}")')
        text = text.replace('"algo 1.0"', f'"{name} 0.1"')
        path.write_text(text, encoding="utf-8")
        path.rename(renamed)

    conformance = tests / "test_conformance.py"
    text = _rename(conformance.read_text(encoding="utf-8"), module=module, cls=cls)
    text = text.replace('"hsr-proc-algo"', f'"hsr-proc-{name}"')
    conformance.write_text(text, encoding="utf-8")
    conformance.rename(tests / f"test_{name.replace('-', '_')}_conformance.py")


def _rename(text: str, *, module: str, cls: str) -> str:
    """Заменить в тексте имя модуля и класса действующего расчёта на новые."""
    text = text.replace("from hsr_proc_algo import AlgoProcessor", f"from {module} import {cls}")
    return text.replace("AlgoProcessor()", f"{cls}()")


def _connect(path: Path, *, name: str) -> None:
    """Подключить пакет к workspace: группа dev и источник."""
    text = path.read_text(encoding="utf-8")
    package = f"hsr-proc-{name}"
    if package in text:
        return
    text = text.replace(
        '    "hsr-proc-algo",\n',
        f'    "hsr-proc-algo",\n    "{package}",\n',
        1,
    )
    text = text.replace(
        "hsr-proc-algo = { workspace = true }\n",
        f"hsr-proc-algo = {{ workspace = true }}\n{package} = {{ workspace = true }}\n",
        1,
    )
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
