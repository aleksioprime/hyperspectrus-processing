"""Тесты замены основного расчёта расчётом-победителем.

Операция редкая и разрушительная: пакет перезаписывается, источник удаляется.
Забыть при этом можно многое - версию, зависимость, имя в тестах, - и каждая
пропажа обнаружилась бы уже на приёмке или, хуже, у врача.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> ModuleType:
    """Загрузить команду из ``tools`` по пути."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new_algo = _load("new_algo")
promote_algo = _load("promote_algo")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Собрать подобие репозитория с основным расчётом и расчётом-соседом."""
    shutil.copytree(
        ROOT / "packages" / "hsr-proc-algo",
        tmp_path / "packages" / "hsr-proc-algo",
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"),
    )
    (tmp_path / "pyproject.toml").write_text(
        '[dependency-groups]\ndev = [\n    "hsr-proc-algo",\n]\n\n'
        "[tool.uv.sources]\nhsr-proc-algo = { workspace = true }\n",
        encoding="utf-8",
    )
    new_algo.create("demo", root=tmp_path, blank=False)
    return tmp_path


def test_расчёт_становится_основным(workspace: Path) -> None:
    """Имена основного пакета возвращаются на место, источник исчезает."""
    примета = "# расчёт-победитель\n"
    demo = workspace / "packages" / "hsr-proc-demo" / "src" / "hsr_proc_demo" / "processor.py"
    demo.write_text(примета + demo.read_text(encoding="utf-8"), encoding="utf-8")

    promote_algo.promote("demo", version="1.1", root=workspace)

    processor = (
        workspace / "packages" / "hsr-proc-algo" / "src" / "hsr_proc_algo" / "processor.py"
    ).read_text(encoding="utf-8")
    assert примета in processor
    assert "class AlgoProcessor:" in processor
    assert '    name = "algo"' in processor
    assert '    version = "1.1"' in processor
    assert not (workspace / "packages" / "hsr-proc-demo").exists()


def test_пакет_источник_отключается_от_workspace(workspace: Path) -> None:
    """Иначе uv будет искать каталог, которого уже нет."""
    promote_algo.promote("demo", version="1.1", root=workspace)

    описание = (workspace / "pyproject.toml").read_text(encoding="utf-8")
    assert "hsr-proc-demo" not in описание
    assert '"hsr-proc-algo",' in описание


def test_зависимости_расчёта_переносятся(workspace: Path) -> None:
    """Забытая зависимость превратилась бы в ImportError на приёмке."""
    описание = workspace / "packages" / "hsr-proc-demo" / "pyproject.toml"
    описание.write_text(
        описание.read_text(encoding="utf-8").replace(
            '    "numpy>=2.1",', '    "numpy>=2.1",\n    "scipy>=1.14",'
        ),
        encoding="utf-8",
    )

    promote_algo.promote("demo", version="1.1", root=workspace)

    основной = (workspace / "packages" / "hsr-proc-algo" / "pyproject.toml").read_text("utf-8")
    assert "scipy>=1.14" in основной
    assert 'version = "1.1.0"' in основной


def test_имя_и_версия_в_тестах_обновляются(workspace: Path) -> None:
    """Проверка подписи результата иначе осталась бы требовать прежние."""
    promote_algo.promote("demo", version="1.1", root=workspace)

    тесты = workspace / "packages" / "hsr-proc-algo" / "tests"
    подпись = (тесты / "test_processor.py").read_text(encoding="utf-8")
    assert '"algo 1.1"' in подпись
    assert 'get_processor("algo")' in подпись
    assert not list(тесты.glob("test_demo_*.py"))


def test_версия_назад_не_идёт(workspace: Path) -> None:
    """Версия попадает в карточку сеанса: убывание перепутало бы выпуски."""
    with pytest.raises(ValueError, match="не больше нынешней"):
        promote_algo.promote("demo", version="0.9", root=workspace)

    assert (workspace / "packages" / "hsr-proc-demo").exists()


def test_неизвестный_расчёт_отклоняется(workspace: Path) -> None:
    """Опечатка в имени не должна снести основной пакет."""
    with pytest.raises(ValueError, match="не найден пакет"):
        promote_algo.promote("нетакого", version="1.1", root=workspace)
