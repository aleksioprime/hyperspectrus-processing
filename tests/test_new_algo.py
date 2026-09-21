"""Тесты команды, создающей пакет для ещё одного расчёта.

Команда нужна редко, и ошибиться в ней легко незаметно: пакет собирается, а
подключается неправильно. Поэтому проверяется и раскладка, и то, что пустой
расчёт из неё проходит обязательный уровень приёмки - иначе заготовка отстанет
от контракта ровно так же, как отстала бы отдельная папка с образцом.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest
from hsr_proc_conformance import ProcessorConformance, data

ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, name: str) -> ModuleType:
    """Загрузить модуль по пути, не устанавливая пакет."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


new_algo = _load(ROOT / "tools" / "new_algo.py", "new_algo")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """Собрать подобие репозитория: действующий пакет и корневой pyproject."""
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
    return tmp_path


def test_пакет_собирается_под_новым_именем(workspace: Path) -> None:
    """Переименованы должны быть все четыре имени сразу."""
    package = new_algo.create("newalgo", root=workspace, blank=False)

    assert (package / "src" / "hsr_proc_newalgo" / "processor.py").is_file()
    assert not (package / "src" / "hsr_proc_algo").exists()

    description = (package / "pyproject.toml").read_text(encoding="utf-8")
    assert 'name = "hsr-proc-newalgo"' in description
    assert 'newalgo = "hsr_proc_newalgo:NewalgoProcessor"' in description
    assert 'version = "0.1.0"' in description

    processor = (package / "src" / "hsr_proc_newalgo" / "processor.py").read_text(encoding="utf-8")
    assert "class NewalgoProcessor:" in processor
    assert '    name = "newalgo"' in processor


def test_имена_тестов_не_сталкиваются_с_прежними(workspace: Path) -> None:
    """Два `test_processor.py` в разных пакетах не соберутся вместе."""
    package = new_algo.create("newalgo", root=workspace, blank=False)

    names = {path.name for path in (package / "tests").glob("*.py")}
    assert "test_processor.py" not in names
    assert "test_conformance.py" not in names
    assert "test_newalgo_processor.py" in names
    assert "test_newalgo_conformance.py" in names


def test_пакет_подключается_к_workspace(workspace: Path) -> None:
    """Без двух строк в корневом файле пакет не поставится."""
    new_algo.create("newalgo", root=workspace, blank=False)

    description = (workspace / "pyproject.toml").read_text(encoding="utf-8")
    assert '"hsr-proc-newalgo",' in description
    assert "hsr-proc-newalgo = { workspace = true }" in description


def test_повторное_создание_отклоняется(workspace: Path) -> None:
    """Молча затереть чужой пакет команда не должна."""
    new_algo.create("newalgo", root=workspace, blank=False)

    with pytest.raises(FileExistsError):
        new_algo.create("newalgo", root=workspace, blank=False)


def test_пустой_расчёт_проходит_обязательный_уровень(workspace: Path) -> None:
    """Заготовка обязана оставаться пригодной для рабочего места.

    Проверка идёт тем же набором, что и приёмка: если в контракт добавится
    обязательное поле, тест упадёт здесь, а не у того, кто возьмёт заготовку за
    основу.
    """
    package = new_algo.create("blank", root=workspace, blank=True)
    module = _load(package / "src" / "hsr_proc_blank" / "processor.py", "hsr_proc_blank_processor")
    processor = module.BlankProcessor()

    случай = data.reference_case()
    результат = processor.process(случай.request)
    проверки = type("Проверки", (ProcessorConformance,), {"processor": processor})()

    проверки.test_реализация_соответствует_протоколу(processor)
    проверки.test_имя_и_версия_объявлены(processor)
    проверки.test_результат_заполнен_целиком(результат, случай)
    проверки.test_результат_подписан_именем_и_версией(processor, результат)
    проверки.test_ход_обработки_сообщается(processor)
    проверки.test_вход_не_изменяется(processor)
