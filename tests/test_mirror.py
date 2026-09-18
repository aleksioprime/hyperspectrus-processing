"""Сверка зеркала контракта с монорепозиторием.

Контракт правится в монорепозитории комплекса, а сюда попадает копией. Правка,
внесённая прямо здесь, разошлась бы с тем, по чему работает рабочее место, и
обнаружилась бы на приёмке готового пакета - то есть после того, как алгоритм
уже написан по неверному описанию.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

#: Каталог зеркала и файл с его описанием.
MIRROR = Path(__file__).resolve().parent.parent / "packages" / "hsr-proc-base"
DESCRIPTION = MIRROR / "MIRROR.json"


@pytest.fixture(scope="module")
def описание() -> dict[str, object]:
    """Прочитать описание зеркала."""
    assert DESCRIPTION.is_file(), (
        f"нет {DESCRIPTION.name}: соберите зеркало командой "
        "`uv run python tools/sync_contract.py ../hyperspectrus`"
    )
    return json.loads(DESCRIPTION.read_text(encoding="utf-8"))


def test_зеркало_не_правили_на_месте(описание: dict[str, object]) -> None:
    """Файлы зеркала обязаны совпадать с тем, что собрал скрипт."""
    записанные = описание["файлы"]
    assert isinstance(записанные, dict)

    расхождения: list[str] = []
    for относительный, сумма in записанные.items():
        путь = MIRROR / относительный
        if not путь.is_file():
            расхождения.append(f"{относительный}: файла нет")
            continue
        if hashlib.sha256(путь.read_bytes()).hexdigest() != сумма:
            расхождения.append(f"{относительный}: содержимое изменено")

    assert not расхождения, (
        "зеркало контракта расходится с монорепозиторием: "
        + "; ".join(расхождения)
        + ". Правки вносятся в монорепозиторий, сюда зеркало пересобирается "
        "командой `uv run python tools/sync_contract.py ../hyperspectrus`"
    )


def test_в_зеркале_нет_лишних_файлов(описание: dict[str, object]) -> None:
    """Файл, добавленный в зеркало здесь, пропадёт при следующей пересборке."""
    записанные = описание["файлы"]
    assert isinstance(записанные, dict)

    найденные = {
        путь.relative_to(MIRROR).as_posix()
        for путь in MIRROR.rglob("*")
        if путь.is_file() and "__pycache__" not in путь.parts
    }
    лишние = sorted(найденные - set(записанные) - {DESCRIPTION.name})

    assert not лишние, f"файлы, которых нет в описании зеркала: {', '.join(лишние)}"


def test_известен_коммит_источника(описание: dict[str, object]) -> None:
    """По отчёту приёмки должно быть видно, по какому контракту писали алгоритм."""
    assert описание.get("коммит") not in (None, "", "неизвестен"), (
        "в описании зеркала нет коммита монорепозитория"
    )
