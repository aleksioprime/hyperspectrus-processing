"""Тесты команды ``python -m hsr_proc``.

Команда отвечает на единственный вопрос: видит ли приложение алгоритм. Её
запускают, когда что-то не так, поэтому падать она не имеет права - в том числе
когда реализаций несколько и ни одна не выбрана.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from hsr_proc import registry
from hsr_proc.__main__ import main
from hsr_proc.contract import ProgressCallback
from hsr_proc.models import ProcessingRequest, ProcessingResult
from hsr_proc.registry import PROCESSOR_ENV


class FakeProcessor:
    """Реализация-пустышка: для списка важны только имя и версия."""

    def __init__(self, name: str, version: str = "0.1") -> None:
        """Создать реализацию с заданным именем."""
        self.name = name
        self.version = version

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработка не выполняется: реализация нужна только для реестра."""
        raise NotImplementedError


@pytest.fixture(autouse=True)
def only_registered(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Оставить в реестре только то, что зарегистрировано тестом."""
    monkeypatch.delenv(PROCESSOR_ENV, raising=False)
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: [])
    registry.refresh()
    yield
    for name in list(registry.available()):
        registry.unregister(name)
    registry.refresh()


def test_единственная_реализация_отмечена_выбранной(capsys: pytest.CaptureFixture[str]) -> None:
    """Обычный случай: один алгоритм, и он же будет считать."""
    registry.register(FakeProcessor("algo", "1.0"))

    assert main() == 0

    output = capsys.readouterr().out
    assert "→ algo 1.0" in output
    assert "Выбрана: algo 1.0" in output


def test_несколько_реализаций_не_роняют_команду(capsys: pytest.CaptureFixture[str]) -> None:
    """Два расчёта рядом - рабочее состояние при сравнении.

    Прежде команда падала с трассировкой именно здесь, хотя список реализаций
    в этот момент и нужен: разработчик как раз выясняет, что подключено.
    """
    registry.register(FakeProcessor("algo", "1.0"))
    registry.register(FakeProcessor("newalgo", "0.1"))

    assert main() == 0

    output = capsys.readouterr().out
    assert "algo 1.0" in output
    assert "newalgo 0.1" in output
    assert "Выбор не сделан" in output
    assert PROCESSOR_ENV in output


def test_без_реализаций_команда_говорит_что_делать(capsys: pytest.CaptureFixture[str]) -> None:
    """Пустой список сам по себе ничего не объясняет."""
    assert main() == 1

    assert "установите пакет с алгоритмом" in capsys.readouterr().out
