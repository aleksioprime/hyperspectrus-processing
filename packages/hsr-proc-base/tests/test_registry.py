"""Тесты реестра реализаций обработки."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from hsr_proc import registry
from hsr_proc.contract import ProgressCallback
from hsr_proc.errors import ProcessorNotFound
from hsr_proc.models import ProcessingRequest, ProcessingResult


class FakeProcessor:
    """Реализация-пустышка для проверки подключения."""

    def __init__(self, name: str = "fake", version: str = "0.1") -> None:
        """Создать реализацию с заданным именем."""
        self.name = name
        self.version = version

    def process(
        self, request: ProcessingRequest, progress: ProgressCallback | None = None
    ) -> ProcessingResult:
        """Обработка не выполняется: реализация нужна только для реестра."""
        raise NotImplementedError


@pytest.fixture(autouse=True)
def clean_registry() -> Iterator[None]:
    """Вернуть реестр в исходное состояние после теста."""
    yield
    for name in list(registry.available()):
        registry.unregister(name)
    registry.refresh()


def test_без_реализаций_обработка_недоступна(monkeypatch: pytest.MonkeyPatch) -> None:
    """Своей реализации у пакета нет: он описывает контракт, а не считает.

    Отказ должен быть внятным: молча посчитать нечем, и приложение обязано
    сказать об этом врачу.
    """
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: [])
    registry.refresh()

    with pytest.raises(ProcessorNotFound, match="установите пакет"):
        registry.get_processor()


def test_единственная_реализация_берётся_без_указания_имени(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Имён реализаций контракт не знает.

    Раньше имя по умолчанию было зашито строкой, и переименование алгоритма
    ломало бы выбор: обработка перестала бы работать до правки переменной
    окружения.
    """
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: [])
    registry.refresh()
    registry.register(FakeProcessor(name="какой-угодно"))

    assert registry.get_processor().name == "какой-угодно"


def test_из_нескольких_реализаций_выбирает_человек(monkeypatch: pytest.MonkeyPatch) -> None:
    """Молча взять первую попавшуюся значило бы посчитать сеанс неизвестно чем."""
    monkeypatch.setattr(registry.metadata, "entry_points", lambda **_: [])
    registry.refresh()
    registry.register(FakeProcessor(name="первый"))
    registry.register(FakeProcessor(name="второй"))

    with pytest.raises(ProcessorNotFound, match="выберите одну"):
        registry.get_processor()


def test_реализация_подключается_из_кода() -> None:
    """Прямая регистрация нужна для отладки и тестов."""
    registry.register(FakeProcessor())

    assert registry.get_processor("fake").version == "0.1"
    assert "fake" in registry.available()


def test_повторная_регистрация_требует_подтверждения() -> None:
    """Случайная подмена алгоритма другим пакетом недопустима."""
    registry.register(FakeProcessor())

    with pytest.raises(ValueError, match="уже зарегистрирована"):
        registry.register(FakeProcessor())

    registry.register(FakeProcessor(version="0.2"), replace=True)
    assert registry.get_processor("fake").version == "0.2"


def test_неизвестное_имя_называет_доступные() -> None:
    """Ошибка должна подсказывать, что именно можно выбрать."""
    registry.register(FakeProcessor())

    with pytest.raises(ProcessorNotFound, match="fake"):
        registry.get_processor("нет-такой")


def test_реализация_выбирается_переменной_окружения(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Позволяет проверить другой алгоритм, не меняя настройки приложения."""
    registry.register(FakeProcessor())
    monkeypatch.setenv(registry.PROCESSOR_ENV, "fake")

    assert registry.get_processor().name == "fake"


def test_реализация_находится_по_точке_входа(monkeypatch: pytest.MonkeyPatch) -> None:
    """Установка пакета с алгоритмом - единственное, что нужно для подключения."""
    monkeypatch.setattr(
        registry.metadata, "entry_points", lambda **_: [_Entry("внешний", FakeProcessor)]
    )
    registry.refresh()

    assert registry.get_processor("fake").version == "0.1"


def test_сломанный_пакет_не_мешает_работе(monkeypatch: pytest.MonkeyPatch) -> None:
    """Один неверно собранный пакет не должен лишать врача обработки.

    Рядом с ним стоит исправная реализация, и она обязана остаться в реестре.
    """

    def explode() -> None:
        raise RuntimeError("пакет собран неверно")

    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda **_: [_Entry("битый", explode), _Entry("исправный", FakeProcessor)],
    )
    registry.refresh()

    assert set(registry.available()) == {"fake"}


def test_несоответствие_контракту_отбрасывается(monkeypatch: pytest.MonkeyPatch) -> None:
    """Объект без метода process не должен попадать в реестр."""
    monkeypatch.setattr(
        registry.metadata,
        "entry_points",
        lambda **_: [_Entry("чужой", lambda: object()), _Entry("исправный", FakeProcessor)],
    )
    registry.refresh()

    assert set(registry.available()) == {"fake"}


def test_регистрация_из_кода_перекрывает_точку_входа(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Выбор разработчика важнее состава окружения."""
    monkeypatch.setattr(
        registry.metadata, "entry_points", lambda **_: [_Entry("внешний", FakeProcessor)]
    )
    registry.refresh()
    registry.register(FakeProcessor(version="своя"))

    assert registry.get_processor("fake").version == "своя"


class _Entry:
    """Подделка точки входа: у настоящей есть имя и метод load."""

    def __init__(self, name: str, factory: Any) -> None:
        """Создать подделку точки входа."""
        self.name = name
        self._factory = factory

    def load(self) -> Any:
        """Вернуть объект, на который указывает точка входа."""
        return self._factory
