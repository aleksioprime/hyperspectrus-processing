"""Реестр реализаций обработки.

Алгоритм приходит отдельным пакетом и объявляет себя точкой входа группы
``hsr_proc.processors``. Рабочее место выбирает реализацию по имени, а
не по импорту конкретного модуля - благодаря этому замена алгоритма сводится к
установке другого пакета и смене имени в настройках.

Точки входа опрашиваются один раз и запоминаются: перебор метаданных всех
установленных пакетов занимает заметное время, а состав окружения во время
работы приложения не меняется.
"""

from __future__ import annotations

import logging
import os
from importlib import metadata

from .contract import Processor
from .errors import ProcessorNotFound

logger = logging.getLogger(__name__)

#: Группа точек входа, в которой ищутся реализации обработки.
ENTRY_POINT_GROUP = "hsr_proc.processors"

#: Переменная окружения, задающая реализацию по умолчанию.
PROCESSOR_ENV = "HSR_PROCESSOR"


#: Реализации, зарегистрированные из кода.
_registered: dict[str, Processor] = {}

#: Реализации, найденные среди точек входа. ``None`` - поиск ещё не выполнялся.
_discovered: dict[str, Processor] | None = None


def register(processor: Processor, *, replace: bool = False) -> None:
    """Зарегистрировать реализацию из кода.

    Нужно там, где точки входа неудобны: в тестах и при отладке алгоритма
    прямо в рабочем месте.
    """
    name = processor.name
    if not replace and name in _registered:
        raise ValueError(f"реализация {name!r} уже зарегистрирована")
    _registered[name] = processor


def unregister(name: str) -> None:
    """Убрать реализацию, зарегистрированную из кода."""
    _registered.pop(name, None)


def refresh() -> None:
    """Забыть найденные точки входа: следующий запрос выполнит поиск заново.

    Требуется после установки пакета с алгоритмом в уже запущенном процессе.
    """
    global _discovered
    _discovered = None


def available() -> dict[str, Processor]:
    """Вернуть доступные реализации: найденные по точкам входа и добавленные.

    Своей реализации у пакета нет: он описывает контракт, а считает алгоритм из
    отдельного пакета. Поэтому список может оказаться пустым - тогда обработка
    недоступна, и приложение обязано сказать об этом врачу, а не считать чем
    попало.
    """
    processors: dict[str, Processor] = {}
    processors.update(_discover())
    # Зарегистрированные из кода перекрывают остальные: это осознанный выбор
    # разработчика, а не результат состава окружения.
    processors.update(_registered)
    return processors


def get_processor(name: str | None = None) -> Processor:
    """Вернуть реализацию обработки.

    Имя берётся из аргумента, затем из ``HSR_PROCESSOR``. Если не задано ни
    там, ни там, а подключена ровно одна реализация - берётся она: имён
    реализаций контракт не знает и знать не должен, иначе переименование
    алгоритма ломало бы выбор по умолчанию.

    Когда реализаций несколько, выбор делает человек: молча взять первую
    попавшуюся значило бы посчитать сеанс неизвестно чем.
    """
    processors = available()
    requested = name or os.environ.get(PROCESSOR_ENV) or ""

    if requested:
        try:
            return processors[requested]
        except KeyError as exc:
            known = ", ".join(sorted(processors)) or "нет ни одной"
            raise ProcessorNotFound(
                f"реализация обработки {requested!r} не подключена; доступны: {known}"
            ) from exc

    if not processors:
        raise ProcessorNotFound("реализация обработки не подключена: установите пакет с алгоритмом")
    if len(processors) > 1:
        known = ", ".join(sorted(processors))
        raise ProcessorNotFound(
            f"подключено несколько реализаций ({known}) - выберите одну "
            f"настройкой приложения или переменной {PROCESSOR_ENV}"
        )
    return next(iter(processors.values()))


def _discover() -> dict[str, Processor]:
    """Найти реализации среди точек входа установленных пакетов."""
    global _discovered
    if _discovered is not None:
        return _discovered

    found: dict[str, Processor] = {}
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        try:
            factory = entry.load()
            processor = factory() if callable(factory) else factory
        except Exception:
            # Сломанный сторонний пакет не должен мешать работе рабочего места:
            # врач продолжит работать на встроенной реализации.
            logger.exception("не удалось загрузить реализацию обработки %r", entry.name)
            continue

        if not isinstance(processor, Processor):
            logger.error(
                "точка входа %r не соответствует контракту обработки - пропущена", entry.name
            )
            continue
        found[processor.name] = processor

    _discovered = found
    return found
