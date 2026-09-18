"""Контракт обработки: что обязана уметь любая реализация алгоритма.

Алгоритм поставляется отдельным пакетом и подключается через реестр. Здесь
описано единственное, что связывает его с рабочим местом: принять
:class:`ProcessingRequest` и вернуть :class:`ProcessingResult`.

Как оформить и подключить свой пакет - см. ``CONTRACT.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

from .models import ProcessingRequest, ProcessingResult

#: Обратный вызов хода обработки: доля выполненного в процентах и описание шага.
#:
#: Обработка серии занимает секунды и выполняется в фоновом потоке, поэтому
#: врач должен видеть, что происходит.
ProgressCallback = Callable[[int, str], None]


@runtime_checkable
class Processor(Protocol):
    """Реализация алгоритма обработки."""

    @property
    def name(self) -> str:
        """Короткое имя реализации, по которому её выбирают в настройках."""
        ...

    @property
    def version(self) -> str:
        """Версия алгоритма. Попадает в результат и в карточку сеанса."""
        ...

    def process(
        self,
        request: ProcessingRequest,
        progress: ProgressCallback | None = None,
    ) -> ProcessingResult:
        """Обработать серию и вернуть карты концентраций и показатели.

        Поднимает :class:`~hsr_proc.errors.InvalidInput`, если данные
        непригодны, и :class:`~hsr_proc.errors.ProcessingFailed`, если
        алгоритм не смог получить результат.
        """
        ...


def describe(processor: Processor) -> str:
    """Вернуть подпись реализации для журналов и карточки сеанса."""
    return f"{processor.name} {processor.version}"


def report(progress: ProgressCallback | None, percent: int, message: str) -> None:
    """Сообщить о ходе обработки, если вызывающий этого просил.

    Проверка на ``None`` вынесена сюда, чтобы не повторять её в каждом шаге
    алгоритма.
    """
    if progress is not None:
        progress(max(0, min(100, percent)), message)
