"""Показать подключённые реализации обработки.

Быстрая проверка после установки пакета с алгоритмом: видно ли его рабочему
месту и какая реализация будет выбрана по умолчанию.

    uv run python -m hsr_proc
"""

from __future__ import annotations

import sys
from typing import IO, Any

from .registry import PROCESSOR_ENV, available, get_processor


def use_utf8() -> None:
    """Перевести стандартные потоки на UTF-8.

    Повторяет ``hsr_common.use_utf8``, а не вызывает его: контракт обработки
    ставится и вне монорепозитория - в окружении стороннего разработчика
    алгоритма, где остальных пакетов комплекса нет. Ради одной строки вывода
    тянуть туда пакет с паролями и идентификаторами незачем.
    """
    for stream in (sys.stdout, sys.stderr):
        _reconfigure(stream)


def _reconfigure(stream: IO[Any] | None) -> None:
    """Перевести один поток на UTF-8, если это возможно."""
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    try:
        reconfigure(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return


def main() -> int:
    """Вывести список доступных реализаций обработки."""
    # Сообщения программы написаны по-русски, а консоль Windows по умолчанию
    # работает в однобайтовой кодировке - без этого вывод оборвал бы работу.
    use_utf8()

    processors = available()
    active = get_processor()

    print("Доступные реализации обработки:")
    for name in sorted(processors):
        processor = processors[name]
        mark = "→" if name == active.name else " "
        print(f" {mark} {name} {processor.version}")

    print()
    print(f"Выбрана: {active.name} {active.version}")
    print(f"Сменить: переменная окружения {PROCESSOR_ENV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
