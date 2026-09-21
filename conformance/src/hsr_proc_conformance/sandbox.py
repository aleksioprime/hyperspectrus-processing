"""Наблюдение за побочными действиями алгоритма.

Правило контракта - никакого ввода-вывода: алгоритм получает массивы и
возвращает массивы, а файлы сохраняет рабочее место. Правило не формальное.
Алгоритм, пишущий рядом с собой промежуточные файлы, у врача окажется в
каталоге установки без права записи; алгоритм, уходящий в сеть, вынесет
снимки пациента за пределы компьютера. Проверить это в приложении поздно,
поэтому проверяется здесь.

Наблюдение сделано подменой имён и не является защитой: оно ловит обычный
код, а не намеренный обход. Задача - поймать забытый ``np.save`` и запрос к
серверу модели, а не остановить злоумышленника.
"""

from __future__ import annotations

import builtins
import io
import os
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

#: Режимы открытия файла, означающие запись.
WRITE_FLAGS = frozenset("wax+")


@dataclass
class Observed:
    """Побочные действия, замеченные за время вызова."""

    writes: list[str] = field(default_factory=list)
    network: list[str] = field(default_factory=list)
    directory_changed: bool = False

    @property
    def clean(self) -> bool:
        """Не было ли побочных действий вовсе."""
        return not self.writes and not self.network and not self.directory_changed

    def describe(self) -> str:
        """Описать замеченное для сообщения об ошибке."""
        parts: list[str] = []
        if self.writes:
            parts.append("запись файлов: " + ", ".join(sorted(set(self.writes))[:5]))
        if self.network:
            parts.append("обращения к сети: " + ", ".join(sorted(set(self.network))[:5]))
        if self.directory_changed:
            parts.append("сменён рабочий каталог")
        return "; ".join(parts)


@contextmanager
def watch() -> Iterator[Observed]:
    """Следить за записью файлов, обращениями к сети и сменой каталога."""
    observed = Observed()
    directory = os.getcwd()

    original_open = builtins.open
    original_io_open = io.open
    original_socket = socket.socket
    original_connection = socket.create_connection

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if WRITE_FLAGS & set(mode):
            observed.writes.append(str(file))
        return original_open(file, mode, *args, **kwargs)

    def guarded_socket(*args: Any, **kwargs: Any) -> Any:
        observed.network.append("socket")
        return original_socket(*args, **kwargs)

    def guarded_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
        observed.network.append(str(address))
        return original_connection(address, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_open
    socket.socket = guarded_socket  # type: ignore[misc, assignment]
    socket.create_connection = guarded_connection
    try:
        yield observed
    finally:
        builtins.open = original_open
        io.open = original_io_open
        socket.socket = original_socket  # type: ignore[misc]
        socket.create_connection = original_connection
        observed.directory_changed = os.getcwd() != directory
        if observed.directory_changed:
            os.chdir(directory)
