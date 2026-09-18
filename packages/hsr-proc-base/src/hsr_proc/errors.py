"""Исключения обработки."""

from __future__ import annotations


class ProcessingError(Exception):
    """Базовая ошибка обработки."""


class InvalidInput(ProcessingError):
    """Входные данные не годятся для обработки.

    Например, снимки разного размера или матрица коэффициентов, не
    соответствующая набору спектров.
    """


class ProcessorNotFound(ProcessingError):
    """Запрошенная реализация обработки не подключена."""


class ProcessingFailed(ProcessingError):
    """Алгоритм не смог обработать серию."""
