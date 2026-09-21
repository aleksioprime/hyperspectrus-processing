"""Подключение набора проверок к pytest.

Плагин объявляется точкой входа ``pytest11``, поэтому в репозитории с
алгоритмом ничего настраивать не нужно: метки уровней работают сразу после
установки пакета, и ``pytest -m contract`` запускает только обязательное.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - только для проверки типов
    import pytest

#: Уровни проверок и что каждый означает.
LEVELS: dict[str, str] = {
    "contract": "контракт: без этого рабочее место сломается или испортит результат",
    "numeric": "числа: сверка с эталонными наборами",
    "budget": "бюджет: время обработки кадра прибора",
}

#: Измерения, снятые во время проверок. Печатаются в конце прогона.
MEASUREMENTS: dict[str, str] = {}


def remember(name: str, value: str) -> None:
    """Запомнить измерение для итогового сообщения и отчёта."""
    MEASUREMENTS[name] = value


def pytest_configure(config: pytest.Config) -> None:
    """Объявить метки уровней, чтобы они работали и при ``--strict-markers``."""
    for name, description in LEVELS.items():
        config.addinivalue_line("markers", f"{name}: {description}")


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    """Показать итог приёмки по уровням и снятые измерения."""
    outcomes = _by_level(terminalreporter)
    if not outcomes:
        return

    terminalreporter.write_sep("─", "Приёмка пакета обработки")
    for level in LEVELS:
        counts = outcomes.get(level)
        if not counts:
            continue
        line = ", ".join(f"{name} {number}" for name, number in sorted(counts.items()))
        terminalreporter.write_line(f"  {level:9} {line}")
    for name, value in MEASUREMENTS.items():
        terminalreporter.write_line(f"  {name}: {value}")


def _by_level(terminalreporter: pytest.TerminalReporter) -> dict[str, dict[str, int]]:
    """Разложить исходы проверок по уровням."""
    outcomes: dict[str, dict[str, int]] = {}
    for outcome in ("passed", "failed", "skipped", "error"):
        for report in terminalreporter.stats.get(outcome, []):
            keywords = getattr(report, "keywords", {})
            if getattr(report, "when", "call") != "call" and outcome == "passed":
                continue
            for level in LEVELS:
                if level in keywords:
                    outcomes.setdefault(level, {}).setdefault(outcome, 0)
                    outcomes[level][outcome] += 1
    return outcomes
