"""Команда ``hsr-proc-check``: приёмка пакета одной командой.

Одно действие для обеих сторон. Разработчик алгоритма запускает её перед
отправкой и видит ровно то, что увидит принимающий; принимающий запускает её на
установленном пакете и получает отчёт, который можно сохранить рядом с выпуском
рабочего места.

    uv run hsr-proc-check
    uv run hsr-proc-check hsr-proc-algo --level contract
    uv run hsr-proc-check --report приёмка.md
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import pytest
from hsr_proc.registry import PROCESSOR_ENV

from . import config, packaging
from .packaging import DISTRIBUTION_ENV
from .plugin import LEVELS
from .report import Outcome, Report

#: Модуль с набором проверок для установленной реализации.
SUITE = "hsr_proc_conformance.installed"


def main(argv: list[str] | None = None) -> int:
    """Проверить пакет с алгоритмом и напечатать вердикт."""
    parser = argparse.ArgumentParser(
        prog="hsr-proc-check",
        description="Приёмочные проверки пакета с алгоритмом обработки HyperSpectRus",
    )
    parser.add_argument(
        "package",
        nargs="?",
        help="имя проверяемого пакета; по умолчанию проверяется всё установленное",
    )
    parser.add_argument(
        "--level",
        choices=["all", *LEVELS],
        default="all",
        help="какой уровень проверок выполнять (по умолчанию все)",
    )
    parser.add_argument("--report", type=Path, help="куда записать отчёт в формате Markdown")
    parser.add_argument("--verbose", action="store_true", help="показать подробный вывод pytest")
    args = parser.parse_args(argv)

    findings = packaging.inspect(args.package)
    names = packaging.processor_names(args.package)
    packages = [args.package] if args.package else packaging.distribution_names()
    if len(packages) == 1:
        os.environ[DISTRIBUTION_ENV] = packages[0]
    if len(names) == 1:
        # Реализаций в окружении может быть несколько, и тогда контракт
        # отказывается выбирать сам. Здесь выбор очевиден: проверяется та,
        # которую назвали.
        os.environ[PROCESSOR_ENV] = names[0]

    outcomes = _run_suite(args.level, verbose=args.verbose)
    report = Report(
        package=", ".join(packages),
        processor=names[0] if len(names) == 1 else ", ".join(names),
        findings=findings,
        outcomes=outcomes,
        settings=config.load(),
        level=args.level,
    )

    print()
    print(report.to_text())
    if args.report:
        args.report.write_text(report.to_markdown(), encoding="utf-8")
        print(f"\nОтчёт записан: {args.report}")

    return 0 if report.accepted else 1


def _run_suite(level: str, *, verbose: bool) -> list[Outcome]:
    """Прогнать набор проверок и собрать исходы."""
    collector = _Collector()
    arguments = ["--pyargs", SUITE, "-p", "no:cacheprovider"]
    arguments.append("-v" if verbose else "-q")
    if level != "all":
        arguments += ["-m", level]

    pytest.main(arguments, plugins=[collector])
    return collector.outcomes


class _Collector:
    """Плагин pytest, запоминающий исход каждой проверки."""

    def __init__(self) -> None:
        self.outcomes: list[Outcome] = []

    def pytest_runtest_logreport(self, report: Any) -> None:
        """Запомнить исход проверки."""
        if report.when != "call" and not (report.when == "setup" and report.outcome != "passed"):
            return

        level = next((name for name in LEVELS if name in report.keywords), "contract")
        name = report.nodeid.split("::")[-1]
        message = ""
        if report.outcome == "skipped":
            message = _skip_reason(report)
        elif report.outcome == "failed":
            message = _failure(report)
        self.outcomes.append(Outcome(name, level, report.outcome, message))


def _skip_reason(report: Any) -> str:
    """Достать причину пропуска."""
    reason = getattr(report, "longrepr", ("", 0, ""))
    if isinstance(reason, tuple) and len(reason) == 3:
        return str(reason[2]).removeprefix("Skipped: ")
    return str(reason)


def _failure(report: Any) -> str:
    """Достать сообщение упавшей проверки.

    В выводе pytest объяснение помечено буквой ``E`` в начале строки - это
    текст assert, написанный ради принимающего. Остальное - трассировка,
    которая ему ничего не скажет.
    """
    text = str(getattr(report, "longrepr", "")).strip()
    explained = [
        line.strip()[1:].strip()
        for line in text.splitlines()
        if line.startswith("E") and line.strip() != "E"
    ]
    for line in explained:
        message = line.removeprefix("AssertionError:").strip()
        if message and not message.startswith("assert"):
            return message
    return explained[0] if explained else ""


if __name__ == "__main__":
    sys.exit(main())
