"""Отчёт приёмки.

Отчёт нужен обеим сторонам. Разработчик алгоритма прикладывает его к передаче
пакета, принимающий сохраняет рядом с версией рабочего места: через год вопрос
«на чём проверяли этот алгоритм» задаётся про выпуск, которого уже нет в
окружении.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from .config import Settings
from .packaging import NOTE, REQUIRED, Finding
from .plugin import MEASUREMENTS

#: Пометки исходов в отчёте.
MARKS = {"passed": "+", "failed": "×", "skipped": "~", "error": "×"}


@dataclass(frozen=True)
class Outcome:
    """Исход одной проверки."""

    name: str
    level: str
    outcome: str
    message: str = ""


@dataclass
class Report:
    """Собранный итог приёмки."""

    package: str
    processor: str
    findings: list[Finding]
    outcomes: list[Outcome]
    settings: Settings
    level: str = "all"
    """Какие уровни проверок выполнялись. Отчёт обязан это называть."""

    @property
    def accepted(self) -> bool:
        """Можно ли принимать пакет."""
        blocking = any(not finding.ok and finding.level == REQUIRED for finding in self.findings)
        broken = any(outcome.outcome in {"failed", "error"} for outcome in self.outcomes)
        return not blocking and not broken

    def verdict(self) -> str:
        """Вердикт одной строкой.

        Уровень называется прямо: «принят» после одного уровня и «принят»
        после всех - разные утверждения, и через полгода по отчёту их уже не
        различить.
        """
        if not self.accepted:
            return "НЕ ПРИНЯТ"
        if self.level == "all":
            return "ПРИНЯТ"
        return f"ПРОЙДЕН УРОВЕНЬ {self.level} (остальные не выполнялись)"

    def to_markdown(self) -> str:
        """Собрать отчёт для передачи и хранения."""
        lines = [
            f"# Приёмка пакета обработки: {self.package or 'пакет не назван'}",
            "",
            f"Реализация: **{self.processor or 'не определена'}**  ",
            f"Дата: {date.today():%Y-%m-%d}  ",
            f"Настройки приёмки: {self.settings.source or 'по умолчанию'}  ",
            f"Вердикт: **{self.verdict()}**",
            "",
            "## Оформление пакета",
            "",
        ]
        for finding in self.findings:
            mark = "+" if finding.ok else ("×" if finding.level == REQUIRED else "!")
            detail = f" - {finding.detail}" if finding.detail else ""
            lines.append(f"- `{mark}` {finding.title}{detail}")

        lines += ["", "## Проверки", ""]
        for level in ("contract", "numeric", "budget"):
            selected = [outcome for outcome in self.outcomes if outcome.level == level]
            if not selected:
                continue
            lines.append(f"### {level}")
            lines.append("")
            for outcome in selected:
                mark = MARKS.get(outcome.outcome, "?")
                message = f" - {outcome.message}" if outcome.message else ""
                lines.append(f"- `{mark}` {outcome.name}{message}")
            lines.append("")

        if MEASUREMENTS:
            lines += ["## Измерения", ""]
            lines += [f"- {name}: {value}" for name, value in MEASUREMENTS.items()]
            lines.append("")

        relaxed = self.settings.relaxed()
        if self.settings.deviations or relaxed:
            lines += ["## Объявленные отступления", ""]
            lines += [f"- допуск {item}" for item in relaxed]
            lines += [f"- {item}" for item in self.settings.deviations]
            lines.append("")
            lines.append(
                "Отступления объявлены пакетом и приняты к сведению, а не проверены: "
                "их обоснованность решается людьми."
            )
            lines.append("")

        return "\n".join(lines)

    def to_text(self) -> str:
        """Собрать короткий итог для консоли."""
        lines = [f"Пакет: {self.package or '-'}", f"Реализация: {self.processor or '-'}", ""]
        for finding in self.findings:
            if finding.ok:
                continue
            mark = "×" if finding.level == REQUIRED else "!"
            detail = f" - {finding.detail}" if finding.detail else ""
            lines.append(f" {mark} {finding.title}{detail}")

        failed = [item for item in self.outcomes if item.outcome in {"failed", "error"}]
        for outcome in failed:
            lines.append(f" × {outcome.name}")
            if outcome.message:
                lines.append(f"   {outcome.message}")

        skipped = [item for item in self.outcomes if item.outcome == "skipped"]
        for outcome in skipped:
            lines.append(f" ~ {outcome.name}: {outcome.message}")

        for name, value in MEASUREMENTS.items():
            lines.append(f" · {name}: {value}")

        passed = sum(1 for item in self.outcomes if item.outcome == "passed")
        lines += [
            "",
            f"Пройдено {passed} из {len(self.outcomes)}. "
            f"Замечаний по оформлению: {sum(1 for f in self.findings if not f.ok)}.",
            f"Вердикт: {self.verdict()}",
        ]
        if not self.accepted:
            lines.append("")
            lines.append(
                "Что делать: разберите отмеченные × проверки. "
                "Объяснение каждой - в её докстринге, там же написано, чем она вызвана."
            )
        elif any(finding.level == NOTE and not finding.ok for finding in self.findings):
            lines.append("Замечания не блокируют приёмку, но их стоит поправить.")
        return "\n".join(lines)
