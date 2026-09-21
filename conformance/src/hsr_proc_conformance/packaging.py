"""Проверки оформления пакета с алгоритмом.

Алгоритм может считать безупречно и всё равно не дойти до врача: рабочее место
находит реализацию не импортом, а по точке входа в метаданных установленного
пакета. Ошибка в ``pyproject.toml`` проявляется тем, что алгоритма попросту нет
в списке - без единого сообщения об ошибке.

Проверяется установленный пакет, а не исходники: точки входа читаются из
метаданных, и разница между «написано в файле» и «установлено в окружении» -
самая частая причина пропавшего алгоритма.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import metadata

from hsr_proc.contract import Processor
from hsr_proc.registry import ENTRY_POINT_GROUP

#: Переменная окружения с именем проверяемого пакета. Выставляется командой
#: ``hsr-proc-check``, чтобы набор проверок знал, чьё оформление сверять.
DISTRIBUTION_ENV = "HSR_PROC_DISTRIBUTION"

#: Проверка обязательная: без неё пакет не принимается.
REQUIRED = "обязательно"

#: Замечание: принять можно, но лучше поправить.
NOTE = "замечание"

#: Пакеты, которых у алгоритма быть не должно.
#:
#: Рабочее место и эталонная реализация тянут за собой интерфейс и базу: внутри
#: сборки они уже есть, а в окружении исследователя такая зависимость означает,
#: что алгоритм зовёт приложение вместо того, чтобы считать.
FORBIDDEN = ("hsr-workstation", "hsr-proc-algo", "pyside6", "pyqt5", "pyqt6")

#: Версия по PEP 440 в упрощённом виде.
VERSION = re.compile(r"^\d+(\.\d+)*([a-z]+\d+)?(\.(post|dev)\d+)?$")


@dataclass(frozen=True)
class Finding:
    """Итог одной проверки оформления."""

    ok: bool
    level: str
    title: str
    detail: str = ""


def inspect(distribution: str | None = None) -> list[Finding]:
    """Проверить оформление установленного пакета с алгоритмом."""
    entries = [
        entry
        for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)
        if distribution is None or _distribution_name(entry) == _normalize(distribution)
    ]

    if not entries:
        where = f" в пакете {distribution}" if distribution else ""
        return [
            Finding(
                False,
                REQUIRED,
                f"точка входа группы {ENTRY_POINT_GROUP} не найдена{where}",
                "рабочее место не увидит алгоритм; проверьте раздел "
                '[project.entry-points."hsr_proc.processors"] и переустановите пакет',
            )
        ]

    findings: list[Finding] = []
    for entry in entries:
        findings.extend(_check_entry(entry))
        name = _distribution_name(entry)
        if name:
            findings.extend(_check_metadata(name))
    return findings


def processor_names(distribution: str | None = None) -> list[str]:
    """Вернуть имена реализаций, объявленных установленными пакетами."""
    names: list[str] = []
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        if distribution is not None and _distribution_name(entry) != _normalize(distribution):
            continue
        try:
            factory = entry.load()
            processor = factory() if callable(factory) else factory
            names.append(str(processor.name))
        except Exception:
            continue
    return names


def distribution_names() -> list[str]:
    """Вернуть имена установленных пакетов, объявивших реализацию обработки."""
    names = {_distribution_name(entry) for entry in metadata.entry_points(group=ENTRY_POINT_GROUP)}
    return sorted(name for name in names if name)


def _check_entry(entry: metadata.EntryPoint) -> list[Finding]:
    """Проверить, что точка входа загружается и даёт реализацию контракта."""
    try:
        factory = entry.load()
        processor = factory() if callable(factory) else factory
    except Exception as error:
        return [
            Finding(
                False,
                REQUIRED,
                f"точка входа {entry.name} не загружается",
                f"{type(error).__name__}: {error}",
            )
        ]

    findings = [Finding(True, REQUIRED, f"точка входа {entry.name} загружается", entry.value)]

    if not isinstance(processor, Processor):
        missing = [
            member for member in ("name", "version", "process") if not hasattr(processor, member)
        ]
        findings.append(
            Finding(
                False,
                REQUIRED,
                "объект не соответствует контракту Processor",
                f"не хватает членов: {', '.join(missing) or 'подпись process не совпадает'}",
            )
        )
        return findings

    findings.append(
        Finding(True, REQUIRED, f"реализация {processor.name} {processor.version}", entry.value)
    )
    if not VERSION.match(str(processor.version)):
        findings.append(
            Finding(
                False,
                NOTE,
                f"версия {processor.version!r} записана не по PEP 440",
                "версия попадает в карточку сеанса, и по ней сравнивают выпуски",
            )
        )
    return findings


def _check_metadata(distribution: str) -> list[Finding]:
    """Проверить зависимости и требования пакета."""
    try:
        info = metadata.metadata(distribution)
    except metadata.PackageNotFoundError:
        return [Finding(False, REQUIRED, f"пакет {distribution} не установлен")]

    findings: list[Finding] = []
    requires = info.get_all("Requires-Dist") or []
    lowered = [item.lower() for item in requires]

    contract = [item for item in lowered if item.startswith("hsr-proc-base")]
    if not contract:
        findings.append(
            Finding(
                False,
                REQUIRED,
                "пакет не объявляет зависимость от hsr-proc-base",
                "контракт обязан быть в dependencies, иначе пакет ставится в пустое окружение",
            )
        )
    elif not any(symbol in contract[0] for symbol in ("==", ">=", "~=", "<")):
        findings.append(
            Finding(
                False,
                NOTE,
                "версия контракта не ограничена",
                'укажите диапазон, например "hsr-proc-base>=0.1,<0.2": '
                "иначе пакет молча соберётся с несовместимым контрактом",
            )
        )

    forbidden = sorted(
        {name for name in FORBIDDEN for item in lowered if item.split()[0].strip() == name}
    )
    if forbidden:
        findings.append(
            Finding(
                False,
                REQUIRED,
                f"лишние зависимости: {', '.join(forbidden)}",
                "алгоритм не зависит ни от приложения, ни от интерфейса, ни от другой реализации",
            )
        )
    else:
        findings.append(Finding(True, REQUIRED, "лишних зависимостей нет"))

    python = str(info["Requires-Python"] or "")
    if "3.11" not in python and not python.startswith(">=3.1"):
        findings.append(
            Finding(
                False,
                NOTE,
                f"requires-python = {python!r}",
                "рабочее место собирается на 3.11; более новая граница отрежет сборку",
            )
        )
    return findings


def _distribution_name(entry: metadata.EntryPoint) -> str:
    """Вернуть нормализованное имя пакета, объявившего точку входа."""
    distribution = getattr(entry, "dist", None)
    return _normalize(distribution.name) if distribution is not None else ""


def _normalize(name: str) -> str:
    """Привести имя пакета к виду, в котором его сравнивают."""
    return re.sub(r"[-_.]+", "-", name).lower()
