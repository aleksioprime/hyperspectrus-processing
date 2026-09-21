"""Настройки приёмки: допуски и объявленные отступления.

Проверки должны отсеивать неверные алгоритмы, а не непривычные. Эталонный
набор данных построен на линейной модели: оптическая плотность складывается
из концентраций по матрице коэффициентов. Алгоритм, учитывающий рассеяние или
обученный на снимках, эту модель не воспроизводит точно и провалил бы точную
сверку концентраций, будучи при этом лучше эталонного.

Поэтому послабления разрешены, но только объявленные вслух: они пишутся рядом
с кодом алгоритма вместе с причиной и попадают отдельным разделом в отчёт
приёмки. Тихо ослабить допуск нельзя - проверка ``test_послабления_объяснены``
этого не даст.

Настройки лежат в самом пакете, файлом ``conformance.toml`` рядом с модулем
реализации:

    src/hsr_proc_myalgo/conformance.toml

    profile = "custom"
    concentration_atol = 1e-3
    deviations = [
        "Модель учитывает рассеяние, точное обращение линейной системы не воспроизводится",
    ]

Место выбрано так, чтобы настройки ехали вместе с алгоритмом. ``pyproject.toml``
в колесо не попадает, и приёмка у издателя читала бы файл его собственного
репозитория: объявленные отступления исчезли бы вместе с причинами, а допуски
молча стали бы эталонными.
"""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass, field, fields
from importlib import metadata, resources
from pathlib import Path
from typing import Any

from hsr_proc.registry import ENTRY_POINT_GROUP

#: Файл настроек внутри пакета с алгоритмом.
FILE_NAME = "conformance.toml"

#: Раздел настроек в ``pyproject.toml``. В ``conformance.toml`` ключи лежат и
#: без него: файл целиком принадлежит приёмке, оборачивать их незачем.
SECTION = ("tool", "hsr_proc_conformance")

#: Переменная окружения с явным путём к файлу настроек.
CONFIG_ENV = "HSR_PROC_CONFORMANCE_CONFIG"

#: Профиль, при котором алгоритм обязан точно воспроизводить эталонную модель.
REFERENCE = "reference"

#: Профиль алгоритма на другой модели: числовые сверки, опирающиеся на
#: линейную задачу, пропускаются, но выделение очага и коэффициент s остаются.
CUSTOM = "custom"


@dataclass(frozen=True)
class Settings:
    """Допуски приёмки и объявленные отступления от эталонной модели."""

    profile: str = REFERENCE

    concentration_atol: float = 1e-4
    """Допуск сверки восстановленных концентраций с эталонными."""

    thb_atol: float = 1e-4
    """Допуск сверки карты THb с суммой модулей концентраций гемоглобина."""

    mean_atol: float = 0.05
    """Допуск средних концентраций в очаге и в коже."""

    s_atol: float = 0.2
    """Допуск коэффициента s. Отношение безразмерно и не зависит от модели."""

    runtime_budget_seconds: float = 20.0
    """Сколько секунд отводится на кадр прибора. Мягкая проверка."""

    budget_frame: tuple[int, int] = (720, 1280)
    """Размер кадра, на котором меряется время."""

    deviations: tuple[str, ...] = field(default_factory=tuple)
    """Объявленные отступления: что именно и почему не совпадает с эталоном."""

    source: Path | None = None
    """Файл, из которого прочитаны настройки. Нужен отчёту."""

    @property
    def is_reference(self) -> bool:
        """Обязан ли алгоритм точно воспроизводить эталонную модель."""
        return self.profile == REFERENCE

    def relaxed(self) -> tuple[str, ...]:
        """Вернуть допуски, ослабленные против эталонных."""
        default = Settings()
        names = ("concentration_atol", "thb_atol", "mean_atol", "s_atol")
        return tuple(
            f"{name}: {getattr(self, name)} вместо {getattr(default, name)}"
            for name in names
            if getattr(self, name) > getattr(default, name)
        )


def load(start: Path | None = None, *, distribution: str | None = None) -> Settings:
    """Прочитать настройки приёмки.

    Источники в порядке убывания доверия:

    1. файл, названный переменной ``HSR_PROC_CONFORMANCE_CONFIG``;
    2. ``conformance.toml`` внутри проверяемого пакета - он один едет вместе с
       алгоритмом и читается что у разработчика, что у издателя;
    3. ``pyproject.toml`` вверх от текущего каталога - на время разработки,
       пока пакет ещё не установлен.

    Имя пакета известно не всегда: набор проверок запускают и на реализации,
    переданной прямо в коде. Тогда второй источник пропускается - иначе
    настройки чужого установленного пакета молча подменили бы эталонные.
    """
    override = os.environ.get(CONFIG_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"файл настроек приёмки не найден: {path}")
        return _read(path, wrapped=False)

    packaged = _packaged(distribution)
    if packaged is not None:
        return _read(packaged, wrapped=False)

    pyproject = _locate(start)
    if pyproject is None:
        return Settings()
    return _read(pyproject, wrapped=True)


def _read(path: Path, *, wrapped: bool) -> Settings:
    """Прочитать файл настроек.

    В ``pyproject.toml`` ключи лежат в разделе ``[tool.hsr_proc_conformance]``;
    в ``conformance.toml`` допустимы оба написания - и раздел, и просто ключи.
    """
    with path.open("rb") as stream:
        document = tomllib.load(stream)

    section: Any = document
    for key in SECTION:
        if isinstance(section, dict) and key in section:
            section = section[key]
        elif wrapped:
            return Settings(source=path)

    if not isinstance(section, dict):
        return Settings(source=path)
    return _build(section, path)


def _packaged(distribution: str | None) -> Path | None:
    """Найти ``conformance.toml`` внутри установленного пакета с алгоритмом."""
    if not distribution:
        return None

    wanted = _normalize(distribution)
    for entry in metadata.entry_points(group=ENTRY_POINT_GROUP):
        owner = getattr(entry, "dist", None)
        if owner is None or _normalize(owner.name) != wanted:
            continue
        try:
            root = resources.files(entry.module.split(".")[0])
        except (ImportError, TypeError):
            continue
        candidate = Path(str(root)) / FILE_NAME
        if candidate.is_file():
            return candidate
    return None


def _normalize(name: str) -> str:
    """Привести имя пакета к виду, в котором его сравнивают."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _build(section: dict[str, Any], path: Path) -> Settings:
    """Собрать настройки из раздела файла, отвергая незнакомые ключи.

    Опечатка в имени настройки иначе осталась бы незамеченной: допуск выглядел
    бы объявленным, а действовал бы прежний.
    """
    known = {item.name for item in fields(Settings)} - {"source"}
    unknown = sorted(set(section) - known)
    if unknown:
        raise ValueError(
            f"{path}: неизвестные настройки приёмки: {', '.join(unknown)}; "
            f"допустимы: {', '.join(sorted(known))}"
        )

    values: dict[str, Any] = {"source": path}
    for name, value in section.items():
        if name == "deviations":
            values[name] = tuple(str(item) for item in value)
        elif name == "budget_frame":
            frame = [int(item) for item in value]
            values[name] = (frame[0], frame[1])
        elif name == "profile":
            values[name] = _profile(str(value), path)
        else:
            values[name] = float(value)
    return Settings(**values)


def _profile(value: str, path: Path) -> str:
    """Проверить имя профиля."""
    if value not in {REFERENCE, CUSTOM}:
        raise ValueError(f"{path}: профиль {value!r} неизвестен; допустимы {REFERENCE}, {CUSTOM}")
    return value


def _locate(start: Path | None) -> Path | None:
    """Найти ближайший ``pyproject.toml`` вверх по дереву каталогов."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None
