"""Настройки приёмки: допуски и объявленные отступления.

Проверки должны отсеивать неверные алгоритмы, а не непривычные. Эталонный
набор данных построен на линейной модели: оптическая плотность складывается
из концентраций по матрице коэффициентов. Алгоритм, учитывающий рассеяние или
обученный на снимках, эту модель не воспроизводит точно и провалил бы точную
сверку концентраций, будучи при этом лучше эталонного.

Поэтому послабления разрешены, но только объявленные вслух: они пишутся в
``pyproject.toml`` пакета вместе с причиной и попадают отдельным разделом в
отчёт приёмки. Тихо ослабить допуск нельзя - проверка
``test_послабления_объяснены`` этого не даст.

    [tool.hsr_proc_conformance]
    profile = "custom"
    concentration_atol = 1e-3
    deviations = [
        "Модель учитывает рассеяние, точное обращение линейной системы не воспроизводится",
    ]
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

#: Раздел настроек в ``pyproject.toml`` пакета с алгоритмом.
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


def load(start: Path | None = None) -> Settings:
    """Прочитать настройки из ``pyproject.toml`` пакета с алгоритмом.

    Файл ищется вверх от указанного каталога - так проверки берут настройки
    пакета, который проверяют, независимо от того, откуда запущен pytest.
    Настроек может не быть вовсе: тогда действуют эталонные допуски.
    """
    path = _locate(start)
    if path is None:
        return Settings()

    with path.open("rb") as stream:
        document = tomllib.load(stream)

    section: Any = document
    for key in SECTION:
        if not isinstance(section, dict) or key not in section:
            return Settings(source=path)
        section = section[key]
    if not isinstance(section, dict):
        return Settings(source=path)

    return _build(section, path)


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
    """Найти файл настроек: по переменной окружения или вверх по дереву."""
    override = os.environ.get(CONFIG_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"файл настроек приёмки не найден: {path}")
        return path

    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / "pyproject.toml"
        if candidate.is_file():
            return candidate
    return None
