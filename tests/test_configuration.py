"""Согласованность рабочей конфигурации и учебных файловых серий."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from hsr_proc.models import SpectralCube
from hsr_proc_conformance.runner import _matrix, find_reference

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "references" / "hsr-example.json"
SERIES = ROOT / "datasets" / "example" / "jpeg"
WAVELENGTHS = (450, 517, 671, 775, 803, 851, 888, 939)


def test_рабочая_конфигурация_полная() -> None:
    """У каждой длины волны есть коэффициенты всех хромофоров."""
    document = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert document["format"] == "hyperspectrus-reference"
    assert len(document["models"]) == 1

    symbols = {item["symbol"] for item in document["chromophores"]}
    spectra = document["models"][0]["spectra"]
    assert tuple(item["wavelength_nm"] for item in spectra) == WAVELENGTHS
    assert all(set(item["overlaps"]) == symbols for item in spectra)


def test_учебные_наблюдения_соответствуют_конфигурации() -> None:
    """В каждом наблюдении есть ровно восемь настроенных каналов."""
    expected = {f"{wavelength}nm.jpg" for wavelength in WAVELENGTHS}
    for number in (1, 2, 3):
        actual = {path.name for path in (SERIES / str(number)).glob("*.jpg")}
        assert actual == expected


def test_runner_читает_формат_рабочего_места() -> None:
    """Локальный прогон использует тот же JSON, который загружает приложение."""
    cube = SpectralCube(
        wavelengths_nm=WAVELENGTHS,
        data=np.zeros((len(WAVELENGTHS), 1, 1), dtype=np.float32),
    )
    matrix = _matrix(REFERENCE, cube)

    assert matrix.wavelengths_nm == WAVELENGTHS
    assert matrix.symbols == ("HbO2", "Hb")
    assert matrix.values.shape == (8, 2)


def test_справочник_находится_без_указания(monkeypatch: pytest.MonkeyPatch) -> None:
    """Путь к справочнику не набирают руками: он лежит в репозитории один.

    Прежде без ``--reference`` подставлялись синтетические коэффициенты на
    четыре длины волны, и прогон на восьмиканальной серии падал с жалобой на
    матрицу - искать причину приходилось не там, где она была.
    """
    monkeypatch.chdir(SERIES / "1")

    assert find_reference() == REFERENCE
