"""Тесты сборки куба из файлов серии."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hsr_proc.errors import InvalidInput
from hsr_proc.loading import discover_frames, load_cube, load_session_set
from PIL import Image


def write_frame(path: Path, level: int, size: tuple[int, int] = (4, 3)) -> None:
    """Записать кадр заданной яркости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=level).save(path)


def test_кадры_находятся_по_имени(tmp_path: Path) -> None:
    """Длина волны берётся из имени файла, как её записывает устройство."""
    write_frame(tmp_path / "450nm.jpg", 10)
    write_frame(tmp_path / "939nm.png", 20)
    (tmp_path / "заметки.txt").write_text("не кадр", encoding="utf-8")
    (tmp_path / "450nm.dng").write_bytes(b"raw")

    frames = discover_frames(tmp_path)

    assert set(frames) == {450, 939}


def test_повтор_длины_волны_отклоняется(tmp_path: Path) -> None:
    """Иначе выбор кадра зависел бы от порядка обхода каталога."""
    write_frame(tmp_path / "450nm.jpg", 10)
    write_frame(tmp_path / "450nm.png", 20)

    with pytest.raises(InvalidInput, match="450"):
        discover_frames(tmp_path)


def test_каталог_без_кадров_отклоняется(tmp_path: Path) -> None:
    """Пустой набор лучше обнаружить до начала обработки."""
    with pytest.raises(InvalidInput):
        discover_frames(tmp_path)


def test_слои_идут_по_возрастанию_длины_волны(tmp_path: Path) -> None:
    """От порядка слоёв зависит соответствие строкам матрицы коэффициентов."""
    write_frame(tmp_path / "939nm.jpg", 200)
    write_frame(tmp_path / "450nm.jpg", 100)

    cube = load_cube(discover_frames(tmp_path))

    assert cube.wavelengths_nm == (450, 939)
    assert cube.data[0].mean() < cube.data[1].mean()


def test_яркости_приводятся_к_единичному_диапазону(tmp_path: Path) -> None:
    """Алгоритм ждёт значения от нуля до единицы независимо от формата снимка."""
    write_frame(tmp_path / "450nm.jpg", 255)

    cube = load_cube(discover_frames(tmp_path))

    assert np.allclose(cube.data, 1.0)


def test_кадры_разного_размера_отклоняются(tmp_path: Path) -> None:
    """Собрать куб из кадров разного размера невозможно."""
    write_frame(tmp_path / "450nm.jpg", 10, size=(4, 3))
    write_frame(tmp_path / "517nm.jpg", 10, size=(5, 3))

    with pytest.raises(InvalidInput, match="разный размер"):
        load_cube(discover_frames(tmp_path))


def test_набор_серии_читается_по_раскладке_устройства(tmp_path: Path) -> None:
    """Раскладка каталогов совпадает с той, что создаёт устройство."""
    write_frame(tmp_path / "jpeg" / "2" / "450nm.jpg", 10)
    write_frame(tmp_path / "jpeg" / "2" / "517nm.jpg", 20)

    cube = load_session_set(tmp_path, set_number=2)

    assert cube.wavelengths_nm == (450, 517)
