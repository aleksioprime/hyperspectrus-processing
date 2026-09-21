"""Передать контракт обработки в репозиторий HyperSpectRus.

Главный исходник ``hsr-proc-base`` находится рядом с алгоритмом: состав входных
и выходных данных меняется вместе с расчётом. После согласованного изменения
интерфейса его передают в приложение:

    uv run python tools/sync_contract.py ../hyperspectrus
"""

from __future__ import annotations

import sys
from pathlib import Path

from mirror import target_argument, transfer

PACKAGE = Path("packages/hsr-proc-base")

if __name__ == "__main__":
    sys.exit(transfer(PACKAGE, target_argument("Передать контракт обработки HyperSpectRus")))
