"""Передать пакет алгоритма в репозиторий HyperSpectRus.

Алгоритм разрабатывается здесь, а работает в рабочем месте врача. Передавать
его колесом нельзя: в приложении пакет с тем же именем подключён членом
workspace, и ближайший ``uv sync`` вернул бы прежний расчёт - молча, вместе с
собранной версией программы.

Поэтому пакет переносится зеркалом, как контракт:

    uv run python tools/sync_algo.py ../hyperspectrus

Передавать имеет смысл только то, что прошло полную приёмку:

    uv run hsr-proc-check hsr-proc-algo --report приёмка.md
"""

from __future__ import annotations

import sys
from pathlib import Path

from mirror import target_argument, transfer

PACKAGE = Path("packages/hsr-proc-algo")

if __name__ == "__main__":
    sys.exit(transfer(PACKAGE, target_argument("Передать пакет алгоритма HyperSpectRus")))
