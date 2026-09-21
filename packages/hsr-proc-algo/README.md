# hsr-proc-algo

Редактируемый пакет алгоритма обработки для HyperSpectRus.

Расчёт находится в `src/hsr_proc_algo/processor.py`. Точка входа, имя пакета,
имя модуля и имя реализации `algo` уже согласованы с рабочим местом — менять их
не нужно. При изменении расчёта поднимите поле `version` класса `AlgoProcessor`.

```bash
uv sync --all-packages
uv run pytest -m contract packages/hsr-proc-algo/tests
uv run hsr-proc-check
uv run hsr-proc-run --synthetic --out out
```

Пока расчёт не реализован, обязательные проверки проходят, а числовые ожидаемо
падают. Требования к входу, выходу и ограничения описаны в корневом
[`README.md`](../../README.md#контракт-обработки). Допуски приёмки находятся в
корневом `pyproject.toml`.
