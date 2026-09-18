"""Приёмочные проверки реализации обработки.

Набор подключается наследованием, а не копированием: пакет с алгоритмом
объявляет свой класс, и все проверки достаются ему готовыми.

    from hsr_proc_conformance import ProcessorConformance
    from hsr_proc_myalgo import MyProcessor


    class TestМойАлгоритм(ProcessorConformance):
        processor = MyProcessor()
        distribution = "hsr-proc-myalgo"

Проверки разделены метками на три уровня:

``contract``
    Обязательное. Реализация, не прошедшая этот уровень, сломает рабочее место
    или испортит результат, и в сборку не берётся.
``numeric``
    Числа. Опираются на эталонную линейную модель; алгоритму на другой модели
    часть из них можно отключить объявленным отступлением - см. ``config``.
``budget``
    Время и память на кадре прибора. Не приговор, но число в отчёте.

Запуск только обязательного уровня: ``pytest -m contract``.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

import numpy as np
import pytest
from hsr_proc.contract import Processor
from hsr_proc.errors import InvalidInput, ProcessingFailed
from hsr_proc.models import (
    OverlapMatrix,
    ProcessingRequest,
    ProcessingResult,
    SpectralCube,
)
from hsr_proc.registry import get_processor

from . import data, sandbox
from .config import Settings, load
from .packaging import DISTRIBUTION_ENV, REQUIRED, inspect, processor_names
from .plugin import remember

#: Допуск сравнения двух прогонов одного алгоритма между собой. К модели
#: отношения не имеет: один и тот же расчёт обязан давать одно и то же.
SAME = 1e-6


class ProcessorConformance:
    """Проверки, обязательные для любой реализации обработки."""

    processor: ClassVar[Processor | None] = None
    """Проверяемая реализация. Если не задана - берётся из точек входа."""

    distribution: ClassVar[str | None] = None
    """Имя пакета для проверок оформления.

    Если не задано - берётся из переменной ``HSR_PROC_DISTRIBUTION``: её
    выставляет ``hsr-proc-check``, когда проверяет уже установленный пакет.
    """

    # ── Приспособления ───────────────────────────────────────────────────────

    @pytest.fixture
    def algo(self) -> Processor:
        """Проверяемая реализация."""
        if self.processor is not None:
            return self.processor
        return get_processor()

    @pytest.fixture
    def settings(self) -> Settings:
        """Допуски приёмки из ``pyproject.toml`` проверяемого пакета."""
        return load()

    @pytest.fixture
    def case(self) -> data.Case:
        """Основной набор данных с известным правильным ответом."""
        return data.reference_case()

    @pytest.fixture
    def result(self, algo: Processor, case: data.Case) -> ProcessingResult:
        """Результат обработки основного набора."""
        return algo.process(case.request)

    # ── Уровень 1: контракт ──────────────────────────────────────────────────

    @pytest.mark.contract
    def test_реализация_соответствует_протоколу(self, algo: Processor) -> None:
        """Рабочее место проверяет это при загрузке и молча пропускает чужой пакет.

        Не прошедшая проверку реализация просто не появится в списке, и
        разбираться, почему алгоритма нет, придётся уже на рабочем месте врача.
        """
        assert isinstance(algo, Processor)

    @pytest.mark.contract
    def test_пакет_объявляет_реализацию_точкой_входа(self, algo: Processor) -> None:
        """Рабочее место находит алгоритм по метаданным пакета, а не импортом.

        Ошибка в разделе точек входа выглядит так: алгоритма просто нет в
        списке, и ни одного сообщения об ошибке при этом не появляется.
        Проверяется установленный пакет: «написано в pyproject.toml» и
        «установлено в окружении» - разные вещи.
        """
        package = self.distribution or os.environ.get(DISTRIBUTION_ENV) or None
        if package is None:
            pytest.skip('имя пакета не объявлено в классе: distribution = "hsr-proc-…"')

        broken = [
            finding for finding in inspect(package) if not finding.ok and finding.level == REQUIRED
        ]
        assert not broken, "; ".join(f"{item.title} ({item.detail})" for item in broken)
        assert algo.name in processor_names(package), (
            f"пакет {package} объявляет другие реализации: "
            f"{processor_names(package)}; переустановите пакет"
        )

    @pytest.mark.contract
    def test_имя_и_версия_объявлены(self, algo: Processor) -> None:
        """Имя выбирает реализацию, версия попадает в карточку сеанса.

        Пустое имя сделало бы выбор невозможным, а пустая версия - результат
        обработки неотличимым от результата прежнего расчёта.
        """
        assert isinstance(algo.name, str) and algo.name.strip(), "имя реализации не объявлено"
        assert algo.name == algo.name.strip().lower(), (
            f"имя {algo.name!r} записывается строчными буквами без пробелов: "
            "оно попадает в настройки и в переменную окружения"
        )
        assert isinstance(algo.version, str) and algo.version.strip(), "версия не объявлена"

    @pytest.mark.contract
    def test_результат_заполнен_целиком(self, result: ProcessingResult, case: data.Case) -> None:
        """Рабочее место сохраняет все поля результата и ни одного не достраивает."""
        _check_result(result, case.request.cube)

    @pytest.mark.contract
    def test_результат_подписан_именем_и_версией(
        self, algo: Processor, result: ProcessingResult
    ) -> None:
        """По карточке сеанса должно быть видно, чем он посчитан.

        Без подписи результаты двух версий алгоритма в одной таблице выглядят
        одинаково достоверными.
        """
        assert result.processor == f"{algo.name} {algo.version}", (
            f"подпись {result.processor!r} не совпадает с {algo.name!r} и {algo.version!r}"
        )

    @pytest.mark.contract
    def test_карты_есть_для_всех_хромофоров(
        self, result: ProcessingResult, case: data.Case
    ) -> None:
        """Справочник задаёт рабочее место, и карта нужна по каждому хромофору."""
        assert set(result.concentrations) == set(case.request.overlap.symbols), (
            f"карты {sorted(result.concentrations)} не соответствуют справочнику "
            f"{sorted(case.request.overlap.symbols)}"
        )

    @pytest.mark.contract
    def test_повторный_прогон_даёт_тот_же_результат(self, algo: Processor) -> None:
        """Два запуска на одних данных обязаны совпасть.

        Случайная инициализация или необъявленное состояние между вызовами
        означают, что повторная обработка сеанса даст врачу другие числа при
        тех же снимках.
        """
        case = data.reference_case()
        first = algo.process(case.request)
        second = algo.process(case.request)

        assert np.allclose(first.thb_map, second.thb_map, atol=SAME), "карты THb разошлись"
        assert np.array_equal(first.lesion_mask, second.lesion_mask), "маски разошлись"
        assert first.metrics.s_coefficient == pytest.approx(
            second.metrics.s_coefficient, abs=SAME
        ), "коэффициент s разошёлся"

    @pytest.mark.contract
    def test_вход_не_изменяется(self, algo: Processor) -> None:
        """Запрос принадлежит рабочему месту, и оно обрабатывает им и другие участки.

        Правка входа на месте испортила бы соседний участок сеанса, причём
        только при обработке нескольких участков подряд.
        """
        case = data.reference_case()
        cube_before = case.request.cube.data.copy()
        matrix_before = case.request.overlap.values.copy()

        algo.process(case.request)

        assert np.array_equal(case.request.cube.data, cube_before), "алгоритм изменил куб"
        assert np.array_equal(case.request.overlap.values, matrix_before), (
            "алгоритм изменил матрицу коэффициентов"
        )

    @pytest.mark.contract
    def test_порядок_строк_матрицы_не_влияет(self, algo: Processor) -> None:
        """Порядок кадров задаёт прибор, порядок строк - справочник.

        Совпадать они не обязаны; привести их в соответствие должен вызов
        ``request.overlap.aligned_to(request.cube)``. Без него концентрации
        окажутся перепутаны молча - результат выглядит правдоподобно.
        """
        case = data.reference_case()
        straight = algo.process(case.request)

        order = [3, 1, 0, 2]
        matrix = case.request.overlap
        shuffled = OverlapMatrix(
            wavelengths_nm=tuple(matrix.wavelengths_nm[index] for index in order),
            chromophores=matrix.chromophores,
            values=np.asarray(matrix.values[order, :], dtype=np.float32),
        )
        mixed = algo.process(
            ProcessingRequest(cube=case.request.cube, overlap=shuffled, params=case.request.params)
        )

        # Сверяются карты по каждому хромофору, а не только THb: сумма
        # концентраций симметрична, и перепутанные местами HbO2 и Hb дают ту же
        # карту THb. Именно так эта ошибка и остаётся незамеченной.
        for symbol in straight.concentrations:
            assert np.allclose(
                straight.concentration(symbol), mixed.concentration(symbol), atol=SAME
            ), (
                f"перестановка строк матрицы изменила карту {symbol}: "
                "вызовите request.overlap.aligned_to(request.cube)"
            )
        assert np.allclose(straight.thb_map, mixed.thb_map, atol=SAME), (
            "перестановка строк матрицы изменила карту THb: "
            "вызовите request.overlap.aligned_to(request.cube)"
        )

    @pytest.mark.contract
    def test_обработка_не_пишет_файлов_и_не_ходит_в_сеть(self, algo: Processor) -> None:
        """Файлы сохраняет рабочее место, а снимки не покидают компьютер.

        У врача каталог установки бывает закрыт на запись, а сеть - закрыта
        совсем: это медицинские данные.
        """
        case = data.reference_case()

        with sandbox.watch() as observed:
            algo.process(case.request)

        assert observed.clean, f"алгоритм делает лишнее: {observed.describe()}"

    @pytest.mark.contract
    def test_ход_обработки_сообщается(self, algo: Processor) -> None:
        """Обработка занимает секунды, и окно не должно выглядеть замершим."""
        steps: list[tuple[int, str]] = []
        case = data.reference_case()

        algo.process(case.request, lambda percent, message: steps.append((percent, message)))

        assert steps, "алгоритм ни разу не сообщил о ходе работы"
        percents = [percent for percent, _ in steps]
        assert all(0 <= percent <= 100 for percent in percents), f"проценты вне 0…100: {percents}"
        assert percents == sorted(percents), f"проценты убывают: {percents}"
        assert percents[-1] == 100, "последнее сообщение обязано быть стопроцентным"
        assert all(message.strip() for _, message in steps), "шаг без описания"

    @pytest.mark.contract
    def test_обработка_работает_без_обратного_вызова(self, algo: Processor) -> None:
        """Рабочее место зовёт алгоритм и без показа хода: при проверке установки."""
        case = data.reference_case()

        result = algo.process(case.request, None)

        _check_result(result, case.request.cube)

    @pytest.mark.contract
    def test_непригодный_вход_отклоняется_ошибкой_контракта(self, algo: Processor) -> None:
        """Врач должен увидеть причину, а не «непредвиденный сбой».

        Всё, кроме ``InvalidInput`` и ``ProcessingFailed``, рабочее место
        показывает как неизвестную ошибку: причина остаётся в журнале, а врач
        остаётся без объяснения.
        """
        case = data.reference_case()
        matrix = data.overlap(wavelengths=(450, 517, 671, 940))
        request = ProcessingRequest(cube=case.request.cube, overlap=matrix)

        with pytest.raises((InvalidInput, ProcessingFailed)):
            algo.process(request)

    @pytest.mark.contract
    def test_обработка_идёт_в_фоновом_потоке(self, algo: Processor) -> None:
        """Рабочее место считает в фоне, иначе окно замирало бы на всё время."""
        case = data.reference_case()
        expected = algo.process(case.request)

        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(algo.process, case.request).result()

        assert np.allclose(result.thb_map, expected.thb_map, atol=SAME), (
            "в фоновом потоке результат другой"
        )

    @pytest.mark.contract
    def test_послабления_объяснены(self, settings: Settings) -> None:
        """Ослабленный допуск - это разговор, а не настройка.

        Отступление от эталонной модели бывает обоснованным, но принимающий
        пакет должен знать, в чём оно состоит.
        """
        relaxed = settings.relaxed()
        if settings.is_reference and not relaxed:
            return
        assert settings.deviations, (
            "в pyproject.toml ослаблены допуски или выбран профиль custom, "
            f"но deviations не заполнены ({', '.join(relaxed) or settings.profile})"
        )

    # ── Уровень 2: числа ─────────────────────────────────────────────────────

    @pytest.mark.numeric
    def test_концентрации_восстанавливаются(
        self, result: ProcessingResult, case: data.Case, settings: Settings
    ) -> None:
        """Главная проверка: яркости построены по известным концентрациям.

        Алгоритм обязан вернуть именно их - других правильных ответов у этой
        задачи нет.
        """
        _require_reference(settings, "точное восстановление концентраций")
        assert case.concentrations is not None

        assert np.allclose(
            result.concentration("HbO2"), case.concentrations[0], atol=settings.concentration_atol
        ), "карта HbO2 не совпала с эталонной"
        assert np.allclose(
            result.concentration("Hb"), case.concentrations[1], atol=settings.concentration_atol
        ), "карта Hb не совпала с эталонной"

    @pytest.mark.numeric
    def test_карта_thb_складывается_из_гемоглобинов(
        self, result: ProcessingResult, case: data.Case, settings: Settings
    ) -> None:
        """THb - общая концентрация гемоглобина, и по ней выделяют очаг."""
        _require_reference(settings, "сверка карты THb")
        assert case.concentrations is not None

        expected = case.concentrations[0] + case.concentrations[1]
        assert np.allclose(result.thb_map, expected, atol=settings.thb_atol), (
            "карта THb не равна сумме концентраций гемоглобина"
        )

    @pytest.mark.numeric
    def test_очаг_поражения_выделяется(self, result: ProcessingResult, case: data.Case) -> None:
        """Порог обязан отделить очаг от здоровой кожи, а не разрезать кадр наугад.

        Сглаживание размывает границу, поэтому сверяются середина очага и
        заведомо здоровый угол кадра.
        """
        assert case.lesion is not None
        middle = slice(case.lesion.start + 4, case.lesion.stop - 4)

        assert result.lesion_mask[middle, middle].all(), "середина очага не попала в маску"
        assert not result.lesion_mask[0:6, 0:6].any(), "здоровый угол кадра попал в маску"

    @pytest.mark.numeric
    def test_показатели_соответствуют_концентрациям(
        self, result: ProcessingResult, settings: Settings
    ) -> None:
        """Коэффициент s - отношение средних THb в очаге и в коже.

        Отношение безразмерно и от модели не зависит: в наборе очаг втрое
        насыщеннее кожи, и это обязан увидеть любой алгоритм.
        """
        assert result.metrics.s_coefficient == pytest.approx(3.0, abs=settings.s_atol), (
            f"коэффициент s равен {result.metrics.s_coefficient}, ожидалось около 3"
        )

        _require_reference(settings, "сверка средних концентраций")
        assert result.metrics.mean_lesion_thb == pytest.approx(0.9, abs=settings.mean_atol)
        assert result.metrics.mean_skin_thb == pytest.approx(0.3, abs=settings.mean_atol)

    @pytest.mark.numeric
    @pytest.mark.parametrize("level", [(0.2, 0.1), (0.6, 0.3)])
    def test_однородный_кадр_не_даёт_ложного_очага(
        self, algo: Processor, level: tuple[float, float]
    ) -> None:
        """На коже без поражения делить не на что, а показатели остаются числами.

        Порог на однородном кадре выделяет либо всё, либо ничего - оба исхода
        допустимы. Недопустимо деление на ноль и NaN в карточке сеанса.
        """
        case = data.uniform_case(level)

        result = algo.process(case.request)

        mask = result.lesion_mask
        assert not mask.any() or mask.all(), "на однородном кадре выделен очаг"
        assert np.isfinite(result.metrics.s_coefficient), "коэффициент s не число"
        assert np.isfinite(result.metrics.mean_lesion_thb)
        assert np.isfinite(result.metrics.mean_skin_thb)
        if result.metrics.mean_skin_thb == 0.0:
            assert result.metrics.s_coefficient == 0.0, "деление на нулевую концентрацию кожи"

    @pytest.mark.numeric
    @pytest.mark.parametrize(
        "case_factory",
        [data.without_hemoglobin_case, data.without_deoxy_case],
        ids=["без гемоглобина", "без Hb"],
    )
    def test_неполный_справочник_объясняется(
        self, algo: Processor, case_factory: Callable[[], data.Case]
    ) -> None:
        """Справочник ведёт пользователь, и он бывает неполон.

        Допустимы оба исхода: отказаться считать или посчитать и объяснить, из
        чего получен THb. Недопустим третий - посчитать молча.
        """
        case = case_factory()

        try:
            result = algo.process(case.request)
        except (InvalidInput, ProcessingFailed):
            return

        assert result.notes.strip(), (
            f"справочник неполон ({case.description}), но пояснения для врача нет"
        )

    # ── Уровень 3: бюджет ────────────────────────────────────────────────────

    @pytest.mark.budget
    def test_кадр_прибора_укладывается_в_бюджет(self, algo: Processor, settings: Settings) -> None:
        """Врач ждёт обработку у экрана, и минуты ожидания - это уже другое приложение."""
        height, width = settings.budget_frame
        case = data.frame_case(height, width)

        started = time.perf_counter()
        result = algo.process(case.request)
        elapsed = time.perf_counter() - started

        remember("время обработки кадра", f"{width}×{height}: {elapsed:.1f} с")
        _check_result(result, case.request.cube)
        assert elapsed <= settings.runtime_budget_seconds, (
            f"обработка кадра {width}×{height} заняла {elapsed:.1f} с "
            f"при бюджете {settings.runtime_budget_seconds:.0f} с"
        )


def _require_reference(settings: Settings, what: str) -> None:
    """Пропустить проверку, опирающуюся на эталонную модель, если объявлено иное."""
    if settings.is_reference:
        return
    reasons = "; ".join(settings.deviations) or "отступления не объявлены"
    pytest.skip(f"профиль {settings.profile}: {what} не сверяется ({reasons})")


def _check_result(result: ProcessingResult, cube: SpectralCube) -> None:
    """Проверить, что результат пригоден к сохранению рабочим местом."""
    shape = (cube.height, cube.width)
    assert isinstance(result, ProcessingResult), (
        f"вернулся {type(result).__name__}, а не ProcessingResult"
    )

    assert result.concentrations, "карты концентраций пусты"
    for symbol, values in result.concentrations.items():
        _check_map(values, shape, f"карта концентраций {symbol}")

    _check_map(result.thb_map, shape, "карта THb")

    mask = result.lesion_mask
    assert mask.dtype == np.bool_, f"маска поражения имеет тип {mask.dtype}, ожидается bool"
    assert mask.shape == shape, f"размер маски {mask.shape} не совпадает с кадром {shape}"

    metrics = result.metrics
    for name in ("s_coefficient", "mean_lesion_thb", "mean_skin_thb"):
        value = getattr(metrics, name)
        assert isinstance(value, float), f"показатель {name} имеет тип {type(value).__name__}"
        assert np.isfinite(value), f"показатель {name} не число: {value}"
    assert metrics.s_coefficient >= 0.0, "коэффициент s отрицателен"

    segmentation = result.segmentation
    assert segmentation.method.strip(), "способ выделения очага не назван"
    assert np.isfinite(segmentation.threshold), "порог выделения не число"
    assert segmentation.gaussian_sigma >= 0.0, "сглаживание отрицательно"

    assert isinstance(result.notes, str), "пояснение для врача должно быть строкой"


def _check_map(values: np.ndarray, shape: tuple[int, int], title: str) -> None:
    """Проверить одну карту: тип, размер и отсутствие пропусков."""
    assert isinstance(values, np.ndarray), f"{title}: получен {type(values).__name__}"
    assert values.dtype == np.float32, (
        f"{title}: тип {values.dtype}, контракт объявляет float32 - "
        "приведите результат np.asarray(..., dtype=np.float32)"
    )
    assert values.shape == shape, f"{title}: размер {values.shape} не совпадает с кадром {shape}"
    assert np.isfinite(values).all(), (
        f"{title}: есть NaN или бесконечности - они уйдут в карточку сеанса молча"
    )
