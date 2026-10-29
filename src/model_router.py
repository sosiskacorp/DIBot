"""Выбор модели Ollama под конкретную задачу.

Задача роутера — избавить остальной код от знания о том, какая модель где
лежит. Вызывающий код говорит "это математика" или просто передаёт текст,
а роутер возвращает готовый маршрут: модель, запасные модели, температуру,
режим размышлений и размер контекста.

Задачи (ключи в config/settings.json -> routing):
    chat     — обычный диалог
    notes    — конспект по лекционному материалу
    summary  — краткая выжимка, пересказ
    math     — пошаговое решение задач
    code     — код, включая генерацию проверок
    latex    — оформление формул и документов
    vision   — работа со сканами и фотографиями
    verify   — критика чужого ответа (должна идти на другой модели)
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"

TASKS: tuple[str, ...] = (
    "chat",
    "notes",
    "summary",
    "math",
    "code",
    "latex",
    "vision",
    "verify",
)

# Порядок важен: срабатывает первое совпадение. "Конспект по пределам" —
# это notes, а не math, поэтому текстовые задачи идут раньше вычислительных.
_TASK_PATTERNS: Sequence[tuple[str, Sequence[str]]] = (
    (
        "notes",
        (
            r"конспект",
            r"тезис",
            r"оформи\s+(лекци|материал)",
            r"выдели\s+(главное|основное)",
            r"структурируй",
            r"план\s+лекции",
        ),
    ),
    (
        "summary",
        (
            r"кратко\s+(перескажи|изложи)",
            r"перескажи",
            r"выжимк",
            r"суммаризир",
            r"сократи\s+текст",
            r"о\s?ч[её]м\s+(это|лекция)",
        ),
    ),
    (
        "latex",
        (
            r"\\documentclass",
            r"\\begin\{(equation|align|matrix|pmatrix|document)\}",
            r"\blatex\b",
            r"\btex\b",
            r"оформи\s+(формул|решени)",
        ),
    ),
    (
        "code",
        (
            r"```",
            r"\bdef\s+\w+\s*\(",
            r"\b(class|import|from)\s+\w+",
            r"\bpython\b",
            r"\bнапиши\s+(код|скрипт|функци)",
            r"\bрегулярн\w+\s+выражени",
            r"\bsympy\b",
        ),
    ),
    (
        "math",
        (
            r"\bреши(те)?\b",
            r"\bвычисли(те)?\b",
            r"\bдокажи(те)?\b",
            r"\bнайди(те)?\b.{0,60}\b(предел|производн|интеграл|матриц|определител|ранг|собственн)",
            r"\b(предел|производн|интеграл|детерминант|определител|матриц|собственн\w+\s+значени)\w*",
            r"\bсистем\w+\s+линейн",
            r"\bуравнени",
            r"\bвероятност",
            r"\bградиент",
            r"\bпроизводн\w*",
            r"(?<![A-Za-zА-Яа-я])\d+\s*[xXyYzZ]\s*[+\-=^]",
            r"\bsin\b|\bcos\b|\btg\b|\blog\b|\bln\b|\blim\b",
        ),
    ),
)

_COMPILED: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = tuple(
    (task, tuple(re.compile(p, re.IGNORECASE) for p in patterns))
    for task, patterns in _TASK_PATTERNS
)

# Теоретический вопрос: «что такое производная», «объясни метод Крамера».
# Такие вопросы не требуют пошагового счёта, и deepseek-r1 на них тратит
# минуты рассуждений без пользы. Отправляем их универсальной модели.
_CONCEPTUAL_RE = re.compile(
    r"(?i)\b("
    r"что\s+такое|"
    r"объясни(те)?|"
    r"расскажи(те)?|"
    r"в\s+ч[её]м\s+(разница|смысл|отличие)|"
    r"чем\s+отличается|"
    r"почему|"
    r"как\s+(работает|устроен\w*|считается|находится|понимать)|"
    r"дай(те)?\s+определение|"
    r"сформулируй(те)?|"
    r"опиши(те)?"
    r")\b"
)

# Явная команда что-то посчитать отменяет «теоретичность».
_SOLVE_RE = re.compile(
    r"(?i)\b("
    r"реши(ть|те)?|реша(ть|йте)?|"
    r"вычисли(ть|те)?|"
    r"найди(те)?|найти|"
    r"докажи(те)?|доказать|"
    r"построй(те)?|"
    r"посчитай(те)?|"
    r"проверь(те)?|"
    r"упрости(те)?|"
    r"разложи(те)?|"
    r"приведи(те)?"
    r")\b"
)

# Конкретные данные: матрица, выражение с переменной, сравнение.
_CONCRETE_RE = re.compile(r"\[\[|\blim\b|\d\s*[xXyYzZ]\b|[=<>≤≥]")


def classify(text: str, has_images: bool = False) -> str:
    """Угадать задачу по тексту запроса.

    Возвращает ключ задачи. При неуверенности — "chat"; лучше переспросить
    модель-универсал, чем гонять reasoning там, где он не нужен.
    """
    if has_images:
        return "vision"
    if not text:
        return "chat"

    # Теорию без конкретных данных и без команды «реши» считаем разговором.
    if (
        _CONCEPTUAL_RE.search(text)
        and not _SOLVE_RE.search(text)
        and not _CONCRETE_RE.search(text)
    ):
        return "chat"

    for task, patterns in _COMPILED:
        for pattern in patterns:
            if pattern.search(text):
                return task
    return "chat"


@dataclass(frozen=True)
class Route:
    """Готовый маршрут: чем отвечать и с какими параметрами."""

    task: str
    model: str
    fallback: tuple[str, ...]
    temperature: float
    think: bool
    num_ctx: int
    options: Mapping[str, Any]

    @property
    def chain(self) -> tuple[str, ...]:
        """Модель и запасные варианты в порядке приоритета."""
        return (self.model, *self.fallback)


class Router:
    """Читает config/settings.json и отвечает на вопрос "какой моделью?"."""

    def __init__(self, config: Mapping[str, Any] | str | Path | None = None):
        if config is None:
            config = _load_config(DEFAULT_CONFIG_PATH)
        elif isinstance(config, (str, Path)):
            config = _load_config(config)
        self.config = dict(config)

        self.routing: Mapping[str, Any] = self.config.get("routing") or {}
        self.models: Mapping[str, Any] = self.config.get("models") or {}
        self.verification: Mapping[str, Any] = self.config.get("verification") or {}

        if not self.routing:
            raise ValueError(
                "В конфиге нет секции 'routing'. Ожидается словарь задача -> {model, fallback, ...}."
            )

    # ------------------------------------------------------------------ выбор

    def route(
        self,
        task: str | None = None,
        text: str | None = None,
        has_images: bool = False,
    ) -> Route:
        """Вернуть маршрут. Задачу можно указать явно или дать угадать по тексту."""
        resolved_task = task or classify(text or "", has_images=has_images)
        if resolved_task not in self.routing:
            raise KeyError(
                f"Неизвестная задача {resolved_task!r}. Доступны: {', '.join(sorted(self.routing))}"
            )

        spec = self.routing[resolved_task]
        model = spec["model"]
        fallback = tuple(m for m in spec.get("fallback", ()) if m != model)
        num_ctx = self.num_ctx_for(model)

        options = {
            "temperature": float(spec.get("temperature", 0.3)),
            "num_ctx": num_ctx,
        }
        options.update(spec.get("options") or {})

        return Route(
            task=resolved_task,
            model=model,
            fallback=fallback,
            temperature=float(spec.get("temperature", 0.3)),
            think=bool(spec.get("think", False)),
            num_ctx=num_ctx,
            options=options,
        )

    def chain(self, task: str) -> tuple[str, ...]:
        """Все модели, которыми можно закрыть задачу, включая запасные."""
        return self.route(task).chain

    def verifier(self, solver_model: str, task: str = "verify") -> str | None:
        """Выбрать модель для проверки ответа, обязательно отличную от решавшей.

        Смысл в том, что модель, повторившая свою же ошибку, не увидит её.
        Если все кандидаты совпали с решавшей — вернёт None, и вызывающий код
        должен честно сказать "независимая проверка недоступна".
        """
        if not self.verification.get("independent_from_solver", True):
            return self.route(task).model

        for candidate in self.route(task).chain:
            if candidate != solver_model:
                return candidate

        for candidate in self.models:
            if candidate != solver_model:
                return candidate
        return None

    def num_ctx_for(self, model: str) -> int:
        """Размер контекста с поправкой на реальные лимиты модели."""
        spec = self.models.get(model) or {}
        wanted = int(spec.get("num_ctx", 8192))
        limit = int(spec.get("max_ctx", 0)) or wanted
        # Оставляем место под ответ, поэтому не упираемся ровно в максимум.
        return max(1024, min(wanted, limit))

    def model_spec(self, model: str) -> Mapping[str, Any]:
        return self.models.get(model) or {}

    def tasks_for(self, model: str) -> list[str]:
        """Обратная задача: в каких маршрутах участвует модель."""
        found = []
        for task, spec in self.routing.items():
            if spec.get("model") == model or model in (spec.get("fallback") or ()):
                found.append(task)
        return sorted(found)


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    router = Router()
    print(f"{'задача':10} {'модель':18} {'think':6} {'temp':5}  запасные")
    print("-" * 78)
    for name in TASKS:
        try:
            r = router.route(name)
        except KeyError:
            continue
        print(
            f"{name:10} {r.model:18} {str(r.think):6} {r.temperature:<5}  {', '.join(r.fallback) or '—'}"
        )
        print(f"{'':10} контекст: {r.num_ctx}")

    print("\nпроверка независимости:")
    for solver in ("deepseek-r1:8b", "qwen3.5:9b", "qwen2.5-coder:7b"):
        print(f"  решал {solver:18} -> проверяет {router.verifier(solver)}")
