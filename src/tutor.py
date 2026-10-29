"""Оркестратор: вопрос -> контекст -> решение -> проверка.

Это единственное место, где сходятся все слои. Правила:
  * если в материалах есть ответ — он идёт в контекст со ссылками;
  * если материалов нет, модель отвечает по общим знаниям, но это помечено;
  * математика и код проверяются автоматически, а не «на глаз»;
  * в ответе всегда видно, какая модель отвечала и чем проверено.
"""

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ingest import format_math_markdown
from llm_client import ChatResult, LLMClient, LLMError
from materials import MaterialStore
from model_router import classify
from notes import CITATION_RE
from search_client import SearchError, SearchResult, WebSearch
from verification import VerificationReport, Verifier

TUTOR_SYSTEM = """Ты — DIBot, учебный ИИ-ассистент студента НГТУ. Твоя задача — помогать с учёбой,
объяснять теорию и решать задачи так, чтобы студент мог воспроизвести решение сам.

Твои возможности (если спрашивают «что ты можешь / кто ты»):
- Работа с базой курсов и лекций из DiSpace НГТУ (поиск по лекциям, методичкам, формулам и заданиям).
- Распознавание прикреплённых фотографий и сканов тетрадей/заданий (модель зрения `qwen3-vl`).
- Пошаговое решение задач по высшей математике, линейной алгебре, матанализу и физике (`deepseek-r1`) с символьной проверкой ответа через `SymPy`.
- Написание, разбор и отладка кода (Python, C/C++, алгоритмы) и оформление формул в LaTeX (`qwen2.5-coder`).
- Составление проверяемых конспектов со ссылками `[#id]` на фрагменты лекций и генерация тренировочных вариантов.

Правила работы с материалами курсов студента:
1. Если предоставлены материалы из лекций (блок «ТЕОРЕТИЧЕСКИЕ МАТЕРИАЛЫ КУРСА»):
   - Обязательно опирайся на этот материал! Он взят напрямую из учебных курсов студента в DiSpace НГТУ.
   - В начале решения в блоке «Теоретическая база» явно назови дисциплину и тему лекции (например: «По материалам курса {курс}, лекции «{тема}» [#{id}]...»).
   - Выпиши исходную формулу, теорему или определение в тех обозначениях, которые даны в лекции.
   - Обязательно сошлись на соответствующий фрагмент лекции: `[#{id}]`.
   - Если предоставленный контекст явно не относится к вопросу (например, вопрос общий или о твоих возможностях) — не упоминай посторонние курсы.
2. Все математические формулы и символы оформляй ТОЛЬКО через `$...$` (строчные, без пробелов сразу после открывающего и перед закрывающим `$`) и `$$...$$` с новой строки (крупные формулы, матрицы, системы, пределы, интегралы). Не используй `\\(...\\)` и `\\[...\\]`.
3. Любые вычисления показывай пошагово, а не только результат.
4. Для расчётных задач оформляй решение как в тетради: Дано, Ход решения, и в конце отдельными строками:
   - `Ответ: ...` (классическая математическая запись);
   - `Строка для ввода в поле DiSpace: ...` (строго по правилам DiSpace: БЕЗ ПРОБЕЛОВ, константы `pi`, `inf`, `e`/`exp`, аргументы функций в скобках `sin(...)`, умножение ВСЕГДА через `*` (`2*x`), показатели и индексы в скобках `2^(3)`, `(35)^(6)`, `a_(1)`, дроби со скобками в числителе/знаменателе `(x^(2)+1)/4`, корни `sqrt(3)` или `sqrt[3](x)` или `3^(1/2)`).
5. Если вопрос теоретический, общий или разговорный — отвечай связно и по делу без блоков «Дано / Ход решения / Ответ»."""

_META_QUERY_RE = re.compile(
    r"(?i)^\s*(?:привет\w*|здравствуй\w*|добрый\s+\w+|кто\s+ты|что\s+ты\s+(?:можешь|умеешь|такое|за\s+бот)|"
    r"чем\s+(?:ты\s+)?можешь\s+помочь|какие\s+у\s+тебя\s+(?:возможности|функции)|"
    r"как\s+(?:с\s+тобой|тобой)\s+(?:работать|пользоваться)|расскажи\s+о\s+себе|помощь|help)\s*[?!.]*\s*$"
)


def _is_meta_or_conversational(query: str) -> bool:
    """Определить, является ли запрос приветствием или вопросом о самом боте (не требующим поиска по лекциям)."""
    q = (query or "").strip()
    if not q:
        return False
    if _META_QUERY_RE.match(q):
        return True
    lowered = q.lower()
    if len(q) <= 60 and any(
        phrase in lowered
        for phrase in (
            "что ты можешь",
            "что ты умеешь",
            "кто ты такой",
            "чем можешь помочь",
            "твои возможности",
            "какие курсы у тебя",
        )
    ):
        return True
    return False


def _is_generic_image_query(query: str) -> bool:
    """Проверить, является ли текстовый запрос просто общей фразой о картинке без конкретных терминов."""
    q = (query or "").strip().lower()
    if not q:
        return True
    generic_words = (
        "реши задачу", "реши задание", "с картинки", "с фото", "на фото",
        "на картинке", "разбери фото", "помоги с фото", "помоги решить",
        "что на картинке", "что на фото", "найди ответ", "реши тест",
        "разбери и реши задание на изображении", "реши", "помоги",
    )
    clean = q
    for g in generic_words:
        clean = clean.replace(g, "")
    clean = re.sub(r"[^\w\s]", "", clean).strip()
    return len(clean) < 4


_FOLLOWUP_TRIGGERS_RE = re.compile(
    r"(?i)\b("
    r"почему|откуда|как\s+так|а\s+если|что\s+значит|объясни|поясни|поясните|разъясни|подробнее|"
    r"в\s+шаге|в\s+ответе|в\s+формуле|в\s+строке|в\s+знаке|знак|минус|плюс|"
    r"вариант\w*|почему\s+именно|почему\s+не|а\s+почему|а\s+как|а\s+откуда|а\s+зачем|а\s+где|"
    r"там|тут|здесь|в\s+решении|в\s+тво[её]м\s+ответе|в\s+тво[её]м\s+решении|выше|ранее|"
    r"перв\w+\s+шаг|втор\w+\s+шаг|трет\w+\s+шаг|четверт\w+\s+шаг|не\s+понял|не\s+понятно|"
    r"а\s+в\s+чем|в\s+чём|переход|тождеств"
    r")\b"
)

_NEW_PROBLEM_RE = re.compile(
    r"(?i)\b("
    r"реши\s+задачу|найдите\s+производную|вычислите\s+предел|решите\s+уравнение|найдите\s+интеграл|"
    r"дана\s+функция|дана\s+матрица|новое\s+задание|новая\s+задача|следующая\s+задача"
    r")\b"
)


def _is_followup_question(query: str, history: Sequence[dict[str, str]] | None) -> bool:
    """Определить, является ли реплика уточняющим вопросом по предыдущему решению в диалоге."""
    if not history:
        return False
    has_assistant = any(m.get("role") == "assistant" for m in history)
    if not has_assistant:
        return False

    q = (query or "").strip()
    if not q:
        return False
    if _NEW_PROBLEM_RE.search(q):
        return False

    if _FOLLOWUP_TRIGGERS_RE.search(q):
        return True

    # Короткие уточняющие реплики без признаков новой задачи
    if len(q) <= 75 and any(
        w in q.lower()
        for w in (
            "почему", "откуда", "как", "знак", "минус", "плюс", "шаг", "строк",
            "ответ", "вариант", "там", "тут", "это", "где", "зачем", "а почему"
        )
    ):
        return True

    return False


PRACTICE_SYSTEM = """Ты составляешь тренировочные задания по учебной теме.

Отвечай ТОЛЬКО JSON:
{"tasks": [{"statement": "условие (формулы в $...$)", "answer": "краткий ответ",
            "solution": "подробное тетрадное пошаговое решение (формулы в $...$ и $$...$$)", "difficulty": "лёгкий|средний|сложный"}]}

Требования:
- задания должны быть решаемы теми методами, что в материалах;
- ответ должен быть однозначным и проверяемым;
- все слэши LaTeX внутри JSON экранируй двойным слэшем (\\\\frac, \\\\lim, \\\\sqrt);
- условие самодостаточно: никаких «см. выше» и ссылок на рисунки."""

NOTEBOOK_SOLUTION_SYSTEM = """Ты — образцовый студент-отличник и преподаватель НГТУ.
Твоя задача — оформить решение учебного задания из курса DiSpace строго по всем правилам университетского тетрадного оформления, опираясь на теоретический материал текущей лекции.

Структура тетрадного оформления:
1. **📋 Дано и требуется найти:**
   Чёткая математическая запись исходного условия задачи, функции, точки/интервала или предложенных вариантов ответа.
2. **📖 Теоретическая база (по материалам лекции):**
   Укажи точное определение, теорему, формулу из лекции НГТУ (в тех же обозначениях: $\\Delta t$, $\\Delta x$, $f'(x_0) = \\lim_{\\Delta x \\to 0} \\frac{f(x_0+\\Delta x)-f(x_0)}{\\Delta x}$), которая применяется в этой задаче.
3. **✍️ Подробный тетрадный ход решения:**
   Расписывай каждый шаг последовательно, без пропуска выкладок:
   - подстановка значений аргумента и точки $t_0$ / $x_0$;
   - тождественные преобразования;
   - промежуточные упрощения.
   Все строчные формулы оформляй в `$...$`, а выносные формулы, пределы, дроби и матрицы — на отдельных строках в `$$...$$`.

КРИТИЧЕСКОЕ ПРАВИЛО ДЛЯ ТЕСТОВ С ВАРИАНТАМИ ОТВЕТОВ:
Если в задании предложены варианты ответов (Вариант 1, 2, 3, 4 или А, Б, В, Г):
ОБЯЗАТЕЛЕН метод последовательного отсева (элиминации) каждого варианта!
В ходе решения обязательно сделай раздел:
«🔍 Анализ и отбор вариантов ответа:»
Для КАЖДОГО варианта по порядку:
- Запиши формулу варианта;
- Сравни её с точным теоретическим выводом;
- Если вариант НЕВЕРНЫЙ: напиши точную математическую причину ошибки (например: «не та переменная под пределом», «ошибка в знаке», «неверно вычислено $f(t_0)$», «лишняя единица в числителе»);
- Если вариант ВЕРНЫЙ: строго покажи его полное соответствие теории и определению.

4. **✅ Итоговый ответ и точный ввод в DiSpace:**
   - Если в задании были варианты ответов:
     `Ответ: Вариант N ($формула$)`
     `Строка для выбора/ввода в DiSpace: N` (только цифра или буква выбранного варианта).
   - Если расчётная задача с открытым вводом:
     `Ответ: ...` (красивая математическая запись)
     `Строка для ввода в поле DiSpace: ...` (строго по официальным правилам DiSpace НГТУ ниже).

ОФИЦИАЛЬНЫЕ ПРАВИЛА ВВОДА ОТВЕТОВ В СИСТЕМЕ DiSpace НГТУ (СОБЛЮДАЙ СТРОЖАЙШЕ):
1. ГЛАВНОЕ: НИКАКИХ ПРОБЕЛОВ! В строке для DiSpace вообще не должно быть пробелов: `12+2/x^(2)`, а не `12 + 2/x^2`. Любые пробелы ломают синтаксический анализатор DiSpace!
2. Константы: число пи пишется как `pi`, бесконечность — `inf` или русской буквой `б`, экспонента — латинская `e` или `exp`.
3. Имена функций пишутся латинскими буквами, аргументы функций ВСЕГДА в круглых скобках: `sin(2)`, `cos(pi)`, `ln(1/x)`, `3*tg(2*pi*x)`.
4. Знаки арифметических операций не пропускаются, произведение ВСЕГДА пишется `*`: `sin(2*x)`, `cos(pi*x)`, `3*tg(2*pi*x)`, `2*x`, `x*cos(x)`.
5. Степени и индексы: степень пишется через `^`, индекс через `_`. Значения показателей и индексов ВСЕГДА пишутся в скобках, если основание степени состоит более чем из одного знака его также следует взять в скобки: `2^(3)`, `e^(-2*x)`, `a_(1)`, `(35)^(6)`. Порядок написания индекса и степени не важен: `x^(2)_(1)` или `x_(1)^(2)`.
6. Дроби: пишется через `/`. Если числитель или знаменатель состоит более чем из одного символа, их следует записывать в скобках: `2/3`, `(12)/(17)`, `2/(3*x)`, `(x^(2)+1)/4`, `(x+3*x^(2))/(x^(2)-1)`, `2^(2/3)`. (Выражения `2/(3*x)` и `2/3*x` разные!).
7. Арифметические корни: записываются как функция `sqrt[]()` — где в квадратных скобках пишется степень корня, а в круглых подкоренное выражение. У квадратных корней степень [2] можно опустить: `sqrt(3)`, `sqrt[3](x)`, `sqrt[4](x^(2)+1)`. Либо через степень с дробным показателем: `3^(1/2)`, `(pi)^(1/4)`, `(x^(2)-pi)^(1/3)`.
8. Интегралы: `int(2*x*dx)` или с пределами `int_(2)^(3)((2*x-1)/4*dx)`.

Если студент прислал свой вариант ответа или решения на проверку — сначала внимательно проверь его: похвали за верные шаги, укажи точную строчку с ошибкой (если она есть) и затем приведи полное эталонное тетрадное решение."""

VISION_EXTRACT_SYSTEM = """Ты — модуль оптического распознавания учебных материалов и задач.
Твоя задача — максимально точно переписать весь текст, условия задач, таблицы и математические
формулы с прикреплённого изображения.
Правила:
1. Все математические формулы, матрицы, пределы, интегралы и уравнения записывай в LaTeX ($...$ или $$...$$).
2. Сохраняй точные числа, знаки и индексы. Не пропускай условия.
3. Если на фото изображена схема, график или чертёж — подробно опиши его структуру и данные.
4. Выводи только распознанное условие и данные без лишних вступлений."""


_DISPACE_INPUT_RE = re.compile(
    r"(?im)^[ \t>*]*(?:строка для (?:выбора/)?ввода в (?:поле )?dispace|строка для dispace|поле dispace|формат dispace|ответ для dispace|ввод в dispace)\s*[:：][ \t]*(.+)$"
)


def format_dispace_input(s: str) -> str:
    """Привести математическое выражение к строгому синтаксису DiSpace НГТУ."""
    if not s:
        return ""
    out = s.strip()
    out = re.sub(
        r"(?im)^[ \t>*]*(?:ответ|строка для (?:выбора/)?ввода в (?:поле )?dispace|строка для dispace|поле dispace|dispace|answer)\s*[:：]?\s*",
        "",
        out,
    ).strip()
    out = out.replace("$", "").replace('"', "").replace("'", "").strip()

    # Если номер варианта: "Вариант 2" -> "2"
    m_var = re.match(r"(?i)^вариант\s*(\d+|[а-яёa-z])\s*$", out)
    if m_var:
        return m_var.group(1)
    m_var_formula = re.match(r"(?i)^вариант\s*(\d+|[а-яёa-z])\s*\((.+)\)$", out)
    if m_var_formula:
        return m_var_formula.group(1)

    # 1. LaTeX дроби \frac{a}{b} -> (a)/(b)
    while r"\frac" in out:
        out = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))", out)

    # 2. Корни: \sqrt[n]{x} -> sqrt[n](x), \sqrt{x} -> sqrt(x)
    out = re.sub(r"\\sqrt\s*\[\s*([^\[\]]+)\s*\]\s*\{([^{}]*)\}", r"sqrt[\1](\2)", out)
    out = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)", out)

    # 3. Замены математических констант и функций
    subs = [
        (r"\\pi\b", "pi"),
        (r"\\infty\b", "inf"),
        (r"\\cdot|\\times", "*"),
        (r"\\left|\\right", ""),
        (r"\\sin\b", "sin"),
        (r"\\cos\b", "cos"),
        (r"\\tg\b|\\tan\b", "tg"),
        (r"\\ctg\b|\\cot\b", "ctg"),
        (r"\\ln\b", "ln"),
        (r"\\log\b", "log"),
        (r"\\exp\b", "exp"),
    ]
    for pat, rep in subs:
        out = re.sub(pat, rep, out)

    # 4. Пропущенные умножения перед удалением пробелов
    out = re.sub(r"([a-zA-Z0-9\)])\s+(sin|cos|tg|ctg|ln|log|exp|sqrt|int)\b", r"\1*\2", out)

    # 5. Убираем ВСЕ пробелы (критическое требование DiSpace НГТУ)
    out = re.sub(r"\s+", "", out)

    # 6. Степень и индексы без скобок: x^2 -> x^(2), a_1 -> a_(1)
    out = re.sub(r"\^(-?[a-zA-Z0-9]+)(?!\()", r"^(\1)", out)
    out = re.sub(r"\_(-?[a-zA-Z0-9]+)(?!\()", r"_(\1)", out)

    # 7. Основание степени из нескольких символов: 35^(6) -> (35)^(6)
    out = re.sub(r"(?<![a-zA-Z0-9_\)])(\d{2,})\^\(", r"(\1)^(", out)

    # 8. Восстанавливаем пропущенные умножения
    out = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", out)
    out = re.sub(r"(\))([a-zA-Z0-9])", r"\1*\2", out)
    out = re.sub(r"\)\(", r")*(", out)
    # Сохраняем синтаксис интегралов без знака умножения перед подынтегральным выражением
    out = re.sub(r"(int(?:_\([^\)]+\)\^\([^\)]+\)|_\([^\)]+\)|\([^\)]+\)))\*\(", r"\1(", out)

    # Буква/число перед скобкой, если это не функция (sin, cos, tg, ctg, ln, log, exp, sqrt, int)
    funcs = ("sin", "cos", "tg", "ctg", "ln", "log", "exp", "sqrt", "int")
    def _repl_paren(m: re.Match) -> str:
        prefix = m.group(1)
        for fn in funcs:
            if prefix.endswith(fn):
                return prefix + "("
        return prefix + "*("

    out = re.sub(r"([a-zA-Z0-9]+)\(", _repl_paren, out)

    # Двойные скобки в дробях упрощаем: ((a)/(b)) -> (a)/(b)
    out = re.sub(r"^\(\((.+)\)\)$", r"(\1)", out)

    return out


def extract_dispace_input(text: str) -> str:
    """Извлечь строку для ввода в DiSpace из текста решения."""
    if not text:
        return ""
    matches = _DISPACE_INPUT_RE.findall(text)
    if matches:
        raw_val = matches[-1].strip().strip("`$* ")
        return format_dispace_input(raw_val)

    ans_m = re.findall(
        r"(?im)^[ \t>*]*(?:\*\*)?(?:итоговый ответ|ответ|answer)(?:\*\*)?[ \t]*[:：][ \t]*(?:\*\*)?(.+)$",
        text,
    )
    if ans_m:
        raw_ans = ans_m[-1].strip().strip("`$* ")
        v_match = re.search(r"(?i)\bвариант\s*(\d+|[а-яёa-z])\b", raw_ans)
        if v_match:
            return v_match.group(1)
        return format_dispace_input(raw_ans)

    return ""


@dataclass
class Answer:
    question: str
    text: str
    task: str
    model: str
    seconds: float
    sources: list[str] = field(default_factory=list)
    verification: VerificationReport | None = None
    context_used: str = "нет"
    thinking: str = ""
    extracted_image_text: str = ""
    dispace_input: str = ""

    @property
    def verdict(self) -> str:
        return self.verification.verdict if self.verification else "не проверялось"

    def as_text(self) -> str:
        lines = []
        if self.extracted_image_text:
            lines.extend(["**Распознано с фото:**", self.extracted_image_text, ""])
        lines.extend([self.text, ""])
        if self.dispace_input:
            lines.extend([f"**Строка для поля ввода DiSpace:** `{self.dispace_input}`", ""])
        lines.append("---")
        badges = [f"модель: {self.model}", f"задача: {self.task}", f"контекст: {self.context_used}"]
        if self.sources:
            badges.append(f"источники: {len(self.sources)}")
        lines.append(" · ".join(badges))
        if self.verification:
            lines.append(self.verification.as_text())
        return "\n".join(lines)


class Tutor:
    def __init__(
        self,
        llm: LLMClient | None = None,
        store: MaterialStore | None = None,
        search: WebSearch | None = None,
        verify_tasks: Sequence[str] = ("math", "latex", "code"),
    ):
        self.llm = llm or LLMClient()
        self.store = store or MaterialStore()
        self.search = search or WebSearch(self.llm.config.get("search", {}))
        self.verifier = Verifier(self.llm)
        self.verify_tasks = tuple(verify_tasks)

    def compact_history(
        self,
        history: Sequence[dict[str, str]],
        keep_recent: int = 3,
    ) -> list[dict[str, str]]:
        """Сжатие контекста: суммаризация старых сообщений диалога для экономии токенов."""
        if not history or len(history) <= keep_recent:
            return list(history)

        older = list(history[:-keep_recent])
        recent = list(history[-keep_recent:])

        total_chars = sum(len(m.get("content", "")) for m in older)
        if total_chars < 1000:
            return list(history)

        dialogue_text = "\n\n".join(
            f"{'Пользователь' if m.get('role') == 'user' else 'Ассистент'}: {m.get('content', '')[:500]}"
            for m in older
        )
        prompt = (
            "Сделай краткую, но ёмкую выжимку предыдущей части диалога для экономии контекста нейросети.\n"
            "Сохрани все ключевые факты, условия решаемых задач, введённые формулы и выводы.\n"
            "Не пиши вводных слов, пиши строго суть в 2–4 пунктах:\n\n"
            f"{dialogue_text}"
        )
        try:
            res = self.llm.chat(
                task="summary",
                messages=[{"role": "user", "content": prompt}],
                timeout=25.0,
            )
            summary_text = res.text.strip()
            if summary_text:
                compacted_turn = {
                    "role": "user",
                    "content": f"📌 [Контекст предыдущей беседы]:\n{summary_text}",
                }
                return [compacted_turn, *recent]
        except Exception:
            pass

        truncated_older = []
        for m in older:
            c = m.get("content", "")
            truncated_older.append(
                {
                    "role": m.get("role", "user"),
                    "content": c[:200] + ("…" if len(c) > 200 else ""),
                }
            )
        return [*truncated_older, *recent]

    # --------------------------------------------------------------- вопрос

    def ask(
        self,
        question: str,
        course: str | None = None,
        use_web: bool = False,
        verify: bool | None = None,
        history: Sequence[dict[str, str]] | None = None,
        task: str | None = None,
        images: Sequence[str | Path | bytes] | None = None,
        on_stream: Any = None,
    ) -> Answer:
        started = time.perf_counter()
        if history and (len(history) >= 6 or sum(len(m.get("content", "")) for m in history) >= 4500):
            history = self.compact_history(history, keep_recent=3)
        clean_question = (question or "").strip() or "Разбери и реши задание на изображении."
        extracted_image_text = ""
        vision_model_used = ""

        # Шаг 1: если прикреплены фото, извлекаем условие и формулы через vision-модель (qwen3-vl:8b)
        if images:
            if on_stream is not None:
                try:
                    on_stream("status", "👁️ Распознаю условие и формулы с изображения (qwen3-vl)…", "qwen3-vl:8b")
                except Exception:
                    pass
            vision_prompt = (
                "Перепиши точный текст условия задачи, все предложенные варианты ответов и все формулы (в LaTeX) с изображения."
                if not question.strip()
                else f"Запрос пользователя: {clean_question}\n\nПерепиши весь текст, варианты ответов и формулы (в LaTeX) с изображения, необходимые для решения задачи."
            )
            try:
                vision_res = self.llm.chat(
                    [
                        {"role": "system", "content": VISION_EXTRACT_SYSTEM},
                        {"role": "user", "content": vision_prompt},
                    ],
                    task="vision",
                    images=images,
                    on_stream=on_stream,
                )
                extracted_image_text = vision_res.text.strip()
                vision_model_used = vision_res.model
            except Exception as exc:
                extracted_image_text = ""

        # Совмещаем вопрос пользователя и распознанный текст с фото для поиска и классификации
        combined_query = (
            f"{clean_question}\n{extracted_image_text}".strip()
            if extracted_image_text
            else clean_question
        )

        is_followup = bool(not images and _is_followup_question(clean_question, history))

        if task and task != "vision":
            resolved_task = task
        elif is_followup:
            # Наследуем режим задачи от предыдущего хода диалога, если решалась математика или код
            last_assistant_msg = ""
            for m in reversed(history or []):
                if m.get("role") == "assistant" and m.get("content"):
                    last_assistant_msg = m["content"]
                    break
            has_math_context = any(
                ind in last_assistant_msg
                for ind in ("$", "\\lim", "\\frac", "\\int", "\\sqrt", "Дано", "Ход решения", "Ответ:", "Вариант")
            )
            resolved_task = "math" if has_math_context else "chat"
        else:
            resolved_task = classify(combined_query, has_images=False)

        parts: list[str] = []
        sources: list[str] = []
        used = "нет"
        primary_course = ""

        if extracted_image_text:
            extracted_image_text = format_math_markdown(extracted_image_text)
            parts.append("РАСПОЗНАННЫЙ ТЕКСТ И ФОРМУЛЫ С ФОТО:\n" + extracted_image_text)
            used = "фото"

        is_meta = not extracted_image_text and not is_followup and _is_meta_or_conversational(clean_question)
        if is_meta:
            loaded_courses = self.store.list_courses()
            if loaded_courses:
                parts.append(
                    "ДОСТУПНЫЕ СЕЙЧАС В БАЗЕ КУРСЫ СТУДЕНТА:\n"
                    + "\n".join(f"- {c}" for c in loaded_courses)
                )
        elif is_followup:
            # При уточняющем вопросе не ищем вслепую по стоп-словам вроде «знак минус», сохраняем контекст предыдущего решения
            used = "контекст диалога"
        else:
            hits = self.store.search(combined_query, k=4, course=course, min_score=2.0)
            if hits:
                primary_course = hits[0][0].course
                # Если запрос точно сошёлся с курсом высшей математики, решаем специализированной моделью math
                if resolved_task == "chat" and primary_course in ("Математический анализ", "Линейная алгебра"):
                    resolved_task = "math"

                blocks, chars_used = [], 0
                for chunk, _score in hits:
                    block = (
                        f"📘 ДИСЦИПЛИНА (КУРС): «{chunk.course}»\n"
                        f"📑 ТЕМА / ЛЕКЦИЯ: «{chunk.title}» (фрагмент {chunk.index + 1}, ссылка [#{chunk.id}])\n"
                        f"{chunk.text}"
                    )
                    if chars_used + len(block) > 4800:
                        break
                    blocks.append(block)
                    chars_used += len(block)
                    sources.append(f"«{chunk.course}» — {chunk.title} (фр. {chunk.index + 1})")
                if blocks:
                    parts.append(
                        f"📚 ТЕОРЕТИЧЕСКИЕ МАТЕРИАЛЫ КУРСА «{primary_course}»:\n"
                        f"Ниже приведены официальные материалы лекций студента из базы курсов DiSpace НГТУ.\n"
                        f"При решении задачи строго опирайся на эти определения, формулы и обозначения:\n\n"
                        + "\n\n---\n\n".join(blocks)
                    )
                    used = "фото + материалы" if extracted_image_text else "материалы"

        web_results: list[SearchResult] = []
        if use_web and not is_meta and not is_followup:
            try:
                web_context, web_results = self.search.context(combined_query, limit=4)
                if web_context:
                    parts.append("ВЕБ-РЕЗУЛЬТАТЫ (проверяй критически):\n" + web_context)
                    used = f"{used} + веб" if used != "нет" else "веб"
            except SearchError as exc:
                parts.append(f"(веб-поиск недоступен: {exc})")

        prompt_parts = []
        if parts:
            prompt_parts.append("\n\n".join(parts))

        display_problem = combined_query if extracted_image_text else clean_question
        if is_followup:
            prompt_parts.append(
                f"УТОЧНЯЮЩИЙ ВОПРОС СТУДЕНТА ПО ПРЕДЫДУЩЕМУ РЕШЕНИЮ:\n{clean_question}\n\n"
                f"ИНСТРУКЦИЯ ПО ОТВЕТУ:\n"
                f"1. Ответь строго в контексте ранее разобранной выше задачи и твоего решения из истории диалога.\n"
                f"2. Подробно и понятно объясни математическую причину, знак, формулу или логический переход, о котором спрашивает студент (почему именно так, какое правило или тождество применено).\n"
                f"3. Все строчные формулы оформляй строго в `$...$`, а крупные выносные выкладки — на отдельных строках в `$$...$$`.\n"
                f"4. Не начинай решение задачи заново с нуля — сфокусируйся на разъяснении шага, вызвавшего вопрос."
            )
        elif resolved_task == "math":
            has_options = bool(re.search(r"(?i)(?:вариант\s*\d|\b[1-4]\)\s*|^\s*[-*]\s*(?:вариант|[A-DА-Г]\)))", display_problem, re.M))
            inst_steps = [
                f"1. В начале ответа обязательно укажи дисциплину «{primary_course or 'Математика'}» и тему лекции, откуда берётся теоретическая база, выпиши формулу в LaTeX и укажи ссылку на фрагмент [#{hits[0][0].id if (not is_meta and hits) else 'источник'}].",
                "2. Подробно распиши ход решения по шагам, подставляя данные задачи в формулу из лекции.",
            ]
            if has_options:
                inst_steps.append(
                    "3. В задании даны варианты ответов! ОБЯЗАТЕЛЕН метод отсева: подробно разбери КАЖДЫЙ вариант (1..N), обоснуй ошибку неверных и строго докажи правильный вариант."
                )
                inst_steps.append(
                    "4. В конце отдельной строкой напиши однозначный «Ответ: Вариант N (формула)» и укажи номер варианта для ввода/выбора в DiSpace."
                )
            else:
                inst_steps.append(
                    "3. В конце отдельной строкой напиши однозначный «Ответ: ...» (например: «Ответ: -1/2» или «Ответ: 18*x^(3)-4*x^(-3)»)."
                )
            prompt_parts.append(
                f"УСЛОВИЕ ЗАДАЧИ:\n{display_problem}\n\n"
                f"ИНСТРУКЦИЯ К РЕШЕНИЮ:\n"
                + "\n".join(inst_steps)
            )
        else:
            prompt_parts.append(f"ВОПРОС:\n{clean_question}")
        user_content = "\n\n".join(prompt_parts)

        messages: list[dict[str, str]] = [{"role": "system", "content": TUTOR_SYSTEM}]
        if history:
            for m in history[-8:]:
                role = str(m.get("role") or "user")
                content = str(m.get("content") or "").strip()
                if role in ("user", "assistant") and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_content})

        # Если пользователь явно выбрал режим "vision" без перенаправления в math/code,
        # передаём также сами изображения в финальный запрос.
        pass_images = images if task == "vision" else None
        result: ChatResult = self.llm.chat(
            messages,
            task=resolved_task if not pass_images else "vision",
            text_hint=combined_query,
            images=pass_images,
            on_stream=on_stream,
        )
        answer_text = format_math_markdown(_strip_prompt_echo(result.text, TUTOR_SYSTEM))

        model_badge = (
            f"{vision_model_used} -> {result.model}"
            if vision_model_used and vision_model_used != result.model
            else result.model
        )

        # Проверка (SymPy) запускается для расчётных/кодовых задач в фоне за <0.05 с без выгрузки модели из VRAM
        should_verify = (
            (resolved_task in self.verify_tasks and not is_followup)
            if verify is None
            else (bool(verify) and resolved_task in self.verify_tasks and not is_followup)
        )
        report = None
        if should_verify and answer_text:
            report = self.verifier.verify(
                problem=combined_query,
                solution=answer_text,
                solver_model=result.model,
                use_model=False,
            )

        dispace_val = extract_dispace_input(answer_text)

        return Answer(
            question=clean_question,
            text=answer_text,
            task=resolved_task if not images else f"vision+{resolved_task}",
            model=model_badge,
            seconds=round(time.perf_counter() - started, 2),
            sources=_dedupe(sources),
            verification=report,
            context_used=used,
            thinking=result.thinking,
            extracted_image_text=extracted_image_text,
            dispace_input=dispace_val,
        )

    # ------------------------------------------------------------ тренажёр

    def practice(
        self,
        topic: str,
        count: int = 5,
        difficulty: str = "средний",
        course: str | None = None,
        verify: bool = True,
        on_stream: Any = None,
        deep_think: bool = False,
    ) -> tuple[list[dict[str, Any]], list[VerificationReport]]:
        """Сгенерировать тренировочные задания в реальном времени и проверить эталонные решения."""
        context = self.store.context_for(topic, k=3, max_chars=3500, course=course)
        prompt = f"Тема: {topic}\nУровень: {difficulty}\nКоличество заданий: {count}\n"
        if context:
            prompt += f"\nОпирайся на материалы:\n{context}\n"

        result = self.llm.chat(
            [
                {"role": "system", "content": PRACTICE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            task="math" if deep_think else "practice",
            think=deep_think,
            json_format="json",
            on_stream=on_stream,
        )
        tasks = _parse_tasks(result.text)
        if not verify or not tasks:
            return tasks, []

        reports = self.verifier.verify_batch(
            [
                {
                    "problem": t["statement"],
                    "solution": t.get("solution") or f"Ответ: {t.get('answer', '')}",
                    "solver_model": result.model,
                    "answer": t.get("answer", ""),
                }
                for t in tasks
            ],
            use_model=False,
        )
        return tasks, reports

    def solve_course_assignment(
        self,
        question_text: str,
        *,
        options: Sequence[dict[str, str]] | None = None,
        theme_theory_text: str = "",
        course_name: str | None = None,
        student_attempt: str = "",
        on_stream: Any = None,
    ) -> Answer:
        """Решить задание из темы курса (или проверить решение студента) с полным тетрадным оформлением по лекциям."""
        started = time.perf_counter()
        sources: list[str] = []
        used = "нет"

        theory_context = (theme_theory_text or "").strip()
        if theory_context:
            used = "лекция текущей темы"
            sources.append("Теоретические блоки текущей темы курса")
        else:
            hits = self.store.search(question_text, k=4, course=course_name, min_score=1.5)
            if hits:
                blocks = []
                for ch, _sc in hits[:3]:
                    blocks.append(
                        f"📘 ДИСЦИПЛИНА: «{ch.course}» | 📑 ЛЕКЦИЯ: «{ch.title}» (фр. {ch.index + 1}, [#{ch.id}]):\n{ch.text}"
                    )
                    sources.append(f"«{ch.course}» — {ch.title} (фр. {ch.index + 1})")
                theory_context = "\n\n---\n\n".join(blocks)
                used = f"материалы курса «{hits[0][0].course}»"

        prompt_parts: list[str] = []
        if theory_context:
            prompt_parts.append(f"ТЕОРИЯ И ОБРАЗЦЫ РЕШЕНИЙ ИЗ ЛЕКЦИИ КУРСА:\n{theory_context[:6500]}")

        q_block = f"УСЛОВИЕ ЗАДАНИЯ:\n{question_text.strip()}"
        if options:
            opts_lines = [f"- Вариант {idx}: {opt.get('text', '')}" for idx, opt in enumerate(options, 1)]
            q_block += (
                "\n\nВАРИАНТЫ ОТВЕТА В ТЕСТЕ:\n" + "\n".join(opts_lines)
                + f"\n\n⚠️ ВНИМАНИЕ: Обязательно примени метод отсева! В решении подробно разбери КАЖДЫЙ вариант (1..{len(options)}), "
                f"обоснуй математическую ошибку неверных и строго докажи правильный вариант. В блоке «Итоговый ответ» напиши: «Ответ: Вариант N ($формула$)» и точный номер для выбора в DiSpace."
            )
        prompt_parts.append(q_block)

        if student_attempt.strip():
            prompt_parts.append(
                f"ОТВЕТ / РЕШЕНИЕ СТУДЕНТА НА ПРОВЕРКУ:\n{student_attempt.strip()}\n\n"
                "Проверь ответ студента, укажи верно ли он решён (и где ошибка, если она есть), "
                "а затем приведи полное эталонное тетрадное решение по материалам лекции."
            )
        else:
            prompt_parts.append(
                "Оформи полное пошаговое тетрадное решение этого задания по всем правилам, "
                "опираясь на теорию и образцы из лекции курса."
            )

        result = self.llm.chat(
            [
                {"role": "system", "content": NOTEBOOK_SOLUTION_SYSTEM},
                {"role": "user", "content": "\n\n".join(prompt_parts)},
            ],
            task="math",
            text_hint=question_text,
            on_stream=on_stream,
        )
        answer_text = format_math_markdown(_strip_prompt_echo(result.text, NOTEBOOK_SOLUTION_SYSTEM))
        report = self.verifier.verify(
            problem=question_text,
            solution=answer_text,
            solver_model=result.model,
            use_model=False,
        )

        # Автоматическая самокоррекция: если SymPy обнаружил ошибку в решении, делаем один корректирующий шаг
        if report and any(c.status == "fail" and c.expected for c in report.checks):
            failed_checks = [c for c in report.checks if c.status == "fail" and c.expected]
            feedback = "\n".join(f"- {c.detail} (ожидалось: {c.expected})" for c in failed_checks)
            correction_prompt = (
                f"ВНИМАНИЕ! Автоматическая математическая проверка (SymPy) нашла ошибку в твоём ответе:\n"
                f"{feedback}\n\n"
                f"Пожалуйста, перепроверь вычисления и исходную формулу, найди ошибку и выдай "
                f"исправленное тетрадное решение с верным ответом."
            )
            try:
                corr_res = self.llm.chat(
                    [
                        {"role": "system", "content": NOTEBOOK_SOLUTION_SYSTEM},
                        {"role": "user", "content": "\n\n".join(prompt_parts)},
                        {"role": "assistant", "content": answer_text},
                        {"role": "user", "content": correction_prompt},
                    ],
                    task="math",
                    text_hint=question_text,
                    on_stream=on_stream,
                )
                corr_text = format_math_markdown(_strip_prompt_echo(corr_res.text, NOTEBOOK_SOLUTION_SYSTEM))
                corr_report = self.verifier.verify(
                    problem=question_text,
                    solution=corr_text,
                    solver_model=corr_res.model,
                    use_model=False,
                )
                if corr_report and (corr_report.ok or not any(c.status == "fail" for c in corr_report.checks)):
                    answer_text = corr_text
                    report = corr_report
                    result = corr_res
            except Exception:
                pass
        dispace_val = extract_dispace_input(answer_text)
        return Answer(
            question=question_text,
            text=answer_text,
            task="notebook_math",
            model=result.model,
            seconds=round(time.perf_counter() - started, 2),
            sources=_dedupe(sources),
            verification=report,
            context_used=used,
            thinking=result.thinking,
            dispace_input=dispace_val,
        )

    # ------------------------------------------------------------- служебное

    def status(self) -> dict[str, Any]:
        health = self.llm.health()
        return {
            "ollama": health.as_text(),
            "materials": self.store.stats(),
            "search": self.search.health(),
            "routes": {
                task: self.llm.router.route(task).model for task in self.llm.router.routing
            },
        }


def _parse_tasks(text: str) -> list[dict[str, Any]]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        return []
    raw_json = cleaned[start : end + 1]

    tasks_raw = None
    # 1. Попытка с strict=False (допускает управляющие символы и переносы)
    try:
        data = json.loads(raw_json, strict=False)
        tasks_raw = data.get("tasks") if isinstance(data, dict) else data
    except Exception:
        pass

    # 2. Попытка экранировать одиночные LaTeX слэши (\begin, \lim, \frac), не трогая \"
    if not isinstance(tasks_raw, list):
        safe_json = re.sub(r'\\(?!["/])', r"\\\\", raw_json)
        try:
            data = json.loads(safe_json, strict=False)
            tasks_raw = data.get("tasks") if isinstance(data, dict) else data
        except Exception:
            pass

    # 3. Регекс-парсинг отдельных заданий, если общий JSON повреждён
    if not isinstance(tasks_raw, list):
        items = []
        task_pat = re.compile(
            r'\{\s*"statement"\s*:\s*"(.*?)"\s*,\s*"answer"\s*:\s*"(.*?)"\s*,\s*"solution"\s*:\s*"(.*?)"',
            re.DOTALL,
        )
        for m in task_pat.finditer(raw_json):
            items.append(
                {
                    "statement": m.group(1),
                    "answer": m.group(2),
                    "solution": m.group(3),
                    "difficulty": "средний",
                }
            )
        tasks_raw = items

    if not isinstance(tasks_raw, list):
        return []

    out = []
    for item in tasks_raw:
        if isinstance(item, dict) and item.get("statement"):
            out.append(
                {
                    "statement": format_math_markdown(str(item["statement"]).strip()),
                    "answer": format_math_markdown(str(item.get("answer", "")).strip()),
                    "solution": format_math_markdown(str(item.get("solution", "")).strip()),
                    "difficulty": str(item.get("difficulty", "средний")).strip(),
                }
            )
    return out


def _strip_prompt_echo(text: str, system_prompt: str) -> str:
    """Убрать дословно повторённые строки системной инструкции.

    Небольшие модели иногда дописывают в конец ответа правила из промпта.
    Удаляем только точные совпадения строк, поэтому содержательный текст
    не страдает.
    """
    if not text:
        return text

    system_lines = {
        line.strip()
        for line in system_prompt.splitlines()
        if len(line.strip()) >= 15
    }
    system_lines |= {"Правила:", "Требования:"}

    kept = [line for line in text.splitlines() if line.strip() not in system_lines]
    cleaned = "\n".join(kept).strip()
    # Если от ответа ничего не осталось, лучше показать исходный текст.
    return cleaned or text


def _dedupe(items: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tutor = Tutor()
    print(json.dumps(tutor.status(), ensure_ascii=False, indent=2)[:800], "\n")

    answer = tutor.ask(
        "Вычисли определитель матрицы [[3,1],[2,4]] и объясни ход.", verify=True
    )
    print(answer.as_text())
