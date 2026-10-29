"""Проверка решений: три независимых слоя.

Смысл в том, что нельзя спрашивать модель «ты уверена?» — она подтвердит.
Поэтому проверка строится из источников, не зависящих друг от друга:

  1. sympy        — детерминированный пересчёт. Ловит арифметику, знаки,
                    потерянные корни. Никогда не «мнение».
  2. кросс-модель — решение критикует ДРУГАЯ модель, не та, что решала.
                    Ловит ошибки в рассуждении, которые не видны формулой.
  3. ссылки       — для конспектов: каждое утверждение должно опираться
                    на фрагмент источника, иначе это галлюцинация.

Итоговый вердикт честно отражает, что именно удалось проверить:
    confirmed  — sympy подтвердил и/или независимая модель согласна
    disputed   — есть противоречие, ответу верить нельзя
    unverified — проверить нечем (нет разбора, модель недоступна)

Запуск для самопроверки:
    python src/verification.py
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import sympy as sp
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    SYMPY_OK = True
except ImportError:  # pragma: no cover
    SYMPY_OK = False
    sp = None  # type: ignore[assignment]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient, LLMError  # noqa: E402

TRANSFORMS = None
if SYMPY_OK:
    TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

VERDICT_CONFIRMED = "confirmed"
VERDICT_DISPUTED = "disputed"
VERDICT_UNVERIFIED = "unverified"


# --------------------------------------------------------------------- модель


@dataclass
class Check:
    """Результат одного слоя проверки."""

    layer: str
    status: str  # pass | fail | skipped | error
    detail: str = ""
    expected: str = ""

    @property
    def decisive(self) -> bool:
        return self.status in ("pass", "fail")

    def as_line(self) -> str:
        icons = {"pass": "ok  ", "fail": "FAIL", "skipped": "—   ", "error": "err "}
        line = f"[{icons.get(self.status, '?   ')}] {self.layer}: {self.detail}"
        if self.expected and self.status == "fail":
            line += f" (ожидалось: {self.expected})"
        return line


@dataclass
class VerificationReport:
    problem: str
    solution: str
    solver_model: str
    checks: list[Check] = field(default_factory=list)
    verdict: str = VERDICT_UNVERIFIED
    confidence: float = 0.0
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.verdict == VERDICT_CONFIRMED

    def as_text(self) -> str:
        head = {
            VERDICT_CONFIRMED: "ПОДТВЕРЖДЕНО",
            VERDICT_DISPUTED: "СПОРНО — не доверять",
            VERDICT_UNVERIFIED: "НЕ ПРОВЕРЕНО",
        }[self.verdict]
        lines = [f"Вердикт: {head} (уверенность {self.confidence:.0%})"]
        lines += [c.as_line() for c in self.checks]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "confidence": self.confidence,
            "solver_model": self.solver_model,
            "checks": [
                {"layer": c.layer, "status": c.status, "detail": c.detail, "expected": c.expected}
                for c in self.checks
            ],
        }


# ------------------------------------------------------- извлечение ответа


_ANSWER_LINE = re.compile(
    r"(?im)^[ \t>*]*(?:\*\*)?(?:ответ|answer|итог)(?:\*\*)?[ \t]*[:：][ \t]*(?:\*\*)?(.+)$"
)


def extract_answer(text: str) -> str:
    """Достать финальный ответ из решения (последнее «Ответ: ...»)."""
    matches = _ANSWER_LINE.findall(text or "")
    if matches:
        return matches[-1].strip().strip("*$` ").strip()
    # Резерв: последняя непустая строка, если она короткая и похожа на ответ.
    tail = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if tail and len(tail[-1]) <= 60:
        return tail[-1].strip("*$` ")
    return ""


_LATEX_SUBS = (
    (r"\\left|\\right", ""),
    (r"\\,|\\;|\\!|\\ ", " "),
    (r"\\boxed", ""),
    (r"\\text\s*\{([^{}]*)\}", r"\1"),
    (r"\\mathrm|\\mathbf|\\operatorname", ""),
    (r"\\cdot|\\times", "*"),
    (r"\\div", "/"),
    (r"\\pi", "pi"),
    (r"\\infty", "oo"),
    (r"\\sqrt\s*\{([^{}]*)\}", r"sqrt(\1)"),
    (r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"((\1)/(\2))"),
    (r"\\ln", "log"),
    (r"\\log", "log"),
    (r"\\sin", "sin"),
    (r"\\cos", "cos"),
    (r"\\tan", "tan"),
    (r"\\tg", "tan"),
    (r"\^\{([^{}]*)\}", r"**(\1)"),
    (r"\\%", "%"),
)


def latex_to_text(s: str) -> str:
    out = s
    for pattern, repl in _LATEX_SUBS:
        out = re.sub(pattern, repl, out)
    return out.replace("$", "").replace("{", "(").replace("}", ")").strip()


def _sympify(expr: str) -> Any:
    """Разобрать выражение, вернуть sympy-объект или None."""
    if not SYMPY_OK:
        return None
    cleaned = latex_to_text(expr)
    cleaned = cleaned.replace("×", "*").replace("÷", "/").replace("−", "-").replace(",", ".")
    # Убираем единицы измерения и слова.
    cleaned = re.sub(r"(?<=[0-9.\s)])[а-яА-Яa-zA-Z]{2,}", "", cleaned)
    cleaned = cleaned.strip(" .,;:")
    if not cleaned:
        return None
    try:
        return parse_expr(cleaned, transformations=TRANSFORMS, evaluate=True)
    except Exception:
        return None


def _numbers_match(expected: Any, got: Any) -> bool:
    """Сравнить ожидаемое и полученное с разумным допуском."""
    try:
        diff = sp.simplify(sp.nsimplify(expected) - sp.nsimplify(got))
        if diff == 0:
            return True
        if not diff.free_symbols:
            val = complex(diff.evalf())
            return abs(val) <= 1e-9 * max(1.0, abs(complex(sp.N(expected))))
    except Exception:
        pass
    try:
        return abs(float(sp.N(expected)) - float(sp.N(got))) <= 1e-6 * max(1.0, abs(float(sp.N(expected))))
    except Exception:
        return False


def _parse_answer_numbers(answer: str) -> list[Any]:
    """Кандидаты-ответы из строки: только конкретные значения, без символов.

    Отбрасываем всё, что содержит неизвестные: иначе из фразы «Смотри решение»
    парсер делает произведение букв и проверка ломается.
    """
    text = latex_to_text(answer)
    text = re.sub(r"(?i)^\s*(ответ|answer)\s*[:：]\s*", "", text)
    text = text.replace("\\", " ")
    text = re.sub(r"\s+", " ", text).strip(" .;:")
    text = re.sub(r"\+\s*C\b", "", text)  # константа интегрирования

    values: list[Any] = []

    whole = _sympify(text)
    if whole is not None and not getattr(whole, "free_symbols", set()):
        values.append(whole)

    for chunk in re.split(r"[;,\n]|\s+и\s+", text):
        piece = chunk.split("=")[-1].strip(" ()[]{}")
        if not piece:
            continue
        value = _sympify(piece)
        if value is not None and not getattr(value, "free_symbols", set()):
            values.append(value)

    if not values:
        for raw in re.findall(r"-?\d+(?:[.,]\d+)?", text):
            values.append(_sympify(raw.replace(",", ".")))

    return [v for v in values if v is not None]


def _answer_expressions(answer: str) -> list[Any]:
    """Кандидаты-ответы с сохранением символов: для производных и интегралов."""
    text = latex_to_text(answer)
    text = re.sub(r"(?i)^\s*(ответ|answer|итог)\s*[:：]\s*", "", text).strip()
    text = re.sub(r"\+\s*C\b", "", text)  # константа интегрирования
    text = text.strip(" .;:")

    values: list[Any] = []

    whole = _sympify(text)
    if whole is not None:
        values.append(whole)

    for chunk in re.split(r"[;,\n]|\s+и\s+", text):
        piece = chunk.split("=")[-1].strip(" ()[]{}")
        if not piece:
            continue
        value = _sympify(piece)
        if value is not None:
            values.append(value)

    return values


# ------------------------------------------------------------ слой sympy


_MATRIX_RE = re.compile(r"\[\s*\[([^\[\]]+)\](?:\s*,\s*\[([^\[\]]+)\])+\s*\]")
_LIMIT_RE = re.compile(
    r"lim\s*_?\{?\s*([A-Za-z])\s*(?:->|→|\\to|\\rightarrow)\s*([^{}\s,]+)\s*\}?\s*(.+)",
    re.IGNORECASE,
)
_EQUATION_RE = re.compile(r"([0-9A-Za-z\s^*/+\-().]+?)\s*=\s*([0-9A-Za-z\s^*/+\-().]+)")


def _extract_matrix(text: str) -> Any:
    r"""Найти матрицу вида [[1,2],[3,4]] или \begin{pmatrix}...\end{pmatrix}."""
    match = _MATRIX_RE.search(text)
    if match:
        inner = re.findall(r"\[([^\[\]]+)\]", match.group(0))
        rows = []
        for row in inner:
            values = [_sympify(v) for v in row.split(",")]
            if any(v is None for v in values):
                return None
            rows.append(values)
        try:
            return sp.Matrix(rows)
        except Exception:
            return None

    env = re.search(r"\\begin\{(p|b|v)?matrix\}(.+?)\\end\{(p|b|v)?matrix\}", text, re.DOTALL)
    if env:
        body = env.group(2).replace("\\\\", ";").replace("&", ",")
        rows = []
        for row in body.split(";"):
            if not row.strip():
                continue
            values = [_sympify(v) for v in row.split(",")]
            if any(v is None for v in values):
                return None
            rows.append(values)
        try:
            return sp.Matrix(rows)
        except Exception:
            return None
    return None


def _check_matrix_det(problem: str, answer: str) -> Check | None:
    lowered = problem.lower()
    if "det" not in lowered and "определит" not in lowered:
        return None
    matrix = _extract_matrix(problem)
    if matrix is None:
        return None
    if matrix.rows != matrix.cols:
        return Check("sympy", "error", "матрица не квадратная")
    expected = matrix.det()
    got_values = _parse_answer_numbers(answer)
    if not got_values:
        return Check("sympy", "skipped", "ответ не удалось разобрать как число", str(expected))
    if any(_numbers_match(expected, v) for v in got_values):
        return Check("sympy", "pass", f"det пересчитан независимо: {expected}")
    return Check(
        "sympy", "fail", f"det = {expected}, а в ответе {got_values[0]}", str(expected)
    )


def _check_limit(problem: str, answer: str) -> Check | None:
    match = _LIMIT_RE.search(latex_to_text(problem))
    if not match:
        return None
    var_name, point, expr_src = match.group(1), match.group(2), match.group(3)
    expr_src = expr_src.split("=")[0]
    expr = _sympify(expr_src)
    point_value = _sympify(point.replace("oo", "oo"))
    if expr is None or point_value is None:
        return None
    try:
        expected = sp.limit(expr, sp.Symbol(var_name), point_value)
    except Exception as exc:
        return Check("sympy", "error", f"не удалось вычислить предел: {exc}")
    got_values = _parse_answer_numbers(answer)
    if not got_values:
        return Check("sympy", "skipped", "ответ не разобран", str(expected))
    if any(_numbers_match(expected, v) for v in got_values):
        return Check("sympy", "pass", f"предел пересчитан: {expected}")
    return Check("sympy", "fail", f"предел = {expected}, в ответе {got_values[0]}", str(expected))


# Символы, которые могут встречаться внутри математического выражения.
# Кириллица сюда не входит — поэтому разбор автоматически отсекает слова
# вроде «реши систему» и останавливается на запятой между уравнениями.
_MATH_CHARS = set(
    "0123456789"
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    " \t+-*/^()._"
)


def _parse_equations(text: str) -> list[tuple[Any, Any]]:
    r"""Вытащить все уравнения из условия, включая системы.

    Работает от знака «=»: расширяется влево и вправо, пока символы похожи
    на математику. Поэтому «реши систему 2x + y = 5, x + 3y = 10» даёт два
    уравнения, а не одно склеенное.
    """
    source = latex_to_text(text)
    equations: list[tuple[Any, Any]] = []
    seen: set[tuple[str, str]] = set()

    for position, char in enumerate(source):
        if char != "=":
            continue
        previous = source[position - 1] if position else ""
        following = source[position + 1] if position + 1 < len(source) else ""
        if previous in "<>!=" or following == "=":  # <=, >=, !=, ==
            continue

        start = position
        while start > 0 and source[start - 1] in _MATH_CHARS:
            start -= 1
        end = position + 1
        while end < len(source) and source[end] in _MATH_CHARS:
            end += 1

        left_src = source[start:position].strip()
        right_src = source[position + 1 : end].strip()
        if not left_src or not right_src:
            continue
        if (left_src, right_src) in seen:
            continue

        left, right = _sympify(left_src), _sympify(right_src)
        if left is None or right is None:
            continue
        seen.add((left_src, right_src))
        equations.append((left, right))

    return equations


def _match_multiset(expected: Sequence[Any], got: Sequence[Any]) -> bool:
    """Каждое ожидаемое значение должно найтись среди ответов (без повторов)."""
    remaining = list(got)
    for value in expected:
        for index, candidate in enumerate(remaining):
            if _numbers_match(value, candidate):
                remaining.pop(index)
                break
        else:
            return False
    return True


def _check_equation(problem: str, answer: str) -> Check | None:
    equations = _parse_equations(problem)
    if not equations:
        return None

    lowered = problem.lower()
    triggers = (
        "реши",
        "решить",
        "уравнен",
        "систем",
        "найдите корни",
        "найди корни",
        "найти корни",
    )
    # Два и более уравнения — это уже система, слова-команды не обязательны.
    if len(equations) < 2 and not any(word in lowered for word in triggers):
        return None

    symbols: set[Any] = set()
    for left, right in equations:
        symbols |= (left - right).free_symbols
    ordered = sorted(symbols, key=lambda s: s.name)
    if not ordered:
        return None

    try:
        if len(equations) == 1:
            if len(ordered) != 1:
                return None
            expected = sp.solve(sp.Eq(*equations[0]), ordered[0])
            # Все корни одного уравнения — один ответ. Если ответ содержит
            # только часть корней, это потерянный корень, а не совпадение.
            expected_sets = [list(expected)]
            label = "корни"
        else:
            solutions = sp.solve([sp.Eq(l, r) for l, r in equations], ordered, dict=True)
            if not solutions:
                return Check(
                    "sympy",
                    "fail",
                    "система несовместна либо решения не найдены",
                )
            expected_sets = [[solution.get(symbol) for symbol in ordered] for solution in solutions]
            label = "решение системы"
    except Exception as exc:
        return Check("sympy", "error", f"не удалось решить: {exc}")

    flat = [value for solution in expected_sets for value in solution]
    if any(value is None or getattr(value, "free_symbols", set()) for value in flat):
        return Check(
            "sympy",
            "skipped",
            "решение параметрическое или неединственное — нужна ручная сверка",
        )

    got_values = _parse_answer_numbers(answer)
    if not got_values:
        return Check("sympy", "skipped", "ответ не разобран", str(expected_sets))

    for expected_values in expected_sets:
        if any(value is None for value in expected_values):
            continue
        if _match_multiset(expected_values, got_values):
            rendered = ", ".join(
                f"{symbol.name} = {value}" for symbol, value in zip(ordered, expected_values)
            )
            return Check("sympy", "pass", f"{label} пересчитано: {rendered}")

    first = expected_sets[0] if expected_sets else []
    rendered = ", ".join(
        f"{symbol.name} = {value}" for symbol, value in zip(ordered, first)
    )
    return Check(
        "sympy", "fail", f"{label}: {rendered}, в ответе {got_values}", rendered
    )


def _check_derivative(problem: str, answer: str) -> Check | None:
    lowered = problem.lower()
    if "производн" not in lowered and "d/dx" not in lowered:
        return None
    match = re.search(r"(?:производн\w*|d/dx)\s*(?:функции|от)?\s*[:：]?\s*(.+)", problem, re.IGNORECASE)
    if not match:
        return None
    expr_src = match.group(1).split("=")[0]
    expr = _sympify(expr_src)
    if expr is None:
        return None
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if not symbols:
        return None
    try:
        expected = sp.diff(expr, symbols[0])
    except Exception as exc:
        return Check("sympy", "error", f"не удалось продифференцировать: {exc}")
    got_values = _answer_expressions(answer) or [None]
    for got in got_values:
        if got is None:
            continue
        try:
            if sp.simplify(expected - got) == 0:
                return Check("sympy", "pass", f"производная пересчитана: {expected}")
        except Exception:
            pass
        if _numbers_match(expected, got):
            return Check("sympy", "pass", f"производная пересчитана: {expected}")
    return Check("sympy", "fail", f"производная = {expected}, в ответе {got_values[0]}", str(expected))


def _check_integral(problem: str, answer: str) -> Check | None:
    lowered = problem.lower()
    if "интеграл" not in lowered and "∫" not in problem:
        return None
    match = re.search(r"(?:∫|интеграл\w*\s*(?:от)?)\s*(.+)", problem, re.IGNORECASE)
    if not match:
        return None
    # Знак интеграла не должен попасть в подынтегральное выражение,
    # а «dx» — это дифференциал, а не множитель.
    expr_src = match.group(1).replace("∫", " ")
    expr_src = re.split(r"\bd\s*[a-zA-Z]\b", expr_src)[0]
    expr = _sympify(expr_src)
    if expr is None:
        return None
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)
    if not symbols:
        return None
    try:
        expected = sp.integrate(expr, symbols[0])
    except Exception as exc:
        return Check("sympy", "error", f"не удалось проинтегрировать: {exc}")
    got_values = _answer_expressions(answer) or [None]
    for got in got_values:
        if got is None:
            continue
        try:
            if sp.simplify(expected - got) == 0:
                return Check("sympy", "pass", f"интеграл проверен дифференцированием: {expected}")
        except Exception:
            continue
        if _numbers_match(expected, got):
            return Check("sympy", "pass", f"интеграл совпал: {expected}")
    return Check("sympy", "fail", f"интеграл = {expected}, в ответе {got_values[0]}", str(expected))


def _check_arithmetic(problem: str, answer: str) -> Check | None:
    """Простое выражение без слов: 125*8-25."""
    stripped = latex_to_text(problem).strip()
    stripped = re.sub(r"(?i)^(вычисли\w*|найди\w*|реши\w*)\s*[:：]?", "", stripped).strip()
    if re.search(r"[а-яА-Я]{3,}", stripped):
        return None
    if not re.fullmatch(r"[0-9\s.,+\-*/^()]+", stripped):
        return None
    expr = _sympify(stripped)
    if expr is None:
        return None
    got_values = _parse_answer_numbers(answer)
    if not got_values:
        return Check("sympy", "skipped", "ответ не разобран", str(expr))
    if any(_numbers_match(expr, v) for v in got_values):
        return Check("sympy", "pass", f"выражение вычислено: {expr}")
    return Check("sympy", "fail", f"значение {expr}, в ответе {got_values[0]}", str(expr))


SYMPY_CHECKERS = (
    _check_matrix_det,
    _check_limit,
    _check_derivative,
    _check_integral,
    _check_equation,
    _check_arithmetic,
)


def sympy_layer(problem: str, answer: str) -> Check:
    """Пересчитать задачу независимо от модели. Пропуск, если тип не распознан."""
    if not SYMPY_OK:
        return Check("sympy", "skipped", "sympy не установлен")
    for checker in SYMPY_CHECKERS:
        try:
            result = checker(problem, answer)
        except Exception as exc:  # проверка не должна ронять пайплайн
            return Check("sympy", "error", f"{checker.__name__}: {exc}")
        if result is not None:
            return result
    return Check("sympy", "skipped", "тип задачи не распознан для пересчёта")


# ------------------------------------------------------- слой кросс-модели


JUDGE_SYSTEM = """Ты — объективный математический экзаменатор. Тебе дают условие и чужое решение.
Твоя задача — проверить правильность вычислений и итогового ответа.

Правила:
1. Пересчитай ключевые арифметические и алгебраические переходы и итоговый ответ.
2. НЕ считай ошибкой выбор обозначений (например, \\Delta_{x_1}/\\Delta_{x_2} вместо \\Delta_x/\\Delta_y), оформление или подробность записи, если вычисления и итоговый ответ верны!
3. В поле issues пиши обычный читаемый текст без слэшей LaTeX.

Отвечай ТОЛЬКО JSON без пояснений вокруг:
{"verdict": "correct" | "incorrect" | "uncertain",
 "issues": ["конкретная арифметическая ошибка", ...],
 "correct_answer": "верный ответ или null"}

verdict="correct", если итоговый ответ и вычисления верны.
verdict="incorrect", только если найдена реальная математическая ошибка, меняющая ответ.
verdict="uncertain", если условие неполное."""


def _clean_json_latex_str(val: Any) -> str:
    s = str(val or "")
    s = s.replace("\x0crac", "\\frac").replace("\x0c", "\\f")
    s = s.replace("\x08egin", "\\begin").replace("\x08eta", "\\beta").replace("\x08", "\\b")
    s = re.sub(r"\t(ext\{|imes\b|heta\b|au\b|an\b|riangle\b|o\b)", r"\\t\1", s)
    return s


def _parse_json_block(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("в ответе нет JSON")
    raw_json = cleaned[start : end + 1]
    safe_json = re.sub(r'\\(?![/\"\\]|u[0-9a-fA-F]{4})', r"\\\\", raw_json)
    try:
        return json.loads(safe_json)
    except json.JSONDecodeError:
        return json.loads(raw_json)


def model_layer(
    problem: str,
    solution: str,
    solver_model: str,
    llm: LLMClient,
    verifier_model: str | None = None,
) -> tuple[Check, str]:
    """Независимая критика решения другой моделью."""
    verifier = verifier_model or llm.router.verifier(solver_model)
    if not verifier:
        return Check("кросс-модель", "skipped", "нет второй модели для независимой проверки"), ""
    if verifier == solver_model:
        return Check("кросс-модель", "skipped", "проверяющая модель совпала с решавшей — независимости нет"), ""

    prompt = (
        f"УСЛОВИЕ:\n{problem}\n\n"
        f"ЧУЖОЕ РЕШЕНИЕ:\n{solution[:6000]}\n\n"
        "Проверь и ответь JSON."
    )
    try:
        result = llm.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            task="verify",
            model=verifier,
            json_format="json",
        )
        data = _parse_json_block(result.text)
    except (LLMError, ValueError, json.JSONDecodeError) as exc:
        return Check("кросс-модель", "error", f"{verifier}: {exc}"), ""

    verdict = str(data.get("verdict", "uncertain")).lower()
    issues = data.get("issues") or []
    correct_answer = _clean_json_latex_str(data.get("correct_answer") or "")
    issue_text = (
        "; ".join(_clean_json_latex_str(i) for i in issues[:4])
        if isinstance(issues, list)
        else _clean_json_latex_str(issues)
    )

    if verdict == "correct":
        return Check("кросс-модель", "pass", f"{verifier} не нашёл ошибок"), correct_answer
    if verdict == "incorrect":
        return (
            Check(
                "кросс-модель",
                "fail",
                f"{verifier} нашёл ошибку: {issue_text or 'без пояснений'}",
                correct_answer,
            ),
            correct_answer,
        )
    return Check("кросс-модель", "skipped", f"{verifier} не уверен: {issue_text or 'условие неясно'}"), correct_answer


# --------------------------------------------------------------- верификатор


class Verifier:
    """Собирает слои в один вердикт."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm or LLMClient()

    def verify(
        self,
        problem: str,
        solution: str,
        solver_model: str,
        answer: str | None = None,
        use_model: bool = False,
    ) -> VerificationReport:
        started = time.perf_counter()
        checks: list[Check] = []

        final_answer = answer if answer is not None else extract_answer(solution)
        symbolic = sympy_layer(problem, final_answer or solution)
        checks.append(symbolic)

        # Если SymPy уже математически доказал правильность ответа (pass), не тратим время и VRAM на вторую LLM
        if use_model and symbolic.status not in ("pass", "fail"):
            model_check, _ = model_layer(problem, solution, solver_model, self.llm)
            checks.append(model_check)

        verdict, confidence = _decide(checks)
        return VerificationReport(
            problem=problem,
            solution=solution,
            solver_model=solver_model,
            checks=checks,
            verdict=verdict,
            confidence=confidence,
            seconds=round(time.perf_counter() - started, 2),
        )

    def verify_batch(
        self,
        items: Sequence[dict[str, Any]],
        use_model: bool = True,
    ) -> list[VerificationReport]:
        """Проверить пачку решений.

        Все проверки идут одной моделью, поэтому веса загружаются один раз —
        на 8 ГБ VRAM это экономит десятки секунд на каждой задаче.
        """
        verifier_model = None
        if use_model and items:
            verifier_model = self.llm.router.verifier(items[0]["solver_model"])
        reports = []
        for item in items:
            started = time.perf_counter()
            symbolic = sympy_layer(item["problem"], item.get("answer") or item["solution"])
            checks = [symbolic]
            if use_model and symbolic.status not in ("pass", "fail"):
                model_check, _ = model_layer(
                    item["problem"],
                    item["solution"],
                    item["solver_model"],
                    self.llm,
                    verifier_model=verifier_model,
                )
                checks.append(model_check)
            verdict, confidence = _decide(checks)
            reports.append(
                VerificationReport(
                    problem=item["problem"],
                    solution=item["solution"],
                    solver_model=item["solver_model"],
                    checks=checks,
                    verdict=verdict,
                    confidence=confidence,
                    seconds=round(time.perf_counter() - started, 2),
                )
            )
        return reports


def _decide(checks: Sequence[Check]) -> tuple[str, float]:
    symbolic = next((c for c in checks if c.layer == "sympy"), None)
    model = next((c for c in checks if c.layer == "кросс-модель"), None)

    # Детерминированный символьный пересчёт SymPy имеет абсолютный приоритет над мнением языковой модели.
    if symbolic and symbolic.status == "fail":
        return VERDICT_DISPUTED, 0.95
    if symbolic and symbolic.status == "pass":
        return VERDICT_CONFIRMED, 0.95

    if model and model.status == "fail":
        return VERDICT_DISPUTED, 0.8
    if model and model.status == "pass":
        return VERDICT_CONFIRMED, 0.65

    return VERDICT_UNVERIFIED, 0.25


# -------------------------------------------------------------------- CLI


_DEMO = [
    (
        "Вычисли определитель матрицы [[2,1],[3,4]]",
        "det = 2*4 - 1*3 = 5\n\nОтвет: 5",
        "верный",
    ),
    (
        "Вычисли определитель матрицы [[5,2],[1,3]]",
        "det = 5*3 - 2*1 = 13\n\nОтвет: 13",
        "верный",
    ),
    (
        "Вычисли определитель матрицы [[5,2],[1,3]]",
        "det = 5*3 - 2*1 = 17\n\nОтвет: 17",
        "специально неверный",
    ),
    (
        "Найди предел lim x->0 sin(x)/x",
        "Это первый замечательный предел.\n\nОтвет: 1",
        "верный",
    ),
    (
        "Реши уравнение 3x + 7 = 22",
        "3x = 15, x = 5\n\nОтвет: x = 5",
        "верный",
    ),
]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    verifier = Verifier()
    print(f"sympy: {'доступен' if SYMPY_OK else 'НЕТ'}\n")

    for problem, solution, label in _DEMO:
        report = verifier.verify(problem, solution, solver_model="deepseek-r1:8b")
        print(f"--- {problem}  [{label}]")
        print(f"ответ: {extract_answer(solution)!r}")
        print(report.as_text())
        print(f"({report.seconds} с)\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
