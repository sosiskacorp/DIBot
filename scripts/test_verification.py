"""Регрессионные тесты слоя sympy. Без модели, поэтому идут секунды.

    python scripts/test_verification.py

Каждый кейс проверяет не «что-то ответило», а конкретный ожидаемый статус:
pass — пересчёт совпал, fail — пересчёт поймал ошибку, skipped — тип не распознан.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from verification import Check, extract_answer, sympy_layer  # noqa: E402

CASES: list[tuple[str, str, str, str]] = [
    # (условие, решение, ожидаемый статус, что проверяем)
    (
        "Вычисли определитель матрицы [[5,2],[1,3]]",
        "det = 5*3 - 2*1 = 13\n\nОтвет: 13",
        "pass",
        "определитель 2x2",
    ),
    (
        "Вычисли определитель матрицы [[5,2],[1,3]]",
        "det = 5*3 - 2*1 = 17\n\nОтвет: 17",
        "fail",
        "неверный определитель",
    ),
    (
        "Найди предел lim x->0 sin(x)/x",
        "Первый замечательный предел.\n\nОтвет: 1",
        "pass",
        "предел",
    ),
    (
        "Реши уравнение 3x + 7 = 22",
        "3x = 15\n\nОтвет: x = 5",
        "pass",
        "линейное уравнение",
    ),
    (
        "Реши уравнение x^2 = 4",
        "x = 2\n\nОтвет: x = 2",
        "fail",
        "потерянный корень",
    ),
    (
        "Реши систему 2x + y = 5, x + 3y = 10 методом Крамера.",
        "det = 5, x = 1, y = 3\n\nОтвет: (x, y) = (1, 3)",
        "pass",
        "система, верный ответ",
    ),
    (
        "Реши систему 2x + y = 5, x + 3y = 10 методом Крамера.",
        "Ответ: (x, y) = (2, 1)",
        "fail",
        "система, неверный ответ",
    ),
    (
        "Найди производную функции x^3 + 2x",
        "d/dx(x^3 + 2x) = 3x^2 + 2\n\nОтвет: 3x^2 + 2",
        "pass",
        "производная, символьный ответ",
    ),
    (
        "Найди производную функции x^3 + 2x",
        "Ответ: 3x^2",
        "fail",
        "производная, потерянное слагаемое",
    ),
    (
        "Вычисли интеграл ∫ x^2 dx",
        "∫x^2 dx = x^3/3 + C\n\nОтвет: x^3/3 + C",
        "pass",
        "интеграл с константой",
    ),
    (
        "Вычисли интеграл ∫ x^2 dx",
        "Ответ: x^3/3 + x",
        "fail",
        "неверный интеграл",
    ),
    (
        "Реши систему x + y = 2, 2x + 2y = 4",
        "Ответ: x = 1, y = 1",
        "skipped",
        "система с бесконечным множеством решений",
    ),
    (
        "Вычисли 125*8 - 25",
        "125*8 = 1000, 1000 - 25 = 975\n\nОтвет: 975",
        "pass",
        "арифметика",
    ),
    (
        "Вычисли 125*8 - 25",
        "Ответ: 1000",
        "fail",
        "неверная арифметика",
    ),
    (
        "Что такое алгебраическое дополнение?",
        "Это минор со знаком.\n\nОтвет: A_ij = (-1)^(i+j) M_ij",
        "skipped",
        "теория не пересчитывается",
    ),
]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    failures = 0
    for problem, solution, expected_status, label in CASES:
        answer = extract_answer(solution)
        check: Check = sympy_layer(problem, answer)
        ok = check.status == expected_status
        failures += 0 if ok else 1
        mark = "ok  " if ok else "FAIL"
        print(f"[{mark}] {label}: ждали {expected_status}, получили {check.status}")
        print(f"       ответ {answer!r} → {check.detail}")

    print()
    if failures:
        print(f"провалено кейсов: {failures} из {len(CASES)}")
        return 1
    print(f"все кейсы прошли: {len(CASES)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
