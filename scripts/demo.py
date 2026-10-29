"""Сквозная демонстрация: материалы -> ответ со ссылками -> проверка.

    python scripts/demo.py
    python scripts/demo.py --web      # добавить веб-поиск в один из вопросов
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tutor import Tutor  # noqa: E402


def line(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--web", action="store_true", help="проверить и веб-поиск")
    args = parser.parse_args(argv)

    tutor = Tutor()
    status = tutor.status()
    line("СОСТОЯНИЕ")
    print(status["ollama"])
    print(f"материалы: {status['materials']['sources']} источников, "
          f"{status['materials']['chunks']} фрагментов")
    print(f"поиск: {status['search']}")

    # 1. Вопрос, ответ на который есть в загруженной лекции.
    line("1. ВОПРОС ПО МАТЕРИАЛАМ (конспект + ссылки)")
    answer = tutor.ask("Что такое алгебраическое дополнение и как через него считается определитель?")
    print(answer.text[:1200])
    print(f"\nконтекст: {answer.context_used} · модель: {answer.model} · {answer.seconds:.0f} с")
    if answer.sources:
        print("источники: " + "; ".join(answer.sources))

    # 2. Задача с автоматической проверкой.
    line("2. ЗАДАЧА С ПРОВЕРКОЙ (sympy + кросс-модель)")
    answer = tutor.ask("Реши систему 2x + y = 5, x + 3y = 10 методом Крамера.", verify=True)
    print(answer.text[:1200])
    if answer.verification:
        print("\n" + answer.verification.as_text())

    # 3. Ловушка: неверный ответ должен быть отловлен.
    line("3. КОНТРОЛЬНАЯ ПРОВЕРКА ЛОВУШКИ")
    from verification import Verifier

    report = Verifier(tutor.llm).verify(
        problem="Вычисли определитель матрицы [[4,7],[2,6]]",
        solution="det = 4*6 - 7*2 = 24 - 15 = 9\n\nОтвет: 9",
        solver_model="deepseek-r1:8b",
    )
    print(report.as_text())

    # 4. Веб-поиск, если попросили.
    if args.web:
        line("4. ВЕБ-ПОИСК")
        answer = tutor.ask(
            "Какие свойства определителя при перестановке строк?", use_web=True, verify=False
        )
        print(answer.text[:900])
        print(f"\nконтекст: {answer.context_used} · {answer.seconds:.0f} с")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
