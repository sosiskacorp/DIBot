"""Диагностика слоя моделей.

    python src/check_llm.py                       # связь, модели, маршруты
    python src/check_llm.py --smoke               # реальный запрос на каждую задачу
    python src/check_llm.py --classify "реши предел"   # что роутер думает о тексте
    python src/check_llm.py --task math --prompt "lim x->0 sin(x)/x"
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient, LLMError  # noqa: E402
from model_router import TASKS, classify  # noqa: E402


def _print_routing(llm: LLMClient) -> None:
    print("\nМаршруты:")
    print(f"  {'задача':9} {'модель':18} {'think':6} {'temp':5} {'ctx':6} запасные")
    for task in TASKS:
        route = llm.router.route(task)
        print(
            f"  {task:9} {route.model:18} {str(route.think):6} "
            f"{route.temperature:<5} {route.num_ctx:<6} {', '.join(route.fallback) or '—'}"
        )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Проверка Ollama и роутинга моделей")
    parser.add_argument("--smoke", action="store_true", help="прогнать короткий запрос по каждой задаче")
    parser.add_argument("--classify", metavar="ТЕКСТ", help="показать, какую задачу определит роутер")
    parser.add_argument("--task", help="явно указать задачу для --prompt")
    parser.add_argument("--prompt", help="запрос к модели")
    parser.add_argument("--timeout", type=float, default=None, help="таймаут одного запроса, сек")
    args = parser.parse_args(argv)

    if args.classify:
        declared = args.task or classify(args.classify)
        route = LLMClient().router.route(declared)
        print(f"Текст: {args.classify[:80]}")
        print(f"Задача: {declared}")
        print(f"Модель: {route.model} (think={route.think}, temp={route.temperature}, ctx={route.num_ctx})")
        return 0

    llm = LLMClient()

    if args.prompt:
        started = time.perf_counter()
        try:
            res = llm.chat(
                [{"role": "user", "content": args.prompt}],
                task=args.task,
                text_hint=args.prompt,
                timeout=args.timeout,
            )
        except LLMError as exc:
            print(f"ОШИБКА: {exc}")
            return 2
        print(f"модель: {res.model}  задача: {res.task}")
        print(f"токенов: {res.output_tokens} за {res.seconds:.1f} с "
              f"({res.tokens_per_second:.1f} ток/с)")
        if res.thinking:
            print(f"\n--- размышления ({len(res.thinking)} симв.) ---")
            print(res.thinking[:800])
        print(f"\n--- ответ ---\n{res.text}")
        print(f"\nвсего: {time.perf_counter() - started:.1f} с")
        return 0

    report = llm.health()
    print(report.as_text())
    _print_routing(llm)

    print("\nПроверка независимости верификатора:")
    for solver in llm.router.models:
        verifier = llm.router.verifier(solver)
        mark = "ok" if verifier and verifier != solver else "ПРОБЛЕМА"
        print(f"  решал {solver:18} -> проверяет {verifier or '—':18} [{mark}]")

    if args.smoke:
        print("\nБоевой прогон (может занять несколько минут):")
        for task, model, ok, detail in llm.smoke_test(timeout=args.timeout):
            mark = "ok  " if ok else "FAIL"
            print(f"  [{mark}] {task:9} {model:18} {detail}")

    llm.close()
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
