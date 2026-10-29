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

Правила ответа:
1. Если дан КОНТЕКСТ ИЗ МАТЕРИАЛОВ и он относится к сути вопроса — опирайся на него и ставь ссылку `[#id]` после каждого факта из материалов. Если приведённый контекст не относится к вопросу (например, вопрос общий или о твоих возможностях) — просто игнорируй контекст и не упоминай посторонние курсы.
2. Все математические формулы и символы оформляй ТОЛЬКО через `$...$` (строчные, без пробелов сразу после открывающего и перед закрывающим `$`) и `$$...$$` с новой строки (крупные формулы, матрицы, системы, пределы, интегралы). Не используй `\\(...\\)` и `\\[...\\]`.
3. Любые вычисления показывай пошагово, а не только результат.
4. Для расчётных задач оформляй решение как в тетради: Дано, Ход решения, и в конце отдельной строкой «Ответ: ...» — по ней работает автоматическая проверка SymPy.
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
   Чёткая математическая запись исходного условия задачи, функции, последовательности или вариантов ответа.
2. **📖 Теоретическая база (по материалам лекции):**
   Укажи точное определение, теорему, табличную формулу или правило дифференцирования/пределов/матриц из приведённого конспекта лекции, которое применяется в этой задаче (в тех же обозначениях, что и в лекции).
3. **✍️ Подробный тетрадный ход решения:**
   Расписывай каждый шаг последовательно, без пропуска промежуточных выкладок:
   - что подставляем,
   - какое тождество или правило применяем на каждом переходе,
   - промежуточные упрощения.
   Все строчные формулы оформляй в `$...$`, а ключевые выкладки, пределы, дроби и матрицы — на отдельных строках в `$$...$$`.
4. **✅ Итоговый ответ (и формат для поля ввода DiSpace):**
   - Классическая математическая запись: `Ответ: ...`
   - Строка для ввода в поле DiSpace (если текстовый ответ: например `18*x^(3)-4*x^(-3)` или `-1/2`, либо точный выбор верных пунктов из предложенных вариантов).

Если студент прислал свой вариант ответа или решения на проверку — сначала внимательно проверь его: похвали за верные шаги, укажи точную строчку с ошибкой (если она есть) и затем приведи полное эталонное тетрадное решение."""

VISION_EXTRACT_SYSTEM = """Ты — модуль оптического распознавания учебных материалов и задач.
Твоя задача — максимально точно переписать весь текст, условия задач, таблицы и математические
формулы с прикреплённого изображения.
Правила:
1. Все математические формулы, матрицы, пределы, интегралы и уравнения записывай в LaTeX ($...$ или $$...$$).
2. Сохраняй точные числа, знаки и индексы. Не пропускай условия.
3. Если на фото изображена схема, график или чертёж — подробно опиши его структуру и данные.
4. Выводи только распознанное условие и данные без лишних вступлений."""


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

    @property
    def verdict(self) -> str:
        return self.verification.verdict if self.verification else "не проверялось"

    def as_text(self) -> str:
        lines = []
        if self.extracted_image_text:
            lines.extend(["**Распознано с фото:**", self.extracted_image_text, ""])
        lines.extend([self.text, "", "---"])
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
        clean_question = (question or "").strip() or "Разбери и реши задание на изображении."
        extracted_image_text = ""
        vision_model_used = ""

        # Шаг 1: если прикреплены фото, распознаём текст и формулы через vision-модель (qwen3-vl)
        if images:
            if on_stream is not None:
                try:
                    on_stream("status", "👁️ Распознаю формулы и текст с изображения (qwen3-vl)…", "qwen3-vl:8b")
                except Exception:
                    pass
            vision_prompt = (
                "Перепиши условие задачи, текст и все формулы (в LaTeX) с изображения."
                if not question.strip()
                else f"Запрос пользователя: {clean_question}\n\nПерепиши весь текст, формулы (в LaTeX) и данные с изображения, необходимые для ответа."
            )
            vision_res = self.llm.chat(
                [
                    {"role": "system", "content": VISION_EXTRACT_SYSTEM},
                    {"role": "user", "content": vision_prompt},
                ],
                task="vision",
                images=images,
            )
            extracted_image_text = vision_res.text.strip()
            vision_model_used = vision_res.model

        # Совмещаем вопрос пользователя и распознанный текст с фото для поиска и классификации
        combined_query = (
            f"{clean_question}\n{extracted_image_text}".strip()
            if extracted_image_text
            else clean_question
        )

        if task and task != "vision":
            resolved_task = task
        else:
            resolved_task = classify(combined_query, has_images=False)

        parts: list[str] = []
        sources: list[str] = []
        used = "нет"

        if extracted_image_text:
            extracted_image_text = format_math_markdown(extracted_image_text)
            parts.append("РАСПОЗНАННЫЙ ТЕКСТ И ФОРМУЛЫ С ФОТО:\n" + extracted_image_text)
            used = "фото"

        is_meta = not extracted_image_text and _is_meta_or_conversational(clean_question)
        if is_meta:
            loaded_courses = self.store.list_courses()
            if loaded_courses:
                parts.append(
                    "ДОСТУПНЫЕ СЕЙЧАС В БАЗЕ КУРСЫ СТУДЕНТА:\n"
                    + "\n".join(f"- {c}" for c in loaded_courses)
                )
        else:
            hits = self.store.search(combined_query, k=4, course=course, min_score=2.2)
            if hits:
                blocks, chars_used = [], 0
                for chunk, _score in hits:
                    block = f"[{chunk.id}] «{chunk.title}», фр. {chunk.index + 1}\n{chunk.text}"
                    if chars_used + len(block) > 4500:
                        break
                    blocks.append(block)
                    chars_used += len(block)
                    sources.append(f"{chunk.title} (фр. {chunk.index + 1})")
                if blocks:
                    parts.append("КОНТЕКСТ ИЗ МАТЕРИАЛОВ:\n" + "\n\n---\n\n".join(blocks))
                    used = "фото + материалы" if extracted_image_text else "материалы"

        web_results: list[SearchResult] = []
        if use_web and not is_meta:
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
            (resolved_task in self.verify_tasks)
            if verify is None
            else (bool(verify) and resolved_task in self.verify_tasks)
        )
        report = None
        if should_verify and answer_text:
            report = self.verifier.verify(
                problem=combined_query,
                solution=answer_text,
                solver_model=result.model,
                use_model=False,
            )

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
            task="math",
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
                    blocks.append(f"[{ch.id}] «{ch.title}»:\n{ch.text}")
                    sources.append(f"{ch.title} (фр. {ch.index + 1})")
                theory_context = "\n\n---\n\n".join(blocks)
                used = "материалы курса"

        prompt_parts: list[str] = []
        if theory_context:
            prompt_parts.append(f"ТЕОРИЯ И ОБРАЗЦЫ РЕШЕНИЙ ИЗ ЛЕКЦИИ КУРСА:\n{theory_context[:6500]}")

        q_block = f"УСЛОВИЕ ЗАДАНИЯ:\n{question_text.strip()}"
        if options:
            opts_lines = [f"- Вариант {idx}: {opt.get('text', '')}" for idx, opt in enumerate(options, 1)]
            q_block += "\n\nВАРИАНТЫ ОТВЕТА:\n" + "\n".join(opts_lines)
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
    safe_json = re.sub(r'\\(?![/\"\\]|u[0-9a-fA-F]{4})', r"\\\\", raw_json)
    try:
        data = json.loads(safe_json)
    except json.JSONDecodeError:
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return []
    tasks = data.get("tasks") if isinstance(data, dict) else data
    if not isinstance(tasks, list):
        return []
    out = []
    for item in tasks:
        if isinstance(item, dict) and item.get("statement"):
            out.append(
                {
                    "statement": format_math_markdown(str(item["statement"]).strip()),
                    "answer": format_math_markdown(str(item.get("answer", "")).strip()),
                    "solution": format_math_markdown(str(item.get("solution", "")).strip()),
                    "difficulty": str(item.get("difficulty", "")).strip(),
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
