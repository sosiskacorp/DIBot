"""Оркестратор: вопрос -> контекст -> решение -> проверка.

Это единственное место, где сходятся все слои. Правила:
  * если в материалах есть ответ — он идёт в контекст со ссылками;
  * если материалов нет, модель отвечает по общим знаниям, но это помечено;
  * математика и код проверяются автоматически, а не «на глаз»;
  * в ответе всегда видно, какая модель отвечала и чем проверено.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from llm_client import ChatResult, LLMClient, LLMError
from materials import MaterialStore
from model_router import classify
from notes import CITATION_RE
from search_client import SearchError, SearchResult, WebSearch
from verification import VerificationReport, Verifier

TUTOR_SYSTEM = """Ты — учебный ассистент студента. Твоя задача — объяснять и решать
так, чтобы студент мог воспроизвести решение сам.

Правила:
1. Если дан КОНТЕКСТ ИЗ МАТЕРИАЛОВ, опирайся на него и ставь ссылку [#id]
   после каждого утверждения, взятого оттуда. Не пересказывай контекст целиком.
2. Если в материалах ответа нет — реши сам и явно напиши: «в материалах этого
   нет, отвечаю по общим знаниям».
3. Любые вычисления показывай пошагово, а не только результат.
4. Для задач оформляй решение как в тетради: дано, ход, ответ.
5. Завершай строкой «Ответ: ...» — по ней работает автоматическая проверка.
6. Никаких обращений к читателю вроде «как видите» и воды.
7. Если вопрос теоретический («что такое», «объясни»), отвечай связным
   объяснением без блоков «Дано / Ход решения / Ответ»."""

PRACTICE_SYSTEM = """Ты составляешь тренировочные задания по учебной теме.

Отвечай ТОЛЬКО JSON:
{"tasks": [{"statement": "условие", "answer": "краткий ответ",
            "solution": "пошаговое решение", "difficulty": "лёгкий|средний|сложный"}]}

Требования:
- задания должны быть решаемы теми методами, что в материалах;
- ответ должен быть однозначным и проверяемым;
- condition самодостаточен: никаких «см. выше» и ссылок на рисунки."""

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
    ) -> Answer:
        started = time.perf_counter()
        clean_question = (question or "").strip() or "Разбери и реши задание на изображении."
        extracted_image_text = ""
        vision_model_used = ""

        # Шаг 1: если прикреплены фото, распознаём текст и формулы через vision-модель (qwen3-vl)
        if images:
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
            parts.append("РАСПОЗНАННЫЙ ТЕКСТ И ФОРМУЛЫ С ФОТО:\n" + extracted_image_text)
            used = "фото"

        notes_context = self.store.context_for(combined_query, k=4, course=course)
        if notes_context:
            parts.append("КОНТЕКСТ ИЗ МАТЕРИАЛОВ:\n" + notes_context)
            used = "фото + материалы" if extracted_image_text else "материалы"
            sources.extend(
                f"{chunk.title} (фр. {chunk.index + 1})"
                for chunk, _ in self.store.search(combined_query, k=4, course=course)
            )

        web_results: list[SearchResult] = []
        if use_web:
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
        )
        answer_text = _strip_prompt_echo(result.text, TUTOR_SYSTEM)

        model_badge = (
            f"{vision_model_used} -> {result.model}"
            if vision_model_used and vision_model_used != result.model
            else result.model
        )

        should_verify = (
            resolved_task in self.verify_tasks if verify is None else verify
        )
        report = None
        if should_verify and answer_text:
            report = self.verifier.verify(
                problem=combined_query,
                solution=answer_text,
                solver_model=result.model,
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
    ) -> tuple[list[dict[str, Any]], list[VerificationReport]]:
        """Сгенерировать тренировочные задания и проверить эталонные решения.

        Это замена «прохождению теста»: те же типы задач, но ответы проверяются
        заранее, чтобы студент учился на корректных эталонах.
        """
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
            json_format="json",
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
            ]
        )
        return tasks, reports

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
    try:
        data = json.loads(cleaned[start : end + 1])
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
                    "statement": str(item["statement"]).strip(),
                    "answer": str(item.get("answer", "")).strip(),
                    "solution": str(item.get("solution", "")).strip(),
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
