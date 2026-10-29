"""Генерация конспектов с обязательными ссылками на источник.

Главное правило: каждый тезис конспекта несёт маркер фрагмента, из которого
он взят. После генерации ссылки проверяются — выдуманный тезис останется без
ссылки, и это видно в отчёте о покрытии, а не прячется.

Маркер ссылки короткий: [#3] — третий фрагмент этого источника. Конспект
строится по одному источнику, поэтому номеров достаточно и они читаемы.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

from llm_client import LLMClient, LLMError
from materials import Chunk, MaterialStore, Source, slugify

NOTES_SYSTEM = """Ты конспектируешь учебный материал для студента.

Жёсткие правила:
1. Используй ТОЛЬКО информацию из фрагмента. Ничего не добавляй от себя.
2. Сделай 3–7 тезисов — самое важное, а не пересказ каждого предложения.
3. Не копируй предложения дословно: сжимай формулировки, сохраняя смысл.
4. Каждый тезис — отдельная строка, начинается с "- ".
5. В конце каждого тезиса ставь маркер фрагмента, ровно как указано во
   входных данных, например [#3].
6. Формулы сохраняй в $...$ (LaTeX) без изменений.
7. Определения, теоремы и условия — отдельными тезисами.
8. Если во фрагменте нет содержательного материала (колонтитул, нумерация),
   ответь одним словом: ПУСТО.
9. Никаких вступлений, выводов и обращений к читателю."""

CITATION_RE = re.compile(r"\[#(\d+)\]")
_THESIS_PREFIX = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


@dataclass
class NoteResult:
    path: Path
    text: str
    source: Source
    chunks_used: int
    theses: int = 0
    cited: int = 0
    skipped_chunks: list[int] = field(default_factory=list)
    duplicate_theses: int = 0

    @property
    def coverage(self) -> float:
        return self.cited / self.theses if self.theses else 0.0

    def summary_line(self) -> str:
        return (
            f"{self.source.title}: {self.theses} тезисов, "
            f"со ссылками {self.cited} ({self.coverage:.0%}), "
            f"фрагментов {self.chunks_used}"
            + (f", отброшено дублей {self.duplicate_theses}" if self.duplicate_theses else "")
        )


class NoteBuilder:
    def __init__(self, llm: LLMClient | None = None, store: MaterialStore | None = None):
        self.llm = llm or LLMClient()
        self.store = store or MaterialStore()

    def build(
        self,
        source: Source,
        max_chunks: int | None = None,
        on_progress=None,
    ) -> NoteResult:
        chunks: Sequence[Chunk] = [
            self.store.chunks[cid] for cid in source.chunk_ids if cid in self.store.chunks
        ]
        if max_chunks:
            chunks = chunks[:max_chunks]
        if not chunks:
            raise ValueError(f"У источника {source.id} нет фрагментов")

        sections: list[tuple[Chunk, str]] = []
        skipped: list[int] = []
        seen: set[str] = set()
        duplicates = 0

        for position, chunk in enumerate(chunks, start=1):
            if on_progress:
                on_progress(position, len(chunks), chunk)
            lines = self._summarise_chunk(chunk)
            if not lines:
                skipped.append(chunk.index)
                continue

            kept = []
            for line in lines:
                key = _fingerprint(line)
                # Фрагменты перекрываются, поэтому тезисы на стыке повторяются.
                if key in seen:
                    duplicates += 1
                    continue
                seen.add(key)
                kept.append(line)
            if kept:
                sections.append((chunk, "\n".join(kept)))

        text = self._render(source, sections)
        path = self._save(source, text)
        theses, cited = _count_citations(text)
        return NoteResult(
            path=path,
            text=text,
            source=source,
            chunks_used=len(chunks),
            theses=theses,
            cited=cited,
            skipped_chunks=skipped,
            duplicate_theses=duplicates,
        )

    # ------------------------------------------------------------ внутреннее

    def _summarise_chunk(self, chunk: Chunk) -> list[str]:
        marker = f"[#{chunk.index + 1}]"
        prompt = (
            f"ФРАГМЕНТ {chunk.index + 1} из «{chunk.title}»:\n\n"
            f"{chunk.text}\n\n"
            f"Сделай конспект: 3–7 тезисов, маркер ссылки {marker}"
        )
        try:
            result = self.llm.chat(
                [
                    {"role": "system", "content": NOTES_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                task="notes",
            )
        except LLMError:
            return []

        text = result.text.strip()
        if not text or text.upper().startswith("ПУСТО"):
            return []

        lines: list[str] = []
        for raw in text.splitlines():
            clean = _THESIS_PREFIX.sub("", raw).strip()
            if not clean:
                continue
            if not CITATION_RE.search(clean):
                clean = f"{clean} {marker}"
            lines.append(f"- {clean}")
        return lines[:8]

    def _render(self, source: Source, sections: list[tuple[Chunk, str]]) -> str:
        now = datetime.now().strftime("%d.%m.%Y %H:%M")
        header = [
            f"# {source.title}",
            "",
            f"> Курс: **{source.course}**  ",
            f"> Источник: {source.url or 'локальный файл'}  ",
            f"> Собрано: {now}  ",
            f"> Содержательных частей: {len(sections)}  ",
            "",
            "Ссылки вида [#N] указывают на номер фрагмента исходного текста.",
            "",
        ]
        body: list[str] = []
        for chunk, text in sections:
            body.append(f"## Часть {chunk.index + 1}")
            body.append("")
            body.append(text)
            body.append("")

        footer = ["## Источники", ""]
        footer.append(
            f"1. {source.title} — {source.url or 'локальный файл'} "
            f"({source.chars} симв., {len(source.chunk_ids)} фрагментов)"
        )
        return "\n".join(header + body + footer)

    def _save(self, source: Source, text: str) -> Path:
        course_dir = self.store.notes_dir / slugify(source.course)
        course_dir.mkdir(parents=True, exist_ok=True)
        path = course_dir / f"{slugify(source.title)}.md"
        path.write_text(text, encoding="utf-8")
        return path


def _fingerprint(text: str) -> str:
    """Ключ для поиска повторов: без разметки, регистра и пунктуации."""
    clean = CITATION_RE.sub("", text)
    clean = re.sub(r"[^\wа-яА-ЯёЁ]+", "", clean.lower())
    return clean[:160]


def _count_citations(text: str) -> tuple[int, int]:
    theses = [ln for ln in text.splitlines() if ln.strip().startswith("- ")]
    cited = [ln for ln in theses if CITATION_RE.search(ln)]
    return len(theses), len(cited)


def verify_citations(text: str, valid_numbers: set[int]) -> tuple[int, list[int]]:
    """Проверить, что маркеры ссылаются на существующие фрагменты источника."""
    found = [int(n) for n in CITATION_RE.findall(text)]
    bad = sorted({n for n in found if n not in valid_numbers})
    return len(found), bad
