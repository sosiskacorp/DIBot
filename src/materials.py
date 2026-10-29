"""Материалы, конспекты и поиск по ним.

Логика:
  * текст лекции режется на фрагменты с перекрытием, чтобы не рвать формулы;
  * конспект строится по фрагментам и обязан ссылаться на источник — тезис
    без ссылки считается галлюцинацией и помечается;
  * поиск по своим материалам работает на BM25 (без эмбеддингов), потому что
    отдельной embedding-модели в хранилище нет.

Хранилище: data/materials/{index.json, sources/, notes/}
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "materials"

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_SUFFIXES = (
    "иями", "ями", "ами", "ов", "ев", "ий", "ый", "ая", "ое", "ые", "ой",
    "ем", "ом", "ах", "ях", "ия", "ии", "ть", "ся", "ий", "а", "я", "ы", "и",
    "о", "е", "у", "ю", "ь",
)


def tokenize(text: str) -> list[str]:
    words = (w.lower() for w in _WORD_RE.findall(text))
    out = []
    for word in words:
        if len(word) <= 3:
            out.append(word)
            continue
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        out.append(word)
    return out


@dataclass
class Chunk:
    id: str
    source_id: str
    course: str
    title: str
    url: str
    index: int
    text: str

    @property
    def citation(self) -> str:
        return f"{self.title}, фрагмент {self.index + 1}"


@dataclass
class Source:
    id: str
    course: str
    title: str
    url: str
    chars: int
    added: str
    chunk_ids: list[str] = field(default_factory=list)


def slugify(text: str, max_len: int = 80) -> str:
    """Безопасное имя файла из произвольной строки (кириллица сохраняется)."""
    cleaned = re.sub(r"[^\w\s\-]+", "", text, flags=re.UNICODE)
    cleaned = re.sub(r"[\s_]+", "-", cleaned).strip("-").lower()
    return (cleaned or "material")[:max_len]


def chunk_text(text: str, size: int = 1500, overlap: int = 180) -> list[str]:
    """Нарезать текст по абзацам, не разрывая короткие блоки."""
    text = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        # Слишком длинный абзац режем принудительно.
        while len(paragraph) > size:
            head, paragraph = paragraph[:size], paragraph[size - overlap :]
            if current:
                chunks.append(current.strip())
                current = ""
            chunks.append(head.strip())

        if len(current) + len(paragraph) + 2 <= size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current.strip())
            tail = current[-overlap:] if current else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph

    if current:
        chunks.append(current.strip())
    return chunks


class MaterialStore:
    """Диск + индекс + BM25."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.index_path = self.root / "index.json"
        self.sources_dir = self.root / "sources"
        self.notes_dir = self.root / "notes"
        for directory in (self.root, self.sources_dir, self.notes_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.sources: dict[str, Source] = {}
        self.chunks: dict[str, Chunk] = {}
        self._load()

    # ------------------------------------------------------------ хранение

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        for raw in data.get("sources", []):
            source = Source(**raw)
            self.sources[source.id] = source
        for raw in data.get("chunks", []):
            chunk = Chunk(**raw)
            self.chunks[chunk.id] = chunk

    def _save(self) -> None:
        payload = {
            "version": 1,
            "sources": [asdict(s) for s in self.sources.values()],
            "chunks": [asdict(c) for c in self.chunks.values()],
        }
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)

    # ------------------------------------------------------------- контент

    def add_source(
        self,
        course: str,
        title: str,
        text: str,
        url: str = "",
        chunk_size: int = 1500,
        overlap: int = 180,
    ) -> Source:
        source_id = slugify(f"{course}-{title}")[:80]
        if source_id in self.sources:
            source_id = f"{source_id}-{int(time.time()) % 10000}"

        raw_path = self.sources_dir / f"{source_id}.txt"
        raw_path.write_text(text, encoding="utf-8")

        source = Source(
            id=source_id,
            course=course,
            title=title,
            url=url,
            chars=len(text),
            added=datetime.now().isoformat(timespec="seconds"),
        )

        for i, piece in enumerate(chunk_text(text, size=chunk_size, overlap=overlap)):
            chunk = Chunk(
                id=f"{source_id}#{i}",
                source_id=source_id,
                course=course,
                title=title,
                url=url,
                index=i,
                text=piece,
            )
            self.chunks[chunk.id] = chunk
            source.chunk_ids.append(chunk.id)

        self.sources[source_id] = source
        self._save()
        return source

    def remove_source(self, source_id: str) -> bool:
        source = self.sources.pop(source_id, None)
        if not source:
            return False
        for chunk_id in source.chunk_ids:
            self.chunks.pop(chunk_id, None)
        (self.sources_dir / f"{source_id}.txt").unlink(missing_ok=True)
        self._save()
        return True

    # --------------------------------------------------------------- поиск

    def search(
        self,
        query: str,
        k: int = 5,
        course: str | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """BM25 по фрагментам. Возвращает (фрагмент, вес)."""
        candidates = [
            c for c in self.chunks.values() if course is None or c.course == course
        ]
        if not candidates or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        doc_tokens = {c.id: tokenize(c.text) for c in candidates}
        avg_len = sum(len(t) for t in doc_tokens.values()) / len(doc_tokens) or 1.0
        doc_freq: dict[str, int] = {}
        for tokens in doc_tokens.values():
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total_docs = len(doc_tokens)
        k1, b = 1.5, 0.75
        scored: list[tuple[Chunk, float]] = []

        for chunk in candidates:
            tokens = doc_tokens[chunk.id]
            if not tokens:
                continue
            length = len(tokens)
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                freq = counts.get(token, 0)
                if not freq:
                    continue
                df = doc_freq.get(token, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += idf * (freq * (k1 + 1)) / (
                    freq + k1 * (1 - b + b * length / avg_len)
                )
            if score > min_score:
                scored.append((chunk, round(score, 3)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]

    def context_for(self, query: str, k: int = 5, max_chars: int = 6000) -> str:
        """Собрать контекст с маркерами ссылок для промпта."""
        hits = self.search(query, k=k)
        if not hits:
            return ""
        blocks, used = [], 0
        for chunk, _score in hits:
            block = f"[{chunk.id}] «{chunk.title}», фр. {chunk.index + 1}\n{chunk.text}"
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n---\n\n".join(blocks)

    # ------------------------------------------------------------ конспект

    def list_courses(self) -> list[str]:
        return sorted({s.course for s in self.sources.values()})

    def list_sources(self, course: str | None = None) -> list[Source]:
        items = [s for s in self.sources.values() if course is None or s.course == course]
        return sorted(items, key=lambda s: (s.course, s.title))

    def stats(self) -> dict[str, Any]:
        return {
            "sources": len(self.sources),
            "chunks": len(self.chunks),
            "courses": self.list_courses(),
            "chars": sum(s.chars for s in self.sources.values()),
        }
