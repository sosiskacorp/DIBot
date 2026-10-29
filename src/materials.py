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

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "materials"

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_SUFFIXES = (
    "иями", "ями", "ами", "ов", "ев", "ий", "ый", "ая", "ое", "ые", "ой",
    "ем", "ом", "ах", "ях", "ия", "ии", "ть", "ся", "ий", "а", "я", "ы", "и",
    "о", "е", "у", "ю", "ь",
)
_STOP_WORDS = {
    "и", "в", "во", "не", "что", "он", "на", "я", "с", "со", "как", "а", "то",
    "все", "всё", "она", "так", "его", "но", "да", "ты", "к", "у", "же", "вы",
    "за", "бы", "по", "только", "ее", "её", "мне", "было", "вот", "от", "меня",
    "еще", "ещё", "нет", "о", "из", "ему", "теперь", "когда", "даже", "ну",
    "вдруг", "ли", "если", "уже", "или", "ни", "быть", "был", "него", "до",
    "вас", "нибудь", "опять", "уж", "вам", "ведь", "там", "потом", "себя",
    "ничего", "ей", "может", "они", "тут", "где", "есть", "надо", "ней", "для",
    "мы", "тебя", "их", "чем", "была", "сам", "чтоб", "чтобы", "без", "будто",
    "чего", "раз", "тоже", "себе", "под", "будет", "ж", "тогда", "кто", "этот",
    "того", "потому", "этого", "какой", "совсем", "ним", "здесь", "этом",
    "один", "почти", "мой", "тем", "нее", "неё", "сейчас", "были", "куда",
    "зачем", "всех", "никогда", "можно", "при", "наконец", "два", "об", "другой",
    "хоть", "после", "над", "больше", "тот", "через", "эти", "нас", "про",
    "всего", "них", "какая", "какие", "какое", "много", "разве", "три", "эту",
    "моя", "впрочем", "хорошо", "свою", "этой", "перед", "иногда", "лучше",
    "чуть", "том", "нельзя", "такой", "такая", "такое", "такие", "им", "более",
    "всегда", "конечно", "всю", "между", "это", "эта", "можешь", "умеешь",
    "могу", "можете", "можем", "умею", "умеет", "привет", "здравствуй",
    "здравствуйте", "помоги", "помогите", "расскажи", "скажи", "объясни",
    "пожалуйста", "спасибо", "приветствую", "дела", "тебе", "тобой", "вами",
}


def tokenize(text: str) -> list[str]:
    words = (w.lower().replace("ё", "е") for w in _WORD_RE.findall(text))
    out = []
    for word in words:
        if word in _STOP_WORDS or len(word) <= 1:
            continue
        if len(word) <= 3:
            out.append(word)
            continue
        for suffix in _SUFFIXES:
            if word.endswith(suffix) and len(word) - len(suffix) >= 4:
                word = word[: -len(suffix)]
                break
        if word not in _STOP_WORDS:
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
    if not cleaned:
        return "material"
    if len(cleaned) <= max_len:
        return cleaned
    short_hash = hashlib.sha1(cleaned.encode("utf-8")).hexdigest()[:6]
    return f"{cleaned[: max_len - 7].rstrip('-')}-{short_hash}"


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
    """Диск + индекс + BM25 с автоматической синхронизацией при изменениях на диске."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else DEFAULT_ROOT
        self.index_path = self.root / "index.json"
        self.sources_dir = self.root / "sources"
        self.notes_dir = self.root / "notes"
        for directory in (self.root, self.sources_dir, self.notes_dir):
            directory.mkdir(parents=True, exist_ok=True)

        self.sources: dict[str, Source] = {}
        self.chunks: dict[str, Chunk] = {}
        self._token_counts: dict[str, dict[str, int]] = {}
        self._chunk_lengths: dict[str, int] = {}
        self._last_mtime_ns: int = 0
        self._load()

    def _index_chunk_tokens(self, chunk: Chunk) -> None:
        tokens = tokenize(chunk.text)
        counts: dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        self._token_counts[chunk.id] = counts
        self._chunk_lengths[chunk.id] = len(tokens)

    # ------------------------------------------------------------ хранение

    def _disk_mtime_ns(self) -> int:
        try:
            return self.index_path.stat().st_mtime_ns if self.index_path.exists() else 0
        except OSError:
            return 0

    def reload(self) -> None:
        """Принудительно перечитать индекс с диска."""
        self._load()

    def _sync_if_changed(self) -> None:
        """Если index.json обновился другим процессом (например, CLI), перечитать его."""
        current_mtime = self._disk_mtime_ns()
        if current_mtime != self._last_mtime_ns:
            self._load()

    def _load(self) -> None:
        self.sources.clear()
        self.chunks.clear()
        self._token_counts.clear()
        self._chunk_lengths.clear()
        if not self.index_path.exists():
            self._last_mtime_ns = 0
            return
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._last_mtime_ns = self._disk_mtime_ns()
        except (json.JSONDecodeError, OSError):
            return
        for raw in data.get("sources", []):
            source = Source(**raw)
            self.sources[source.id] = source
        for raw in data.get("chunks", []):
            chunk = Chunk(**raw)
            self.chunks[chunk.id] = chunk
            self._index_chunk_tokens(chunk)

    def _save(self) -> None:
        payload = {
            "version": 1,
            "sources": [asdict(s) for s in self.sources.values()],
            "chunks": [asdict(c) for c in self.chunks.values()],
        }
        tmp = self.index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.index_path)
        self._last_mtime_ns = self._disk_mtime_ns()

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
        self._sync_if_changed()
        course = course.strip()
        title = title.strip()
        source_id = slugify(f"{course}-{title}")

        # Если такой материал (курс + название) уже есть — обновляем его на месте без дублей.
        existing_id: str | None = None
        if source_id in self.sources:
            existing_id = source_id
        else:
            for sid, existing in self.sources.items():
                if existing.course.lower() == course.lower() and existing.title.lower() == title.lower():
                    existing_id = sid
                    break

        if existing_id:
            old_source = self.sources.pop(existing_id, None)
            if old_source:
                for cid in old_source.chunk_ids:
                    self.chunks.pop(cid, None)
                    self._token_counts.pop(cid, None)
                    self._chunk_lengths.pop(cid, None)
            source_id = existing_id

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
            self._index_chunk_tokens(chunk)
            source.chunk_ids.append(chunk.id)

        self.sources[source_id] = source
        self._save()
        return source

    def remove_source(self, source_id: str) -> bool:
        self._sync_if_changed()
        source = self.sources.pop(source_id, None)
        if not source:
            return False
        for chunk_id in source.chunk_ids:
            self.chunks.pop(chunk_id, None)
            self._token_counts.pop(chunk_id, None)
            self._chunk_lengths.pop(chunk_id, None)
        (self.sources_dir / f"{source_id}.txt").unlink(missing_ok=True)
        self._save()
        return True

    def get_source_text(self, source_id: str) -> str:
        self._sync_if_changed()
        raw_path = self.sources_dir / f"{source_id}.txt"
        if raw_path.exists():
            try:
                return raw_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        source = self.sources.get(source_id)
        if not source:
            return ""
        return "\n\n".join(
            self.chunks[cid].text for cid in source.chunk_ids if cid in self.chunks
        )

    # --------------------------------------------------------------- поиск

    def search(
        self,
        query: str,
        k: int = 5,
        course: str | None = None,
        min_score: float = 0.0,
    ) -> list[tuple[Chunk, float]]:
        """BM25 по фрагментам с предвычисленным индексом токенов. Возвращает (фрагмент, вес)."""
        self._sync_if_changed()
        candidates = [
            c for c in self.chunks.values() if course is None or c.course == course
        ]
        if not candidates or not query.strip():
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        total_docs = len(candidates)
        avg_len = (
            sum(self._chunk_lengths.get(c.id, 0) for c in candidates) / total_docs
        ) or 1.0

        unique_q = set(query_tokens)
        doc_freq: dict[str, int] = {qt: 0 for qt in unique_q}
        for c in candidates:
            c_counts = self._token_counts.get(c.id) or {}
            for qt in unique_q:
                if qt in c_counts:
                    doc_freq[qt] += 1

        k1, b = 1.5, 0.75
        scored: list[tuple[Chunk, float]] = []

        for chunk in candidates:
            counts = self._token_counts.get(chunk.id)
            if not counts:
                continue
            length = self._chunk_lengths.get(chunk.id, 1)

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

    def context_for(
        self,
        query: str,
        k: int = 5,
        max_chars: int = 6000,
        course: str | None = None,
        min_score: float = 2.0,
    ) -> str:
        """Собрать контекст с маркерами ссылок для промпта (только релевантные фрагменты)."""
        hits = self.search(query, k=k, course=course, min_score=min_score)
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
        self._sync_if_changed()
        return sorted({s.course for s in self.sources.values()})

    def list_sources(self, course: str | None = None) -> list[Source]:
        self._sync_if_changed()
        items = [s for s in self.sources.values() if course is None or s.course == course]
        return sorted(items, key=lambda s: (s.course, s.title))

    def stats(self) -> dict[str, Any]:
        self._sync_if_changed()
        return {
            "sources": len(self.sources),
            "chunks": len(self.chunks),
            "courses": self.list_courses(),
            "chars": sum(s.chars for s in self.sources.values()),
        }
