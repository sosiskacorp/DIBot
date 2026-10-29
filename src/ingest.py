"""Импорт учебных материалов в локальное хранилище.

    python src/ingest.py --course "Линейная алгебра" --title "Лекция 1" --file lecture.txt
    python src/ingest.py --course "ЛУР" --title "Методичка" --file m.pdf
    python src/ingest.py --dir data/samples --course "Линейная алгебра" --notes
    python src/ingest.py --list

Материалы режутся на фрагменты и попадают в поиск. С флагом --notes сразу
строится конспект со ссылками на фрагменты.
"""

from __future__ import annotations

import argparse
import html
import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient, LLMError  # noqa: E402
from materials import MaterialStore, slugify  # noqa: E402
from notes import NoteBuilder, verify_citations  # noqa: E402

SUPPORTED_TEXT = {".txt", ".md", ".markdown"}
SUPPORTED_BINARY = {".pdf", ".docx"}
SUPPORTED = SUPPORTED_TEXT | SUPPORTED_BINARY

# Раздробленные разрядкой заголовки в русских лекциях («О п р е д е л е н и е»).
_SPACED_WORDS = {
    r"О\s+п\s+р\s+е\s+д\s+е\s+л\s+е\s+н\s+и\s+е": "Определение",
    r"Т\s+е\s+о\s+р\s+е\s+м\s+а": "Теорема",
    r"С\s+л\s+е\s+д\s+с\s+т\s+в\s+и\s+е": "Следствие",
    r"Д\s+о\s+к\s+а\s+з\s+а\s+т\s+е\s+л\s+ь\s+с\s+т\s+в\s+о": "Доказательство",
    r"З\s+а\s+м\s+е\s+ч\s+а\s+н\s+и\s+е": "Замечание",
    r"П\s+р\s+и\s+м\s+е\s+р": "Пример",
    r"У\s+т\s+в\s+е\s+р\s+ж\s+д\s+е\s+н\s+и\s+е": "Утверждение",
    r"С\s+в\s+о\s+й\s+с\s+т\s+в\s+а": "Свойства",
    r"С\s+в\s+о\s+й\s+с\s+т\s+в\s+о": "Свойство",
    r"Л\s+е\s+м\s+м\s+а": "Лемма",
}


def decode_tex_cp1251(text: str) -> str:
    """Расшифровать глифы вида /xc2/xe5/xea из старых русскоязычных TeX/LaTeX PDF."""
    if "/x" not in text:
        return text

    def _repl(match: re.Match) -> str:
        hex_bytes = re.findall(r"/x([0-9a-fA-F]{2})", match.group(0))
        return bytes(int(h, 16) for h in hex_bytes).decode("cp1251", errors="replace")

    return re.sub(r"(?:/x[0-9a-fA-F]{2})+", _repl, text)


def extract_pdf_bytes(data: bytes) -> str:
    """Извлечь текст из байтов PDF (с декодированием CP1251 TeX-глифов)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("для PDF нужен pypdf: python -m pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    raw = "\n\n".join(pages).strip()
    return decode_tex_cp1251(raw).strip()


def extract_docx_bytes(data: bytes) -> str:
    """Извлечь текст из байтов DOCX (без внешних зависимостей)."""
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    paragraphs: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        if "word/document.xml" not in zf.namelist():
            return ""
        xml_bytes = zf.read("word/document.xml")
    root = ET.fromstring(xml_bytes)
    for para in root.iter(f"{ns}p"):
        runs = [node.text for node in para.iter(f"{ns}t") if node.text]
        if runs:
            line = "".join(runs).strip()
            if line:
                paragraphs.append(line)
    return "\n\n".join(paragraphs).strip()


def html_to_text(raw_html: str) -> str:
    """Преобразовать HTML лекции DiSpace в чистый текст с сохранением LaTeX ($...$)."""
    if not raw_html or not raw_html.strip():
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for li in soup.find_all("li"):
            li.insert_before("\n• ")
        for block in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote"]):
            block.insert_before("\n")
            block.insert_after("\n")
        text = soup.get_text()
    except ImportError:
        text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)

    text = text.replace("\xa0", " ")
    for pat, replacement in _SPACED_WORDS.items():
        text = re.sub(pat, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise SystemExit(f"не удалось определить кодировку: {path}")

    if suffix == ".pdf":
        try:
            text = extract_pdf_bytes(path.read_bytes())
        except RuntimeError as exc:
            raise SystemExit(f"{exc}\nлибо сохраните текст вручную: {path}") from exc
        if not text:
            raise SystemExit(f"PDF без текстового слоя (скан): {path} — нужен OCR через qwen3-vl")
        return text

    if suffix == ".docx":
        text = extract_docx_bytes(path.read_bytes())
        if not text:
            raise SystemExit(f"DOCX пуст или не содержит текста: {path}")
        return text

    raise SystemExit(f"формат не поддерживается: {suffix}")


def import_file(
    store: MaterialStore,
    path: Path,
    course: str,
    title: str | None = None,
    url: str = "",
) -> str:
    text = read_text_file(path)
    source = store.add_source(course=course, title=title or path.stem, text=text, url=url)
    print(
        f"  + {source.title}: {source.chars} симв., "
        f"{len(source.chunk_ids)} фрагментов  (id: {source.id})"
    )
    return source.id


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Импорт лекций и методичек в хранилище")
    parser.add_argument("--course", help="название курса")
    parser.add_argument("--title", help="название материала")
    parser.add_argument("--file", type=Path, help="файл: .txt, .md, .pdf, .docx")
    parser.add_argument("--dir", type=Path, help="папка с файлами для массового импорта")
    parser.add_argument("--url", default="", help="ссылка на исходник (для конспекта)")
    parser.add_argument("--text", help="текст материала прямо в командной строке")
    parser.add_argument("--notes", action="store_true", help="сразу построить конспект")
    parser.add_argument("--notes-for", metavar="ID", help="построить конспект для существующего источника")
    parser.add_argument("--list", action="store_true", help="показать хранилище")
    parser.add_argument("--remove", metavar="ID", help="удалить источник по id")
    args = parser.parse_args(argv)

    store = MaterialStore()

    if args.notes_for:
        source = store.sources.get(args.notes_for)
        if not source:
            print(f"источник не найден: {args.notes_for}")
            return 1
        builder = NoteBuilder(LLMClient(), store)

        def progress(position: int, total: int, chunk) -> None:
            print(f"    фрагмент {position}/{total}", flush=True)

        result = builder.build(source, on_progress=progress)
        valid = {c.index + 1 for c in store.chunks.values() if c.source_id == source.id}
        _, bad = verify_citations(result.text, valid)
        print(f"  -> {result.path}")
        print(f"     {result.summary_line()}")
        if bad:
            print(f"     ВНИМАНИЕ: ссылки на несуществующие фрагменты: {bad}")
        return 0

    if args.list:
        stats = store.stats()
        print(f"Источников: {stats['sources']}, фрагментов: {stats['chunks']}")
        for course in stats["courses"]:
            print(f"\n{course}:")
            for source in store.list_sources(course):
                print(f"  [{source.id}] {source.title} — {source.chars} симв., {len(source.chunk_ids)} фр.")
        return 0

    if args.remove:
        print("удалено" if store.remove_source(args.remove) else "не найдено")
        return 0

    if not args.course:
        parser.error("нужен --course")

    imported: list[str] = []

    if args.file:
        imported.append(import_file(store, args.file, args.course, args.title, args.url))
    elif args.dir:
        files = [p for p in sorted(args.dir.rglob("*")) if p.suffix.lower() in SUPPORTED]
        if not files:
            print(f"в {args.dir} нет поддерживаемых файлов")
            return 1
        print(f"найдено файлов: {len(files)}")
        for path in files:
            imported.append(import_file(store, path, args.course, None, args.url))
    elif args.text:
        source = store.add_source(
            course=args.course, title=args.title or "Без названия", text=args.text, url=args.url
        )
        imported.append(source.id)
        print(f"  + {source.title}: {source.chars} симв., {len(source.chunk_ids)} фрагментов")
    else:
        parser.error("нужен один из вариантов: --file, --dir, --text")

    if args.notes and imported:
        llm = LLMClient()
        builder = NoteBuilder(llm, store)
        for source_id in imported:
            source = store.sources[source_id]

            def progress(position: int, total: int, chunk) -> None:
                print(f"    фрагмент {position}/{total}", flush=True)

            try:
                result = builder.build(source, on_progress=progress)
            except (LLMError, ValueError) as exc:
                print(f"  ! конспект не построен: {exc}")
                continue

            valid = {
                chunk.index + 1
                for chunk in store.chunks.values()
                if chunk.source_id == source_id
            }
            _, bad = verify_citations(result.text, valid)
            print(f"  -> {result.path}")
            print(f"     {result.summary_line()}")
            if bad:
                print(f"     ВНИМАНИЕ: ссылки на несуществующие фрагменты: {bad}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
