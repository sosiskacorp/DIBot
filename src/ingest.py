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
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import LLMClient, LLMError  # noqa: E402
from materials import MaterialStore, slugify  # noqa: E402
from notes import NoteBuilder, verify_citations  # noqa: E402

SUPPORTED = {".txt", ".md", ".markdown"}


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED:
        for encoding in ("utf-8", "utf-8-sig", "cp1251"):
            try:
                return path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise SystemExit(f"не удалось определить кодировку: {path}")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise SystemExit(
                "для PDF нужен pypdf:  python -m pip install pypdf\n"
                f"либо сохраните текст вручную: {path}"
            )
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise SystemExit(f"PDF без текстового слоя (скан): {path} — нужен OCR через qwen3-vl")
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
    parser.add_argument("--file", type=Path, help="файл: .txt, .md, .pdf")
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
        files = [p for p in sorted(args.dir.rglob("*")) if p.suffix.lower() in SUPPORTED | {".pdf"}]
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
