"""Импорт учебных материалов в локальное хранилище.

    python src/ingest.py --course "Линейная алгебра" --title "Лекция 1" --file lecture.txt
    python src/ingest.py --course "ЛУР" --title "Методичка" --file m.pdf
    python src/ingest.py --dir data/samples --course "Линейная алгебра" --notes
    python src/ingest.py --list

Материалы режутся на фрагменты и попадают в поиск. С флагом --notes сразу
строится конспект со ссылками на фрагменты.
"""

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

SUPPORTED_TEXT = {".txt", ".md", ".markdown", ".tex", ".csv", ".html", ".htm"}
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

# Частотные русские буквы для автоматического выбора между 8-битными кодировками (CP1251, KOI8-R, CP866, ISO-8859-5)
_COMMON_RU_CHARS = set("оеаинтсрвлкмдпуяыьгзбчйхжшюцщэOЕАИНТСРВЛКМДПУЯЫЬГЗБЧЙХЖШЮЦЩЭ")


def decode_bytes_smart(data: bytes) -> str:
    """Умное декодирование текста с поддержкой UTF-8, UTF-16, CP1251, KOI8-R, CP866, ISO-8859-5, MacCyrillic."""
    if not data:
        return ""

    # 1. Проверка BOM (UTF-8-SIG, UTF-16, UTF-32)
    if data.startswith(b"\xef\xbb\xbf"):
        return fix_mojibake_and_tex(data.decode("utf-8-sig", errors="replace"))
    if data.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return fix_mojibake_and_tex(data.decode("utf-32", errors="replace"))
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return fix_mojibake_and_tex(data.decode("utf-16", errors="replace"))

    # 2. Чистый UTF-8
    try:
        decoded = data.decode("utf-8")
        return fix_mojibake_and_tex(decoded)
    except UnicodeDecodeError:
        pass

    # 3. Если много нулевых байтов — вероятно UTF-16 без BOM
    if data.count(b"\x00") > max(4, len(data) // 10):
        for enc in ("utf-16-le", "utf-16-be"):
            try:
                return fix_mojibake_and_tex(data.decode(enc))
            except UnicodeDecodeError:
                continue

    # 4. Выбор лучшей кириллической 8-битной кодировки по доле естественных русских букв
    best_text = ""
    best_score = -1.0
    for enc in ("cp1251", "koi8-r", "cp866", "iso-8859-5", "mac_cyrillic"):
        try:
            candidate = data.decode(enc)
        except UnicodeDecodeError:
            continue
        common_hits = sum(1 for ch in candidate if ch in _COMMON_RU_CHARS)
        all_cyr = sum(1 for ch in candidate if "А" <= ch <= "я" or ch in "Ёё")
        # В настоящем русском тексте частые буквы составляют >75% всех кириллических символов
        score = common_hits * 2.0 - (all_cyr - common_hits) * 1.5
        if score > best_score:
            best_score = score
            best_text = candidate

    if best_text:
        return fix_mojibake_and_tex(best_text)

    return fix_mojibake_and_tex(data.decode("utf-8", errors="replace"))


def decode_tex_cp1251(text: str) -> str:
    """Расшифровать глифы вида /xc2/xe5/xea и кракозябры Latin1->CP1251 из русскоязычных TeX/LaTeX PDF."""
    return fix_mojibake_and_tex(text)


def _fix_latin1_cp1251_span(match: re.Match) -> str:
    chunk = match.group(0)
    try:
        return chunk.encode("latin1").decode("cp1251")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return chunk


def fix_mojibake_and_tex(text: str) -> str:
    """Исправить кодировки TeX PDF (/xNN, Latin1->CP1251), управляющие символы и псевдографику матриц."""
    if not text:
        return ""

    # 1. Глифы вида /xc2/xe5/xea
    if "/x" in text:
        def _repl_hex(match: re.Match) -> str:
            hex_bytes = re.findall(r"/x([0-9a-fA-F]{2})", match.group(0))
            return bytes(int(h, 16) for h in hex_bytes).decode("cp1251", errors="replace")

        text = re.sub(r"(?:/x[0-9a-fA-F]{2})+", _repl_hex, text)

    # 2. Рамки из старых TeX PDF вида $\n%\n\x18\x18...'\n&\n$ (ломающие KaTeX ложными знаками $)
    text = re.sub(r"\$\s*%\s*[\x00-\x1f\s0-9*+\-XJz?]+(.*?)'\s*&\s*\$", r"\1", text, flags=re.DOTALL)

    # 3. Кракозябры вида Ñèñòåìû ëèíåéíûõ óðàâíåíèé (байты CP1251, прочитанные как Latin-1 \xc0-\xff)
    if re.search(r"[\xc0-\xff]{2,}", text):
        text = re.sub(r"[\xc0-\xff]+(?:\s+[\xc0-\xff]+)*", _fix_latin1_cp1251_span, text)

    # 4. Очистка мусорных управляющих символов PDF (\x00-\x08, \x0b-\x1f)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Замена составных скобок матриц/систем из TeX-шрифтов на читаемые символы
    _bracket_map = {
        "": "(",
        "": "|",
        "": "(",
        "": ")",
        "": "|",
        "": ")",
        "": "[",
        "": "|",
        "": "[",
        "": "]",
        "": "|",
        "": "]",
        "": "{",
        "": "{",
        "": "{",
        "": "|",
        "⏐⏐⏐⏐⏐⏐": "|",
        "⏐⏐⏐⏐": "|",
        "⏐⏐": "|",
        "−": "-",
    }
    for k, v in _bracket_map.items():
        if k in text:
            text = text.replace(k, v)

    for pat, replacement in _SPACED_WORDS.items():
        text = re.sub(pat, replacement, text, flags=re.IGNORECASE)

    text = clean_pdf_artifacts(text)
    return format_math_markdown(text)


def paginate_markdown(text: str, page_size: int = 8000) -> list[str]:
    """Разбить текст на страницы по границам абзацев, не разрывая формулы $$...$$ и блоки кода."""
    if not text:
        return [""]
    paragraphs = text.split("\n\n")
    pages: list[str] = []
    buf: list[str] = []
    buf_len = 0
    open_display = False
    open_code = False
    for para in paragraphs:
        buf.append(para)
        buf_len += len(para) + 2
        if para.count("$$") % 2 == 1:
            open_display = not open_display
        if para.count("```") % 2 == 1:
            open_code = not open_code
        if buf_len >= page_size and not open_display and not open_code:
            pages.append("\n\n".join(buf).strip())
            buf, buf_len = [], 0
    if buf:
        tail = "\n\n".join(buf).strip()
        if tail:
            pages.append(tail)
    return pages or [""]


def repair_latex_control_chars(text: str) -> str:
    """Восстановить команды LaTeX (\\frac, \\text, \\begin, \\times и др.), повреждённые при парсинге JSON."""
    if not text:
        return ""
    s = text.replace("\x0crac", r"\frac").replace("\x0c", r"\f")
    s = s.replace("\x08egin", r"\begin").replace("\x08eta", r"\beta").replace("\x08inom", r"\binom").replace("\x08ar", r"\bar")
    s = re.sub(r"\t(ext\{|imes\b|heta\b|au\b|an\b|riangle\b|o\b|ag\{)", r"\\t\1", s)
    return s


_PDF_ARROW_LINE = re.compile(r"^[\s\"!?#%'&|<>^~`Hj]{1,24}$")


def clean_pdf_artifacts(text: str) -> str:
    """Убрать мусор от TeX-схем в PDF (стрелки «"», «!», «?», «HHHH j#», «'&», «%») и склеить «Г лава» -> «Глава»."""
    if not text:
        return ""
    out_lines: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        # строки, состоящие только из символов-стрелок псевдографики TeX-схем
        if s and _PDF_ARROW_LINE.match(s):
            continue
        out_lines.append(ln)
    text = "\n".join(out_lines)
    text = re.sub(r"\bГ\s+лава\b", "Глава", text)
    return text


def _merge_split_left_right(text: str) -> str:
    """Склеить формулы, разорванные при импорте: «$y=\\left\\{ $$\\begin{matrix}..\\end{matrix}$$ \\right.$»."""
    pattern = re.compile(
        r"(?<!\$)\$(?!\$)([^$\n]*\\left[^$\n]*?)\s*\$\$\s*(\\begin\{([a-zA-Z*]+)\}[\s\S]*?\\end\{\3\})\s*\$\$\s*(\\right[^$\n]*?)\$(?!\$)"
    )
    return pattern.sub(lambda m: f"\n\n$$\n{m.group(1).strip()}{m.group(2).strip()}{m.group(4).strip()}\n$$\n\n", text)


def format_math_markdown(text: str) -> str:
    """Нормализовать математические формулы LaTeX для корректного рендеринга KaTeX в Streamlit Markdown."""
    if not text:
        return ""

    text = repair_latex_control_chars(text)
    text = _merge_split_left_right(text)

    # 1. Заменяем блочные \[ ... \] на $$ ... $$
    def _repl_display_bracket(m: re.Match) -> str:
        inner = m.group(1).strip()
        return f"\n\n$$\n{inner}\n$$\n\n"

    text = re.sub(r"\\\[\s*(.*?)\s*\\\]", _repl_display_bracket, text, flags=re.DOTALL)

    # 2. Заменяем строчные \( ... \) на $...$ (или $$...$$, если внутри матрица/многострочник)
    def _repl_inline_paren(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return ""
        if "\n" in inner or "\\tag{" in inner or "\\begin{" in inner:
            return f"\n\n$$\n{inner}\n$$\n\n"
        return f"${inner}$"

    text = re.sub(r"\\\(\s*(.*?)\s*\\\)", _repl_inline_paren, text, flags=re.DOTALL)

    # 3. Если внутри одиночных $ ... $ есть окружение \begin{...}...\end{...}, переводим ВСЮ формулу целиком в $$ ... $$
    #    (чтобы A = \begin{pmatrix}...\end{pmatrix} и \Delta = \begin{vmatrix}...\end{vmatrix} = 5 не разрывались!)
    text = re.sub(
        r"(?<!\$)\$(?!\$)([^$]*?\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}[^$]*?)(?<!\$)\$(?!\$)",
        lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n",
        text,
    )

    # 4. Выносим $$ ... $$ в отдельный блок и защищаем их плейсхолдерами, чтобы шаг 6 не рвал матрицы внутри $$ ... $$
    protected_blocks: list[str] = []

    def _stash_display_dollar(m: re.Match) -> str:
        inner = m.group(1).strip()
        if not inner:
            return ""
        idx = len(protected_blocks)
        protected_blocks.append(f"\n\n$$\n{inner}\n$$\n\n")
        return f"\x00MATHBLOCK{idx}\x00"

    text = re.sub(r"\$\$\s*(.*?)\s*\$\$", _stash_display_dollar, text, flags=re.DOTALL)

    # 5. Убираем пробелы сразу после открывающего $ и перед закрывающим $ в однострочных формулах ($ x + y $ -> $x + y$)
    #    и также защищаем их плейсхолдерами
    def _fix_and_stash_inline_line(line: str) -> str:
        if "$" not in line:
            return line

        def _stash_inline(m: re.Match) -> str:
            inner = m.group(1).strip()
            idx = len(protected_blocks)
            protected_blocks.append(f"${inner}$")
            return f"\x00MATHBLOCK{idx}\x00"

        return re.sub(
            r"(?<!\$)\$(?!\$)[ \t]*([^$\n]+?)[ \t]*(?<!\$)\$(?!\$)",
            _stash_inline,
            line,
        )

    lines = [_fix_and_stash_inline_line(ln) for ln in text.splitlines()]
    text = "\n".join(lines)

    # 6. Оборачиваем только действительно «голые» окружения \begin{pmatrix|...}...\end{...}, которые были вне $ и $$
    def _wrap_bare_env(m: re.Match) -> str:
        full = m.group(0).strip()
        return f"\n\n$$\n{full}\n$$\n\n"

    text = re.sub(
        r"\\begin\{(pmatrix|bmatrix|vmatrix|Bmatrix|matrix|cases|aligned|align\*?|array)\}.*?\\end\{\1\}",
        _wrap_bare_env,
        text,
        flags=re.DOTALL,
    )

    # 7. Восстанавливаем защищённые математические блоки
    for idx, block in enumerate(protected_blocks):
        text = text.replace(f"\x00MATHBLOCK{idx}\x00", block)

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_bytes(data: bytes) -> str:
    """Извлечь текст из байтов PDF (с декодированием CP1251 TeX-глифов и формул)."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("для PDF нужен pypdf: python -m pip install pypdf") from exc

    reader = PdfReader(io.BytesIO(data))
    pages = [page.extract_text() or "" for page in reader.pages]
    raw = "\n\n".join(pages).strip()
    return fix_mojibake_and_tex(raw).strip()


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
    return fix_mojibake_and_tex("\n\n".join(paragraphs).strip())


def _resolve_media_url(src: str, base_url: str = "https://dispace.nstu.ru") -> str:
    s = (src or "").strip()
    if not s or s.startswith("data:"):
        return s
    if s.startswith("//"):
        return f"https:{s}"
    if s.startswith("/"):
        return f"{base_url.rstrip('/')}{s}"
    return s


def html_to_text(raw_html: str, base_url: str = "https://dispace.nstu.ru") -> str:
    """Преобразовать HTML лекции DiSpace в Markdown с сохранением LaTeX ($...$), иллюстраций и видео."""
    if not raw_html or not raw_html.strip():
        return ""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(raw_html, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        # Сохраняем иллюстрации <img> как Markdown ![alt](url)
        for img in soup.find_all("img"):
            src = _resolve_media_url(str(img.get("src") or ""), base_url)
            if src and not src.startswith("data:"):
                alt = str(img.get("alt") or img.get("title") or "Иллюстрация").strip()
                img.replace_with(f"\n\n![{alt}]({src})\n\n")

        # Сохраняем встроенные видео <iframe> и <video>
        for iframe in soup.find_all("iframe"):
            src = _resolve_media_url(str(iframe.get("src") or ""), base_url)
            if src:
                title = str(iframe.get("title") or "Видеоматериал").strip()
                iframe.replace_with(f"\n\n🎬 [{title}]({src})\n\n")

        for video in soup.find_all("video"):
            src = _resolve_media_url(str(video.get("src") or ""), base_url)
            if not src:
                source_tag = video.find("source")
                if source_tag:
                    src = _resolve_media_url(str(source_tag.get("src") or ""), base_url)
            if src:
                video.replace_with(f"\n\n🎬 [Видеозапись]({src})\n\n")

        for br in soup.find_all("br"):
            br.replace_with("\n")
        for li in soup.find_all("li"):
            li.insert_before("\n• ")
        for idx_h, h_tag in enumerate(["h1", "h2", "h3", "h4"], start=1):
            for node in soup.find_all(h_tag):
                node.insert_before(f"\n{'#' * min(idx_h + 2, 4)} ")
                node.insert_after("\n")
        for block in soup.find_all(["p", "div", "tr", "blockquote"]):
            block.insert_before("\n")
            block.insert_after("\n")
        text = soup.get_text()
    except ImportError:
        text = re.sub(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>',
            lambda m: f"\n\n![Иллюстрация]({_resolve_media_url(m.group(1), base_url)})\n\n",
            raw_html,
            flags=re.IGNORECASE,
        )
        text = re.sub(
            r'<iframe[^>]+src=["\']([^"\']+)["\'][^>]*>.*?</iframe>',
            lambda m: f"\n\n🎬 [Видеоматериал]({_resolve_media_url(m.group(1), base_url)})\n\n",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
        text = re.sub(r"</(p|div|h[1-6]|li|tr)>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text)

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(lines)
    return fix_mojibake_and_tex(cleaned)


def read_text_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_TEXT:
        raw_bytes = path.read_bytes()
        text = decode_bytes_smart(raw_bytes)
        if suffix in {".html", ".htm"}:
            return html_to_text(text)
        return text

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
