"""Streamlit-интерфейс DIBot.

    streamlit run src/app.py

Вкладки: чат, материалы, конспекты, тренажёр, модели.
Браузерная читалка запускается отдельным процессом — Playwright несовместим
с потоком, в котором Streamlit исполняет скрипт.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402

from llm_client import LLMClient, LLMError  # noqa: E402
from materials import MaterialStore  # noqa: E402
from notes import NoteBuilder  # noqa: E402
from tutor import Tutor  # noqa: E402

st.set_page_config(page_title="DIBot — учебный ассистент", page_icon="🎓", layout="wide")


# --------------------------------------------------------------- ресурсы


@st.cache_resource(show_spinner=False)
def get_tutor() -> Tutor:
    return Tutor()


@st.cache_resource(show_spinner=False)
def get_store() -> MaterialStore:
    return MaterialStore()


def ollama_status() -> dict:
    """Быстрый статус сервера и загруженных моделей (без запуска модели)."""
    try:
        tags = requests.get("http://localhost:11434/api/tags", timeout=4).json()
        ps = requests.get("http://localhost:11434/api/ps", timeout=4).json()
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc), "models": [], "loaded": []}
    loaded = [
        {
            "name": m.get("name"),
            "size_gb": round(m.get("size", 0) / 1024**3, 1),
            "processor": m.get("details", {}).get("processor", "—"),
            "context": m.get("context_length") or m.get("context"),
        }
        for m in (ps.get("models") or [])
    ]
    return {
        "ok": True,
        "models": [m["name"] for m in tags.get("models", [])],
        "loaded": loaded,
    }


def render_verification(report) -> None:
    colors = {"confirmed": "✅", "disputed": "⛔", "unverified": "⚠️"}
    icon = colors.get(report.verdict, "•")
    with st.expander(f"{icon} Проверка решения — {report.verdict} ({report.confidence:.0%})", expanded=report.verdict != "confirmed"):
        for check in report.checks:
            mark = {"pass": "✅", "fail": "❌", "skipped": "➖", "error": "⚠️"}.get(check.status, "•")
            st.markdown(f"{mark} **{check.layer}** — {check.detail}")


# ------------------------------------------------------------------ шапка

status = ollama_status()
with st.sidebar:
    st.title("🎓 DIBot")
    st.caption("Учебный ассистент: конспекты, разбор задач, самопроверка")

    if status["ok"]:
        st.success(f"Ollama: {len(status['models'])} моделей")
    else:
        st.error("Ollama не отвечает")
        st.caption(status.get("error", ""))

    if status["loaded"]:
        st.markdown("**В памяти:**")
        for item in status["loaded"]:
            st.caption(
                f"{item['name']} · {item['size_gb']} ГБ · {item['processor'] or '—'}"
            )

    st.divider()
    st.caption("Модели выбираются автоматически по типу задачи.")

tabs = st.tabs(["💬 Чат", "📚 Материалы", "📝 Конспекты", "🧪 Тренажёр", "⚙️ Модели"])

# -------------------------------------------------------------------- чат

with tabs[0]:
    st.subheader("Спросить по учёбе")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.form("ask_form", clear_on_submit=False):
        question = st.text_area(
            "Вопрос или задача",
            height=110,
            placeholder="Например: реши систему 2x + y = 5, x + 3y = 10 методом Крамера",
        )
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            use_web = st.toggle("Веб-поиск", value=False)
        with col2:
            do_verify = st.toggle("Проверять решение", value=True)
        with col3:
            courses = ["все"] + get_store().list_courses()
            course_filter = st.selectbox("Курс", courses)

        submitted = st.form_submit_button("Решить", type="primary", use_container_width=True)

    if submitted and question.strip():
        tutor = get_tutor()
        with st.spinner("Думаю…"):
            try:
                answer = tutor.ask(
                    question.strip(),
                    course=None if course_filter == "все" else course_filter,
                    use_web=use_web,
                    verify=do_verify,
                )
            except LLMError as exc:
                st.error(f"Модель недоступна: {exc}")
            else:
                st.session_state.messages.insert(0, answer)

    for answer in st.session_state.messages[:6]:
        with st.container(border=True):
            st.caption(f"❓ {answer.question}")
            st.markdown(answer.text)
            meta = (
                f"модель: `{answer.model}` · задача: `{answer.task}` · "
                f"контекст: {answer.context_used} · {answer.seconds:.0f} с"
            )
            st.caption(meta)
            if answer.sources:
                st.caption("Источники: " + "; ".join(answer.sources[:4]))
            if answer.verification:
                render_verification(answer.verification)

    if st.session_state.messages and st.button("Очистить историю"):
        st.session_state.messages = []
        st.rerun()

# -------------------------------------------------------------- материалы

with tabs[1]:
    st.subheader("Материалы для обучения")
    store = get_store()
    stats = store.stats()

    left, right = st.columns([2, 3])
    with left:
        st.metric("Источников", stats["sources"])
        st.metric("Фрагментов", stats["chunks"])
        st.metric("Символов", f"{stats['chars']:,}".replace(",", " "))
    with right:
        st.markdown("**Добавить материал**")
        with st.form("add_material"):
            course = st.text_input("Курс", placeholder="Линейная алгебра")
            title = st.text_input("Название", placeholder="Лекция 1. Матрицы")
            url = st.text_input("Ссылка (необязательно)", placeholder="https://dispace.nstu.ru/...")
            uploaded = st.file_uploader("Файл .txt / .md / .pdf", type=["txt", "md", "markdown", "pdf"])
            pasted = st.text_area("…или вставь текст", height=120)
            add = st.form_submit_button("Сохранить", type="primary")

        if add:
            if not course or not (uploaded or pasted.strip()):
                st.warning("Нужны название курса и текст (файл или вставка).")
            else:
                text = pasted
                if uploaded:
                    if uploaded.name.lower().endswith(".pdf"):
                        try:
                            from pypdf import PdfReader

                            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                                tmp.write(uploaded.getbuffer())
                                tmp_path = Path(tmp.name)
                            text = "\n\n".join(
                                (page.extract_text() or "") for page in PdfReader(str(tmp_path)).pages
                            )
                            tmp_path.unlink(missing_ok=True)
                        except ImportError:
                            st.error("Для PDF нужен pypdf: pip install pypdf")
                            text = ""
                        except Exception as exc:
                            st.error(f"Не удалось прочитать PDF: {exc}")
                            text = ""
                    else:
                        text = uploaded.getvalue().decode("utf-8", errors="replace")

                if text.strip():
                    source = store.add_source(
                        course=course.strip(),
                        title=(title or uploaded.name if uploaded else title or "Без названия").strip(),
                        text=text,
                        url=url.strip(),
                    )
                    st.success(
                        f"Сохранено: {source.title} — {len(source.chunk_ids)} фрагментов"
                    )
                    get_tutor.clear()
                    st.rerun()

    st.divider()
    st.markdown("**Что уже загружено**")
    if not store.sources:
        st.info("Пока пусто. Добавь лекцию выше или импортируй страницу диспейса.")
    for course_name in store.list_courses():
        with st.expander(f"{course_name} — {len(store.list_sources(course_name))} источников"):
            for source in store.list_sources(course_name):
                cols = st.columns([5, 1])
                with cols[0]:
                    st.markdown(f"**{source.title}** — {source.chars} симв., {len(source.chunk_ids)} фр.")
                    if source.url:
                        st.caption(source.url)
                with cols[1]:
                    if st.button("Удалить", key=f"del_{source.id}"):
                        store.remove_source(source.id)
                        get_tutor.clear()
                        st.rerun()

    st.divider()
    st.markdown("**Импорт из диспейса**")
    st.caption(
        "Читалка работает в отдельном окне браузера: ты входишь сам, пароль не сохраняется. "
        "Скрипт только читает страницы."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("1. Войти в диспейс"):
            subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "src" / "lms_reader.py"), "--login"],
                cwd=str(PROJECT_ROOT),
            )
            st.info("Открылось окно браузера. Войди в нём, затем вернись сюда.")
    with col_b:
        with st.form("import_page"):
            page_url = st.text_input("Ссылка на страницу курса")
            page_course = st.text_input("Курс для сохранения")
            run_import = st.form_submit_button("2. Импортировать страницу")
        if run_import and page_url and page_course:
            with st.spinner("Читаю страницу…"):
                proc = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "src" / "lms_reader.py"),
                        "--import",
                        page_url,
                        "--course",
                        page_course,
                    ],
                    cwd=str(PROJECT_ROOT),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=180,
                )
            st.code((proc.stdout or "") + (proc.stderr or ""))
            get_tutor.clear()

# --------------------------------------------------------------- конспекты

with tabs[2]:
    st.subheader("Конспекты")
    store = get_store()
    if not store.sources:
        st.info("Сначала добавь материалы на вкладке «Материалы».")
    else:
        course_names = store.list_courses()
        selected_course = st.selectbox("Курс", course_names, key="notes_course")
        sources = store.list_sources(selected_course)
        selected = st.selectbox(
            "Материал", sources, format_func=lambda s: s.title, key="notes_source"
        )

        col1, col2 = st.columns([1, 3])
        with col1:
            rebuild = st.button("Собрать конспект", type="primary")
        with col2:
            st.caption(
                f"{selected.chars} символов, {len(selected.chunk_ids)} фрагментов. "
                "Каждый тезис получит ссылку на фрагмент источника."
            )

        notes_dir = store.notes_dir
        from materials import slugify

        note_path = notes_dir / slugify(selected_course) / f"{slugify(selected.title)}.md"

        if rebuild:
            builder = NoteBuilder(get_tutor().llm, store)
            progress = st.progress(0.0, text="Готовлю фрагменты…")
            total_holder = {"total": len(selected.chunk_ids)}

            def on_progress(position: int, total: int, chunk) -> None:
                progress.progress(position / max(total, 1), text=f"Фрагмент {position} из {total}")

            try:
                result = builder.build(selected, on_progress=on_progress)
            except (LLMError, ValueError) as exc:
                st.error(f"Не удалось собрать конспект: {exc}")
            else:
                progress.progress(1.0, text="Готово")
                st.success(result.summary_line())
                st.rerun()

        if note_path.exists():
            st.download_button(
                "Скачать .md",
                data=note_path.read_text(encoding="utf-8"),
                file_name=note_path.name,
                mime="text/markdown",
            )
            st.markdown(note_path.read_text(encoding="utf-8"))
        else:
            st.info("Конспект ещё не собран.")

# --------------------------------------------------------------- тренажёр

with tabs[3]:
    st.subheader("Тренажёр: тренировочные задания")
    store = get_store()

    with st.form("practice_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            topic = st.text_input("Тема", placeholder="Определители и метод Крамера")
        with col2:
            count = st.slider("Заданий", 1, 8, 3)
        with col3:
            difficulty = st.selectbox("Уровень", ["лёгкий", "средний", "сложный"], index=1)
        verify_all = st.toggle(
            "Проверить эталонные решения", value=True,
            help="sympy пересчитает ответ, вторая модель проверит ход решения",
        )
        generate = st.form_submit_button("Сгенерировать вариант", type="primary")

    if generate and topic.strip():
        with st.spinner("Составляю задания и проверяю эталоны…"):
            try:
                tasks, reports = get_tutor().practice(
                    topic=topic.strip(), count=count, difficulty=difficulty, verify=verify_all
                )
            except LLMError as exc:
                st.error(f"Модель недоступна: {exc}")
                tasks, reports = [], []
        st.session_state.practice = (topic, tasks, reports, difficulty)

    if st.session_state.get("practice"):
        topic, tasks, reports, difficulty = st.session_state.practice
        st.caption(f"Тема: **{topic}** · уровень: {difficulty} · заданий: {len(tasks)}")
        if not tasks:
            st.warning("Задания не распознаны в ответе модели. Попробуй ещё раз.")
        for i, task in enumerate(tasks, start=1):
            report = reports[i - 1] if i - 1 < len(reports) else None
            badge = ""
            if report:
                badge = {"confirmed": " ✅ эталон проверен", "disputed": " ⛔ эталон спорный",
                         "unverified": " ⚠️ эталон не проверен"}[report.verdict]
            with st.container(border=True):
                st.markdown(f"**Задание {i}.** {task['statement']}")
                st.caption(f"Уровень: {task.get('difficulty') or difficulty}{badge}")
                with st.expander("Показать ответ"):
                    st.markdown(f"**Ответ:** {task.get('answer') or '—'}")
                    if task.get("solution"):
                        st.markdown(task["solution"])
                    if report:
                        for check in report.checks:
                            mark = {"pass": "✅", "fail": "❌", "skipped": "➖", "error": "⚠️"}.get(
                                check.status, "•"
                            )
                            st.caption(f"{mark} {check.layer}: {check.detail}")

        if tasks:
            as_md = "\n\n".join(
                f"**{i}.** {t['statement']}\n\nОтвет: {t.get('answer', '')}" for i, t in enumerate(tasks, 1)
            )
            st.download_button("Скачать вариант без решений", as_md, file_name="вариант.md")

# ------------------------------------------------------------------ модели

with tabs[4]:
    st.subheader("Модели и маршрутизация")
    llm = get_tutor().llm

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Маршруты задач**")
        rows = []
        for task in llm.router.routing:
            route = llm.router.route(task)
            rows.append(
                {
                    "задача": task,
                    "модель": route.model,
                    "размышления": "да" if route.think else "нет",
                    "температура": route.temperature,
                    "контекст": route.num_ctx,
                    "запасные": ", ".join(route.fallback) or "—",
                }
            )
        st.dataframe(rows, use_container_width=True, hide_index=True)

        st.markdown("**Независимость проверки**")
        for model in llm.router.models:
            verifier = llm.router.verifier(model)
            ok = verifier and verifier != model
            st.caption(f"{'✅' if ok else '⚠️'} решает `{model}` → проверяет `{verifier or '—'}`")

    with col2:
        st.markdown("**Железо**")
        st.caption("RTX 4060 8 ГБ · 16 ГБ RAM. В VRAM помещается одна 8B-модель.")
        st.caption(
            "Контекст ограничен 8192: при 16384 модель не влезает в 8 ГБ "
            "и 20% слоёв уходят на CPU — скорость падает в 8 раз (5.5 против 26.8 ток/с)."
        )
        if status["loaded"]:
            st.markdown("**Сейчас в памяти**")
            for item in status["loaded"]:
                st.caption(f"{item['name']} · {item['size_gb']} ГБ · {item['processor']}")

    st.divider()
    if st.button("Полная диагностика"):
        with st.spinner("Проверяю…"):
            report = llm.health()
        st.code(report.as_text())
