"""Streamlit-интерфейс DIBot.

    streamlit run src/app.py

Вкладки:
  1. 💬 Чат — полноценный многошаговый диалог с прикреплением фото (qwen3-vl),
     поиском по базе лекций (RAG), решением задач (deepseek-r1 / qwen2.5-coder) и SymPy-проверкой.
  2. 🎓 DiSpace НГТУ — все 6 разделов личного кабинета НГТУ (Мои курсы, Дисциплины и журнал,
     События и вебинары, Каталог курсов, Тесты и контрольные, Сообщения) с прямым импортом в базу.
  3. 📚 Материалы — управление локальной базой знаний, предпросмотр текста, ручная загрузка PDF/DOCX/MD.
  4. 📝 Конспекты — генерация проверяемых конспектов со ссылками [#N] на фрагменты лекций.
  5. 🧪 Тренажёр — генерация тренировочных вариантов с автоматической проверкой эталонных решений.
  6. ⚙️ Модели — мониторинг Ollama, маршрутизация задач и диагностика.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ingest as _ingest_mod  # noqa: E402
import llm_client as _llm_client_mod  # noqa: E402
import lms_reader as _lms_reader_mod  # noqa: E402
import materials as _materials_mod  # noqa: E402
import tutor as _tutor_mod  # noqa: E402

# Автоперезагрузка модулей, если сервер Streamlit был запущен до обновления кода
if (
    not hasattr(_materials_mod.MaterialStore, "reload")
    or not hasattr(_lms_reader_mod.DiSpaceAPI, "import_course_to_store")
    or not hasattr(_lms_reader_mod, "authenticate_dispace_input")
    or not hasattr(_ingest_mod, "decode_bytes_smart")
):
    importlib.reload(_ingest_mod)
    importlib.reload(_materials_mod)
    importlib.reload(_llm_client_mod)
    importlib.reload(_lms_reader_mod)
    importlib.reload(_tutor_mod)

from ingest import (  # noqa: E402
    decode_bytes_smart,
    extract_docx_bytes,
    extract_pdf_bytes,
    format_math_markdown,
    html_to_text,
)

from llm_client import LLMClient, LLMError  # noqa: E402
from lms_reader import (  # noqa: E402
    CATALOG_URL,
    EDUCATION_URL,
    EVENTS_URL,
    MESSAGES_URL,
    MY_COURSES_URL,
    TESTS_URL,
    LoginRequired,
    authenticate_dispace_input,
    get_api_client,
    get_keycloak_login_url,
    login_dispace_direct,
    parse_dispace_target,
)
from materials import MaterialStore, slugify  # noqa: E402
from notes import NoteBuilder  # noqa: E402
from tutor import Answer, Tutor  # noqa: E402

st.set_page_config(
    page_title="DIBot — учебный ассистент НГТУ",
    page_icon="🎓",
    layout="wide",
)


def _push_git_info_to_client() -> None:
    """Отправить метаданные GitHub-репозитория клиенту сразу при загрузке страницы.

    В Streamlit на Windows возникает гонка: при первом клике на кнопку Deploy диалог
    открывается раньше, чем приходит ответ на асинхронный loadGitInfo, из-за чего
    показывается «Unable to deploy». Предварительная отправка git_info_changed устраняет это.
    """
    try:
        from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
        from streamlit.proto.GitInfo_pb2 import GitInfo
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        if ctx is None:
            return
        msg = ForwardMsg()
        msg.git_info_changed.repository = "sosiskacorp/DIBot"
        msg.git_info_changed.branch = "main"
        msg.git_info_changed.module = "streamlit_app.py"
        msg.git_info_changed.state = GitInfo.GitStates.DEFAULT
        ctx.enqueue(msg)
    except Exception:
        pass


_push_git_info_to_client()


# --------------------------------------------------------------- ресурсы и изоляция сессий

# Определяем, запущено ли приложение локально на домашнем ПК пользователя
IS_LOCAL_PC = (
    (PROJECT_ROOT / "data" / "browser_state.json").exists()
    or (PROJECT_ROOT / "data" / "gateway_token.txt").exists()
)


def _default_gateway_url() -> str:
    """Публичный URL туннеля можно подставить по умолчанию (сам по себе без секретного ключа он закрыт)."""
    try:
        if "OLLAMA_BASE_URL" in st.secrets:
            return str(st.secrets["OLLAMA_BASE_URL"]).strip().rstrip("/")
    except Exception:
        pass
    return "http://localhost:11434" if IS_LOCAL_PC else ""


def _init_session_credentials() -> None:
    """Хранить все ключи СТРОГО в st.session_state текущей вкладки/устройства, а НЕ в глобальном os.environ!"""
    if "session_ollama_url" not in st.session_state:
        st.session_state["session_ollama_url"] = _default_gateway_url()
    if "session_ollama_key" not in st.session_state:
        # На локальном ПК ключ не нужен, а в облаке каждое устройство ОБЯЗАНО ввести ключ в своей сессии!
        st.session_state["session_ollama_key"] = ""
    if "session_cloud_key" not in st.session_state:
        st.session_state["session_cloud_key"] = ""
    if "session_cloud_url" not in st.session_state:
        st.session_state["session_cloud_url"] = "https://api.groq.com/openai/v1"
    if "session_cloud_model" not in st.session_state:
        st.session_state["session_cloud_model"] = "llama-3.3-70b-versatile"
    if "session_dispace_token" not in st.session_state:
        st.session_state["session_dispace_token"] = ""


_init_session_credentials()


@st.cache_resource(show_spinner=False)
def get_store() -> MaterialStore:
    return MaterialStore()


def get_tutor() -> Tutor:
    """Создать Tutor строго с ключами текущей пользовательской сессии (без кэширования между устройствами!)."""
    base_url = st.session_state.get("session_ollama_url") or (
        "http://localhost:11434" if IS_LOCAL_PC else "https://127.0.0.1:11435"
    )
    llm = LLMClient(
        base_url=base_url,
        ollama_api_key=st.session_state.get("session_ollama_key", ""),
        cloud_api_key=st.session_state.get("session_cloud_key", ""),
        cloud_base_url=st.session_state.get("session_cloud_url", ""),
        cloud_model=st.session_state.get("session_cloud_model", ""),
    )
    return Tutor(llm=llm, store=get_store())


get_tutor.clear = lambda: None  # type: ignore[attr-defined]


def get_session_dispace_api(auto_refresh: bool = False):
    """Получить клиент DiSpace для текущей сессии браузера (независимо от токена локальной модели)."""
    return get_api_client(
        auto_refresh=auto_refresh and IS_LOCAL_PC,
        explicit_token=st.session_state.get("session_dispace_token", ""),
        allow_local_files=IS_LOCAL_PC,
    )


@st.dialog("🔑 Вход в DiSpace НГТУ через браузер")
def open_dispace_browser_login_dialog(is_cloud_env: bool) -> None:
    """Окно авторизации в DiSpace через браузер для любого устройства (облако и локальный ПК)."""
    sso_url = get_keycloak_login_url()

    tab_sso, tab_direct = st.tabs(
        ["🌐 Вход через браузер (SSO НГТУ)", "🔐 Вход по логину DiSpace"]
    )

    with tab_sso:
        if not is_cloud_env:
            st.markdown("**Автоматический вход через браузер на этом ПК:**")
            if st.button("🖥️ Открыть окно браузера и войти автоматически", type="primary", width="stretch"):
                subprocess.Popen(
                    [sys.executable, str(PROJECT_ROOT / "src" / "lms_reader.py"), "--login"],
                    cwd=str(PROJECT_ROOT),
                )
                st.info("Открылось окно браузера. Войди в DiSpace, затем нажми «🔄 Обновить сессию».")
            st.divider()

        st.markdown("**Шаг 1.** Открой страницу единого входа НГТУ в новой вкладке браузера и войди в аккаунт:")
        st.link_button(
            "🌐 1. Открыть вход DiSpace НГТУ в браузере",
            sso_url,
            type="primary" if is_cloud_env else "secondary",
            use_container_width=True,
        )

        st.markdown(
            "**Шаг 2.** Если ты уже вошёл в DiSpace в браузере, при повторном нажатии кнопки выше в адресной строке "
            "появится ссылка со строкой `?session_state=...&code=...` — просто скопируй этот URL из адресной строки "
            "(либо выполни в консоли `F12` на вкладке DiSpace команду `copy(localStorage.authToken)`) и вставь сюда:"
        )
        browser_input = st.text_input(
            "Ссылка после входа в браузере (?code=...) или authToken",
            placeholder="https://dispace.nstu.ru/?session_state=...&code=... или eyJ...",
        )
        if st.button("✅ Завершить вход в этом браузере", type="primary", width="stretch"):
            with st.spinner("Проверяю сессию в DiSpace НГТУ…"):
                tok, err = authenticate_dispace_input(browser_input)
            if tok:
                st.session_state["session_dispace_token"] = tok
                st.success("Вход в DiSpace выполнен! Сессия сохранена для этого браузера.")
                st.rerun()
            else:
                st.error(err)

    with tab_direct:
        st.caption(
            "Прямой вход через форму DiSpace (`/api/v1/auth/login`). "
            "Пароль нигде не сохраняется — в сессии текущего браузера хранится только токен."
        )
        d_user = st.text_input("Логин / e-mail DiSpace")
        d_pass = st.text_input("Пароль", type="password")
        if st.button("Войти в DiSpace", type="primary", width="stretch"):
            with st.spinner("Выполняю вход в DiSpace…"):
                tok, err = login_dispace_direct(d_user, d_pass)
            if tok:
                st.session_state["session_dispace_token"] = tok
                st.success("Вход в DiSpace выполнен!")
                st.rerun()
            else:
                st.error(err)


def sync_materials() -> MaterialStore:
    """Перечитать индекс материалов с диска, чтобы импортированные курсы сразу появлялись в UI."""
    store = get_store()
    if not hasattr(store, "reload") or not hasattr(store, "get_source_text"):
        importlib.reload(_materials_mod)
        importlib.reload(_llm_client_mod)
        importlib.reload(_lms_reader_mod)
        importlib.reload(_tutor_mod)
        get_store.clear()
        store = get_store()
    if hasattr(store, "reload"):
        store.reload()
    elif hasattr(store, "_load"):
        store._load()
    return store


def ollama_status() -> dict[str, Any]:
    """Быстрый статус сервера Ollama строго для текущей сессии устройства."""
    base_url = (st.session_state.get("session_ollama_url") or "").strip().rstrip("/")
    api_key = (st.session_state.get("session_ollama_key") or "").strip()

    if not IS_LOCAL_PC and not api_key:
        return {
            "ok": False,
            "base_url": base_url,
            "error": "Требуется ввод секретного токена на этом устройстве",
            "models": [],
            "loaded": [],
        }
    if not base_url:
        base_url = "http://localhost:11434" if IS_LOCAL_PC else ""
    if not base_url:
        return {"ok": False, "base_url": "", "error": "Не указан URL шлюза", "models": [], "loaded": []}

    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-DIBot-Token"] = api_key

    try:
        tags_resp = requests.get(f"{base_url}/api/tags", headers=headers, timeout=4)
        tags_resp.raise_for_status()
        tags = tags_resp.json()
        ps_resp = requests.get(f"{base_url}/api/ps", headers=headers, timeout=4)
        ps = ps_resp.json() if ps_resp.ok else {}
    except requests.RequestException as exc:
        return {"ok": False, "base_url": base_url, "error": str(exc), "models": [], "loaded": []}
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
        "base_url": base_url,
        "is_remote": "localhost" not in base_url and "127.0.0.1" not in base_url,
        "models": [m["name"] for m in tags.get("models", []) if "name" in m],
        "loaded": loaded,
    }


def _fmt_cell(val: Any) -> str:
    """Единообразно форматировать значение ячейки таблицы в строку, чтобы избежать ArrowTypeError в pyarrow."""
    if val is None or val == "":
        return "—"
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def render_video_url(url: str, title: str = "") -> None:
    """Отрисовать встроенный видеоплеер для прямых видеофайлов, YouTube, Kinescope, Rutube, VK Video."""
    u = (url or "").strip()
    if not u:
        return
    lower = u.lower().split("?", 1)[0]
    if lower.endswith((".mp4", ".webm", ".mov", ".ogg")) or "youtube.com/watch" in u.lower() or "youtu.be/" in u.lower():
        try:
            st.video(u)
            return
        except Exception:
            pass
    if any(host in u.lower() for host in ("kinescope.io/", "rutube.ru/play/embed", "vk.com/video_ext")):
        try:
            st.components.v1.iframe(u, height=400, scrolling=False)
            return
        except Exception:
            pass
    if title:
        st.markdown(f"🎬 **[{title}]({u})**")


def render_rich_content(text: str) -> None:
    """Отрисовать текст с нормализованными формулами LaTeX ($...$, $$...$$), фото и встроенными видеоплеерами."""
    if not text:
        return
    import re as _re

    formatted = format_math_markdown(text)
    st.markdown(formatted)

    # Если в тексте встречаются ссылки на видео (🎬 [Название](url) или прямые ссылки на kinescope/mp4/youtube),
    # отобразим встроенный видеоплеер под блоком
    video_matches = _re.findall(r"🎬\s*\[([^\]]*)\]\((https?://[^\s)]+)\)", formatted)
    seen_vids: set[str] = set()
    for v_title, v_url in video_matches:
        if v_url not in seen_vids:
            seen_vids.add(v_url)
            render_video_url(v_url, v_title)


def render_verification(report) -> None:
    colors = {"confirmed": "✅", "disputed": "⛔", "unverified": "⚠️"}
    labels = {
        "confirmed": "подтверждено",
        "disputed": "обнаружено расхождение",
        "unverified": "не проверено",
    }
    icon = colors.get(report.verdict, "•")
    label = labels.get(report.verdict, report.verdict)
    with st.expander(
        f"{icon} Проверка решения — {label} ({report.confidence:.0%})",
        expanded=report.verdict == "disputed",
    ):
        for check in report.checks:
            mark = {"pass": "✅", "fail": "❌", "skipped": "➖", "error": "⚠️"}.get(
                check.status, "•"
            )
            st.markdown(f"{mark} **{check.layer}** — {check.detail}")


def render_assistant_turn(turn: dict[str, Any]) -> None:
    """Отрисовать сообщение ассистента со всеми деталями (фото-OCR, формулы, видео, ход мыслей, проверка)."""
    ans: Answer | None = turn.get("answer")
    if ans is None:
        render_rich_content(turn.get("content", ""))
        return

    if ans.extracted_image_text:
        with st.expander("📷 Распознано с прикреплённого фото (qwen3-vl)", expanded=False):
            render_rich_content(ans.extracted_image_text)

    if ans.thinking:
        with st.expander("💭 Ход рассуждений модели", expanded=False):
            st.caption(ans.thinking)

    render_rich_content(ans.text)

    badges = [
        f"модель: `{ans.model}`",
        f"задача: `{ans.task}`",
        f"контекст: **{ans.context_used}**",
        f"время: {ans.seconds:.1f} с",
    ]
    st.caption(" · ".join(badges))

    if ans.sources:
        st.caption("📚 Источники: " + " · ".join(ans.sources[:5]))

    if ans.verification:
        render_verification(ans.verification)


# ------------------------------------------------------------------ сайдбар

store = sync_materials()
stats = store.stats()
status = ollama_status()
dispace_api = get_session_dispace_api(auto_refresh=False)

with st.sidebar:
    st.title("🎓 DIBot")
    st.caption("Учебный ИИ-ассистент студента НГТУ")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Курсов в базе", len(stats["courses"]))
    with col_s2:
        st.metric("Фрагментов", stats["chunks"])

    if status["ok"]:
        conn_label = "Удалённый локальный Ollama" if status.get("is_remote") else "Локальный Ollama"
        st.success(f"{conn_label}: {len(status['models'])} моделей готово")
    elif st.session_state.get("session_cloud_key"):
        st.success(f"Облачный ИИ: {st.session_state.get('session_cloud_model', 'llama-3.3-70b-versatile')}")
    else:
        st.warning("🔒 Локальная модель не подключена (введи токен доступа ниже)")

    with st.expander("🔒 Подключение к локальной модели (для этого устройства)", expanded=not status["ok"]):
        st.caption(
            "Ключ сохраняется **только в памяти текущей вкладки браузера** (`st.session_state`) "
            "и недоступен другим устройствам."
        )
        remote_ollama_url = st.text_input(
            "HTTPS URL шлюза (OLLAMA_BASE_URL)",
            value=st.session_state.get("session_ollama_url", ""),
            placeholder="https://xxxx-xxxx.trycloudflare.com",
        )
        remote_ollama_key = st.text_input(
            "Секретный токен (OLLAMA_API_KEY)",
            value=st.session_state.get("session_ollama_key", ""),
            type="password",
            placeholder="dibot_sk_...",
        )
        if st.button("Подключить на этом устройстве", type="primary", width="stretch"):
            st.session_state["session_ollama_url"] = remote_ollama_url.strip().rstrip("/")
            st.session_state["session_ollama_key"] = remote_ollama_key.strip()
            st.rerun()

        if st.session_state.get("session_ollama_key") or st.session_state.get("session_cloud_key"):
            if st.button("🔒 Отключить модель на этом устройстве", width="stretch"):
                st.session_state["session_ollama_key"] = ""
                st.session_state["session_cloud_key"] = ""
                st.rerun()

        st.divider()
        st.caption("Резервный облачный API (если локальный ПК выключен):")
        cloud_key = st.text_input(
            "API-ключ (Groq / OpenRouter)",
            value=st.session_state.get("session_cloud_key", ""),
            type="password",
            placeholder="gsk_... или sk-or-...",
        )
        cloud_url = st.text_input(
            "Base URL",
            value=st.session_state.get("session_cloud_url", "https://api.groq.com/openai/v1"),
        )
        cloud_mod = st.text_input(
            "Модель",
            value=st.session_state.get("session_cloud_model", "llama-3.3-70b-versatile"),
        )
        if st.button("Сохранить резервный API", width="stretch") and cloud_key.strip():
            st.session_state["session_cloud_key"] = cloud_key.strip()
            st.session_state["session_cloud_url"] = cloud_url.strip()
            st.session_state["session_cloud_model"] = cloud_mod.strip()
            st.rerun()

    if status["loaded"]:
        for item in status["loaded"]:
            st.caption(f"⚡ В памяти: `{item['name']}` ({item['size_gb']} ГБ)")

    if dispace_api:
        st.success("DiSpace НГТУ: сессия активна")
    else:
        st.warning("DiSpace НГТУ: нужен вход через браузер")

    st.divider()
    st.markdown("**Параметры чата**")
    course_options = ["Все курсы"] + store.list_courses()
    selected_chat_course = st.selectbox(
        "Фильтр по курсу",
        course_options,
        help="Ограничить поиск по материалам выбранной дисциплиной",
    )
    task_mode_options = {
        "Авто (по запросу)": None,
        "Математика (deepseek-r1)": "math",
        "Диалог / теория (qwen3.5)": "chat",
        "Код / Python (qwen2.5-coder)": "code",
        "LaTeX-оформление": "latex",
        "Только зрение (qwen3-vl)": "vision",
    }
    selected_task_label = st.selectbox("Режим модели", list(task_mode_options.keys()))
    forced_task = task_mode_options[selected_task_label]

    use_web = st.toggle("🌐 Веб-поиск (SearXNG)", value=False)
    do_verify = st.toggle("✅ Проверять решения (SymPy + LLM)", value=True)

    if st.button("🗑️ Очистить диалог", width="stretch"):
        st.session_state.chat_turns = []
        st.rerun()


tabs = st.tabs(
    [
        "💬 Чат",
        "🎓 DiSpace НГТУ",
        "📚 Материалы",
        "📝 Конспекты",
        "🧪 Тренажёр",
        "⚙️ Модели",
    ]
)


# ===================================================================== 1. ЧАТ

with tabs[0]:
    if "chat_turns" not in st.session_state:
        st.session_state.chat_turns = []

    # Приветственный экран и быстрые подсказки, когда чат пуст
    if not st.session_state.chat_turns:
        st.markdown("### 💬 Учебный чат с поддержкой фото, видео, формул и файлов")
        st.caption(
            "Задай вопрос по лекциям, отправь условие задачи или **прикрепи фото/видео/документ** "
            "прямо в поле ввода внизу (иконка скрепки). Фото распознаётся моделью `qwen3-vl`, "
            "математические формулы рендерятся через KaTeX, а расчётные задачи проверяются в `SymPy`."
        )
        suggestions = [
            "Реши систему 2x + y = 5, x + 3y = 10 методом Крамера и проверь ответ",
            "Что такое предел числовой последовательности по материалам курса?",
            "Как в линейной алгебре определяются коллинеарные и компланарные векторы?",
            "Какие задания входят в контрольную работу по линейной алгебре?",
        ]
        picked_suggestion = st.pills(
            "Примеры запросов:",
            suggestions,
            selection_mode="single",
            label_visibility="collapsed",
        )
    else:
        picked_suggestion = None

    # Отрисовка истории сообщений
    chat_container = st.container()
    with chat_container:
        for turn in st.session_state.chat_turns:
            with st.chat_message(turn["role"]):
                if turn["role"] == "user":
                    if turn.get("images"):
                        img_cols = st.columns(min(len(turn["images"]), 3))
                        for idx, img_bytes in enumerate(turn["images"]):
                            with img_cols[idx % len(img_cols)]:
                                st.image(img_bytes, width=280)
                    if turn.get("videos"):
                        for vid_bytes in turn["videos"]:
                            st.video(vid_bytes)
                    if turn.get("content"):
                        render_rich_content(turn["content"])
                else:
                    render_assistant_turn(turn)

    # Поле ввода чата с прикреплением фото, видео и документов в любых кодировках
    chat_submission = st.chat_input(
        "Напиши вопрос, задачу или прикрепи фото, видео, PDF/DOCX/TXT…",
        accept_file="multiple",
        file_type=[
            "png", "jpg", "jpeg", "webp", "gif", "bmp",
            "mp4", "webm", "mov",
            "pdf", "docx", "txt", "md", "tex", "csv", "html",
        ],
    )

    user_text = ""
    attached_images: list[bytes] = []
    attached_videos: list[bytes] = []
    attached_docs_text: list[str] = []

    if chat_submission is not None:
        if isinstance(chat_submission, str):
            user_text = chat_submission.strip()
        else:
            user_text = (getattr(chat_submission, "text", "") or "").strip()
            for f_obj in getattr(chat_submission, "files", None) or []:
                fname = str(getattr(f_obj, "name", "") or "").lower()
                raw_data = bytes(f_obj.getvalue())
                if fname.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                    attached_images.append(raw_data)
                elif fname.endswith((".mp4", ".webm", ".mov")):
                    attached_videos.append(raw_data)
                elif fname.endswith(".pdf"):
                    try:
                        attached_docs_text.append(f"--- Файл {f_obj.name} ---\n{extract_pdf_bytes(raw_data)[:6000]}")
                    except Exception:
                        pass
                elif fname.endswith(".docx"):
                    try:
                        attached_docs_text.append(f"--- Файл {f_obj.name} ---\n{extract_docx_bytes(raw_data)[:6000]}")
                    except Exception:
                        pass
                else:
                    decoded_doc = decode_bytes_smart(raw_data)
                    if fname.endswith((".html", ".htm")):
                        decoded_doc = html_to_text(decoded_doc)
                    if decoded_doc.strip():
                        attached_docs_text.append(f"--- Файл {f_obj.name} ---\n{decoded_doc[:6000]}")
    elif picked_suggestion:
        user_text = picked_suggestion

    if attached_docs_text:
        docs_joined = "\n\n".join(attached_docs_text)
        user_text = f"{user_text}\n\nПрикреплённый текст:\n{docs_joined}".strip()

    if user_text or attached_images or attached_videos:
        display_text = (
            user_text
            or ("📷 Разбери и реши задание на прикреплённом фото." if attached_images else "🎬 Прикреплено видео.")
        )

        # Собираем историю предыдущих реплик для многошагового контекста
        history_messages: list[dict[str, str]] = []
        for prev in st.session_state.chat_turns[-8:]:
            role = prev["role"]
            if role == "user":
                content = prev.get("content") or ""
            else:
                ans_obj: Answer | None = prev.get("answer")
                content = ans_obj.text if ans_obj else (prev.get("content") or "")
            if content:
                history_messages.append({"role": role, "content": content})

        st.session_state.chat_turns.append(
            {
                "role": "user",
                "content": display_text,
                "images": attached_images,
                "videos": attached_videos,
            }
        )

        with chat_container:
            with st.chat_message("user"):
                if attached_images:
                    img_cols = st.columns(min(len(attached_images), 3))
                    for idx, img_bytes in enumerate(attached_images):
                        with img_cols[idx % len(img_cols)]:
                            st.image(img_bytes, width=280)
                if attached_videos:
                    for vid_bytes in attached_videos:
                        st.video(vid_bytes)
                render_rich_content(display_text)

            with st.chat_message("assistant"):
                spinner_msg = (
                    "Распознаю фото (qwen3-vl), ищу в материалах и готовлю ответ…"
                    if attached_images
                    else "Готовлю ответ…"
                )
                with st.spinner(spinner_msg):
                    try:
                        tutor = get_tutor()
                        tutor.store.reload()
                        answer = tutor.ask(
                            question=user_text,
                            course=None if selected_chat_course == "Все курсы" else selected_chat_course,
                            use_web=use_web,
                            verify=do_verify,
                            history=history_messages,
                            task=forced_task,
                            images=attached_images if attached_images else None,
                        )
                    except LLMError as exc:
                        st.error(f"Ошибка обращения к модели: {exc}")
                    else:
                        assistant_turn = {
                            "role": "assistant",
                            "content": answer.text,
                            "answer": answer,
                        }
                        st.session_state.chat_turns.append(assistant_turn)
                        render_assistant_turn(assistant_turn)


# ============================================================ 2. DISPACE НГТУ

with tabs[1]:
    st.subheader("🎓 Интеграция с DiSpace НГТУ (только чтение)")
    st.caption(
        "Все 6 разделов личного кабинета студента НГТУ. Пароль не хранится: используется "
        "локальная сессия браузера (`authToken` и SSO Keycloak). Скачанные курсы мгновенно попадают во вкладку «Материалы»."
    )

    _is_cloud_env = importlib.util.find_spec("playwright") is None or sys.platform != "win32"

    top_c1, top_c2, top_c3 = st.columns([2, 1, 1])
    with top_c1:
        if dispace_api:
            st.success("✅ Подключено к dispace.nstu.ru (сессия браузера активна)")
        else:
            st.warning("⚠️ Сессия DiSpace не активна. Нажми «🔑 Войти через браузер» справа.")
    with top_c2:
        if dispace_api and st.session_state.get("session_dispace_token"):
            if st.button("🚪 Выйти из DiSpace", width="stretch"):
                st.session_state["session_dispace_token"] = ""
                st.rerun()
        else:
            if st.button("🔄 Обновить сессию", width="stretch"):
                with st.spinner("Проверяю сессию браузера DiSpace…"):
                    refreshed = get_session_dispace_api(auto_refresh=not _is_cloud_env)
                if refreshed:
                    st.success("Сессия DiSpace активна!")
                    st.rerun()
                else:
                    st.info("Нажми «🔑 Войти через браузер», чтобы авторизоваться в DiSpace НГТУ.")
    with top_c3:
        if st.button("🔑 Войти через браузер", type="primary", width="stretch"):
            if not _is_cloud_env:
                subprocess.Popen(
                    [sys.executable, str(PROJECT_ROOT / "src" / "lms_reader.py"), "--login"],
                    cwd=str(PROJECT_ROOT),
                )
            open_dispace_browser_login_dialog(_is_cloud_env)

    dispace_section = st.segmented_control(
        "Раздел DiSpace НГТУ",
        options=[
            "📘 Мои курсы",
            "🏛️ Дисциплины и журнал",
            "📅 События и вебинары",
            "🗂️ Каталог курсов",
            "🧪 Тесты и контрольные",
            "✉️ Сообщения",
        ],
        default="📘 Мои курсы",
        label_visibility="collapsed",
    )

    if not dispace_api:
        st.info("Войди в DiSpace НГТУ кнопкой выше, чтобы просматривать разделы и скачивать курсы.")
    else:
        # ----------------------------------------------------- 2.1 Мои курсы
        if dispace_section == "📘 Мои курсы":
            st.markdown(f"#### 📘 Мои курсы ([открыть в DiSpace]({MY_COURSES_URL}))")
            try:
                my_courses = dispace_api.list_courses()
            except LoginRequired as exc:
                st.error(str(exc))
                my_courses = []

            col_mc1, col_mc2 = st.columns([3, 2])
            with col_mc1:
                if not my_courses:
                    st.info("Список курсов пуст.")
                for c in my_courses:
                    cid = c["id"]
                    c_title = c["title"]
                    disc_badge = f" · дисциплина: **{c['discipline']}**" if c.get("discipline") else ""
                    already_in_store = len(store.list_sources(c.get("discipline") or c_title)) > 0 or len(store.list_sources(c_title)) > 0
                    status_icon = "✅ В базе" if already_in_store else "📥 Не скачан"

                    with st.container(border=True):
                        st.markdown(f"**#{cid} — {c_title}** ({status_icon}){disc_badge}")
                        st.caption(f"Ссылка: {c['url']}")

                        b_col1, b_col2 = st.columns([1, 1])
                        with b_col1:
                            if st.button(
                                "📥 Скачать курс в Материалы",
                                key=f"imp_course_{cid}",
                                type="primary",
                                width="stretch",
                            ):
                                target_course_name = c.get("discipline") or c_title
                                prog = st.progress(0.0, text=f"Чтение модулей курса «{c_title}»…")

                                def _cb(idx: int, total: int, item_title: str) -> None:
                                    prog.progress(
                                        idx / max(total, 1),
                                        text=f"Сохраняю ({idx}/{total}): {item_title[:60]}",
                                    )

                                try:
                                    saved = dispace_api.import_course_to_store(
                                        store,
                                        course_id=cid,
                                        course_name=target_course_name,
                                        include_files=True,
                                        on_progress=_cb,
                                    )
                                    get_tutor.clear()
                                    prog.progress(1.0, text="Готово!")
                                    st.success(
                                        f"Курс «{target_course_name}» загружен в Материалы! "
                                        f"Добавлено/обновлено источников: {len(saved)}."
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Ошибка при импорте курса: {exc}")

                        with b_col2:
                            if st.button(
                                "🔍 Показать структуру и баллы",
                                key=f"struct_course_{cid}",
                                width="stretch",
                            ):
                                st.session_state["inspect_course_id"] = cid

            with col_mc2:
                st.markdown("**Импорт по прямой ссылке или ID**")
                with st.form("direct_dispace_import"):
                    direct_url = st.text_input(
                        "Ссылка на курс или тему DiSpace",
                        placeholder="https://dispace.nstu.ru/course/3121/32823",
                    )
                    direct_course_name = st.text_input(
                        "Название курса в базе (необязательно)",
                        placeholder="Математический анализ",
                    )
                    direct_submit = st.form_submit_button(
                        "Скачать в Материалы", type="primary", width="stretch"
                    )
                if direct_submit and direct_url.strip():
                    target_info = parse_dispace_target(direct_url.strip())
                    try:
                        if target_info["kind"] == "course":
                            saved = dispace_api.import_course_to_store(
                                store,
                                course_id=target_info["course_id"],
                                course_name=direct_course_name.strip() or None,
                                include_files=True,
                            )
                            get_tutor.clear()
                            st.success(f"Импортировано источников курса: {len(saved)}")
                            st.rerun()
                        elif target_info["kind"] == "theme":
                            c_obj = dispace_api.get_course(target_info["course_id"])
                            c_name = (
                                direct_course_name.strip()
                                or str(c_obj.get("title") or f"Курс {target_info['course_id']}").strip()
                            )
                            items = dispace_api.extract_theme_materials(
                                target_info["course_id"], target_info["theme_id"], include_files=True
                            )
                            for it in items:
                                store.add_source(course=c_name, title=it.title, text=it.text, url=it.url)
                            store.reload()
                            get_tutor.clear()
                            st.success(f"Импортировано материалов темы: {len(items)} в курс «{c_name}»")
                            st.rerun()
                        else:
                            st.warning("Укажи ссылку вида https://dispace.nstu.ru/course/ID или /course/ID/THEME_ID")
                    except Exception as exc:
                        st.error(f"Ошибка импорта: {exc}")

                inspect_cid = st.session_state.get("inspect_course_id")
                if inspect_cid:
                    try:
                        c_detail = dispace_api.get_course(int(inspect_cid))
                        c_scores = dispace_api.get_course_scores(int(inspect_cid))
                        st.markdown(f"##### Структура: {c_detail.get('title')} (#{inspect_cid})")
                        if c_scores and c_scores.get("total_score") is not None:
                            st.caption(f"Баллы по курсу: **{c_scores.get('total_score')}**")
                        for mod in c_detail.get("modules") or []:
                            if mod.get("hidden"):
                                continue
                            with st.expander(f"📦 {mod.get('title')}", expanded=False):
                                for th in mod.get("themes") or []:
                                    if th.get("hidden"):
                                        continue
                                    tid = th.get("id")
                                    st.markdown(
                                        f"- `{tid}` {th.get('title')} "
                                        f"([открыть](https://dispace.nstu.ru/course/{inspect_cid}/{tid}))"
                                    )
                    except Exception as exc:
                        st.error(f"Не удалось загрузить структуру курса: {exc}")

        # ------------------------------------------ 2.2 Дисциплины и журнал
        elif dispace_section == "🏛️ Дисциплины и журнал":
            st.markdown(f"#### 🏛️ Учебный процесс: дисциплины, контрольные и журнал ([/education]({EDUCATION_URL}))")
            try:
                disciplines = dispace_api.list_disciplines()
            except Exception as exc:
                st.error(f"Ошибка загрузки дисциплин: {exc}")
                disciplines = []

            for d in disciplines:
                did = d["id"]
                d_title = d["title"]
                lectors = ", ".join(d.get("lectors") or []) or "преподаватель не указан"
                with st.container(border=True):
                    head_c1, head_c2 = st.columns([3, 1])
                    with head_c1:
                        st.markdown(f"##### 📚 {d_title} (`#{did}`)")
                        st.caption(f"Преподаватели: {lectors} · [Страница дисциплины]({d['url']})")
                        if d.get("course_id"):
                            st.markdown(
                                f"Прикреплённый электронный курс: **#{d['course_id']} «{d['course_title']}»**"
                            )
                    with head_c2:
                        if d.get("course_id"):
                            if st.button(
                                "📥 Скачать курс дисциплины",
                                key=f"imp_disc_course_{did}",
                                type="primary",
                                width="stretch",
                            ):
                                with st.spinner(f"Скачиваю курс «{d['course_title']}»…"):
                                    saved = dispace_api.import_course_to_store(
                                        store,
                                        course_id=int(d["course_id"]),
                                        course_name=d_title,
                                        include_files=True,
                                    )
                                    get_tutor.clear()
                                st.success(f"Сохранено источников: {len(saved)} в курс «{d_title}»")
                                st.rerun()

                    # Контрольные мероприятия и курсовые работы
                    c_events = d.get("control_events") or []
                    if c_events:
                        st.markdown("**Контрольные работы и задания:**")
                        for ce in c_events:
                            ce_files = ce.get("files") or []
                            f_names = ", ".join(f.get("title", "файл") for f in ce_files)
                            ce_col1, ce_col2 = st.columns([3, 1])
                            with ce_col1:
                                st.markdown(
                                    f"- **{ce['name']}**"
                                    + (f" (макс. балл: {ce['ects_limit']})" if ce.get("ects_limit") else "")
                                    + (f" · стадия/балл: `{ce['score']}`" if ce.get("score") is not None else "")
                                )
                                if ce.get("description"):
                                    st.caption(ce["description"][:300])
                                if f_names:
                                    st.caption(f"📎 Прикреплённые файлы: {f_names}")
                            with ce_col2:
                                if ce_files or ce.get("description"):
                                    if st.button(
                                        "📥 В Материалы",
                                        key=f"imp_ce_{did}_{ce['id']}",
                                        width="stretch",
                                    ):
                                        with st.spinner("Скачиваю файлы задания…"):
                                            saved_ce = dispace_api.import_control_event_to_store(
                                                store, d_title, ce
                                            )
                                            get_tutor.clear()
                                        st.success(f"Добавлено источников: {len(saved_ce)}")
                                        st.rerun()

                    # Журнал успеваемости
                    journal_cols = dispace_api.get_discipline_journal(int(did))
                    if journal_cols:
                        with st.expander(f"📊 Журнал оценок ({len(journal_cols)} контрольных точек)"):
                            j_rows = [
                                {
                                    "Контрольное мероприятие": _fmt_cell(jc.get("title")),
                                    "Макс. балл": _fmt_cell(jc.get("ects_limit")),
                                    "Балл (ECTS)": _fmt_cell(jc.get("ects_mark")),
                                    "Оценка": _fmt_cell(jc.get("text_mark")),
                                }
                                for jc in journal_cols
                            ]
                            st.dataframe(j_rows, width="stretch", hide_index=True)

        # ------------------------------------------ 2.3 События и вебинары
        elif dispace_section == "📅 События и вебинары":
            st.markdown(f"#### 📅 События, онлайн-лекции и вебинары ([/events]({EVENTS_URL}))")
            ev_col1, ev_col2 = st.columns([1, 1])
            with ev_col1:
                sel_month = st.selectbox(
                    "Месяц",
                    list(range(1, 13)),
                    index=9,
                    format_func=lambda m: [
                        "Январь", "Февраль", "Март", "Апрель", "Май", "Июнь",
                        "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь",
                    ][m - 1],
                )
            with ev_col2:
                sel_year = st.selectbox("Год", [2025, 2026, 2027], index=1)

            events = dispace_api.list_events(month=sel_month, year=sel_year)
            records = dispace_api.list_webinar_records(limit=20)

            if events:
                st.markdown(f"**Расписание на {sel_month:02d}.{sel_year} ({len(events)}):**")
                ev_rows = [{k: _fmt_cell(v) for k, v in ev.items()} for ev in events]
                st.dataframe(ev_rows, width="stretch", hide_index=True)
            else:
                st.info(f"На {sel_month:02d}.{sel_year} запланированных вебинаров в календаре нет.")

            st.markdown("**Записи прошедших вебинаров:**")
            if records:
                for rec in records:
                    with st.container(border=True):
                        st.markdown(f"**{rec['title']}** ({rec['date']})")
                        if rec.get("discipline"):
                            st.caption(rec["discipline"])
                        if rec.get("url"):
                            render_video_url(rec["url"], rec["title"])
                            st.markdown(f"[Открыть запись в новой вкладке]({rec['url']})")
            else:
                st.caption("Записей вебинаров пока нет.")

        # ---------------------------------------------- 2.4 Каталог курсов
        elif dispace_section == "🗂️ Каталог курсов":
            st.markdown(f"#### 🗂️ Каталог электронных курсов НГТУ ([/courses]({CATALOG_URL}))")
            cat_query = st.text_input(
                "Поиск по названию, автору или ID курса",
                placeholder="Например: математика, алгебра, физика, программирование…",
            )
            cat_data = dispace_api.list_catalog_courses(search=cat_query, limit=20)
            cat_courses = cat_data.get("courses") or []
            st.caption(
                f"Всего курсов в каталоге НГТУ: **{cat_data.get('count', 0)}** · показано: **{len(cat_courses)}**"
            )

            for cc in cat_courses:
                cid = cc["id"]
                authors_str = ", ".join(cc.get("authors") or []) or "Авторы не указаны"
                with st.container(border=True):
                    c1, c2 = st.columns([4, 1])
                    with c1:
                        st.markdown(f"**#{cid} — {cc['title']}**")
                        st.caption(f"Авторы: {authors_str} · [Открыть в DiSpace]({cc['url']})")
                        if cc.get("description"):
                            st.caption(cc["description"])
                    with c2:
                        if st.button("📥 Импорт", key=f"cat_imp_{cid}", width="stretch"):
                            try:
                                with st.spinner(f"Скачиваю курс #{cid}…"):
                                    saved = dispace_api.import_course_to_store(
                                        store, course_id=cid, course_name=cc["title"]
                                    )
                                    get_tutor.clear()
                                if saved:
                                    st.success(f"Сохранено: {len(saved)} ист.")
                                    st.rerun()
                                else:
                                    st.warning("Курс закрыт или не содержит открытых текстовых тем.")
                            except Exception as exc:
                                st.warning(f"Доступ ограничен или курс пуст: {exc}")

        # ----------------------------------------- 2.5 Тесты и контрольные
        elif dispace_section == "🧪 Тесты и контрольные":
            st.markdown(f"#### 🧪 Тесты и контрольные темы ([/tests]({TESTS_URL}))")
            st.caption(
                "DIBot никогда не проходит и не отправляет тесты за студента. "
                "Вместо этого ты можешь одной кнопкой сгенерировать тренировочный вариант по теме любого теста "
                "во вкладке «Тренажёр» с проверкой эталонных решений через SymPy."
            )
            with st.spinner("Сканирую назначенные тесты и модули курсов…"):
                tests_info = dispace_api.list_tests()

            assigned = tests_info.get("assigned_tests") or []
            course_tests = tests_info.get("course_tests") or []

            st.markdown(f"**Назначенные сессии тестирования (`/tests`): {len(assigned)}**")
            if assigned:
                safe_assigned = [
                    {str(k): _fmt_cell(v) for k, v in row.items()} if isinstance(row, dict) else {"Сессия": _fmt_cell(row)}
                    for row in assigned
                ]
                st.dataframe(safe_assigned, width="stretch", hide_index=True)
            else:
                st.caption("Активных назначенных сессий в разделе `/tests` сейчас нет.")

            st.markdown(f"**Тесты и контрольные работы внутри твоих курсов ({len(course_tests)}):**")
            for idx, ct in enumerate(course_tests):
                with st.container(border=True):
                    tc1, tc2, tc3 = st.columns([3, 1, 1])
                    with tc1:
                        st.markdown(
                            f"**{ct['theme_title']}** ({ct['type']})\n\n"
                            f"Курс: *{ct['course_title']}* · Модуль: *{ct['module_title']}*"
                        )
                        st.caption(f"Ссылка: {ct['url']}")
                    with tc2:
                        if st.button(
                            "🏋️ В Тренажёр",
                            key=f"test_to_practice_{idx}",
                            type="primary",
                            width="stretch",
                        ):
                            st.session_state["practice_prefill_topic"] = (
                                f"{ct['module_title']}: {ct['theme_title']}"
                            )
                            st.session_state["practice_prefill_course"] = ct["course_title"]
                            st.toast(
                                f"Тема «{ct['theme_title']}» передана на вкладку «🧪 Тренажёр»!",
                                icon="🧪",
                            )
                    with tc3:
                        if ct.get("theme_id"):
                            if st.button(
                                "📥 В Материалы",
                                key=f"test_imp_{idx}",
                                width="stretch",
                            ):
                                try:
                                    items = dispace_api.extract_theme_materials(
                                        int(ct["course_id"]),
                                        int(ct["theme_id"]),
                                        module_title=ct["module_title"],
                                        theme_title=ct["theme_title"],
                                    )
                                    for it in items:
                                        store.add_source(
                                            course=ct["course_title"],
                                            title=it.title,
                                            text=it.text,
                                            url=it.url,
                                        )
                                    store.reload()
                                    get_tutor.clear()
                                    if items:
                                        st.success(f"Сохранено: {len(items)}")
                                        st.rerun()
                                    else:
                                        st.info("Эта тема содержит только закрытый банк тестовых вопросов.")
                                except Exception as exc:
                                    st.warning(f"Не удалось прочитать тему: {exc}")

        # --------------------------------------------------- 2.6 Сообщения
        elif dispace_section == "✉️ Сообщения":
            st.markdown(f"#### ✉️ Сообщения и диалоги с преподавателями ([/messages]({MESSAGES_URL}))")
            dialogues = dispace_api.list_messages(limit=25)
            if not dialogues:
                st.info("Диалогов не найдено.")
            else:
                msg_col1, msg_col2 = st.columns([2, 3])
                with msg_col1:
                    st.markdown("**Диалоги:**")
                    selected_conv = st.radio(
                        "Выбери собеседника",
                        dialogues,
                        format_func=lambda d: f"{d['conversant_name']}"
                        + (f" (🔴 {d['count_unread']})" if d.get("count_unread") else ""),
                        label_visibility="collapsed",
                    )
                with msg_col2:
                    if selected_conv and selected_conv.get("conversant_id"):
                        conv_id = int(selected_conv["conversant_id"])
                        conv_name = selected_conv["conversant_name"]
                        st.markdown(f"##### Переписка: {conv_name}")
                        thread = dispace_api.get_dialogue(conv_id, limit=30)

                        if st.button(
                            "📥 Сохранить сообщения диалога в Материалы",
                            key=f"save_dlg_{conv_id}",
                            width="stretch",
                        ):
                            combined_thread = "\n\n---\n\n".join(
                                f"**{m['sender_name']}** ({m['date'][:16]}):\n{m['text']}"
                                for m in thread
                                if m.get("text")
                            )
                            if combined_thread.strip():
                                store.add_source(
                                    course="Сообщения преподавателей",
                                    title=f"Диалог: {conv_name}",
                                    text=combined_thread,
                                    url=MESSAGES_URL,
                                )
                                store.reload()
                                get_tutor.clear()
                                st.success(f"Переписка с «{conv_name}» сохранена в Материалы!")
                                st.rerun()

                        for m in thread:
                            with st.container(border=True):
                                st.caption(f"**{m['sender_name']}** · {m['date'][:19].replace('T', ' ')}")
                                if m.get("theme"):
                                    st.markdown(f"*{m['theme']}*")
                                render_rich_content(m.get("text") or "*(без текста)*")
                                for f_info in m.get("files") or []:
                                    f_title = f_info.get("title") or "Файл"
                                    f_url = f_info.get("url") or ""
                                    if f_url:
                                        st.caption(f"📎 [{f_title}]({f_url})")


# ============================================================= 3. МАТЕРИАЛЫ

with tabs[2]:
    store = sync_materials()
    stats = store.stats()

    head_m1, head_m2 = st.columns([4, 1])
    with head_m1:
        st.subheader("📚 Локальная база учебных материалов")
    with head_m2:
        if st.button("🔄 Обновить список", width="stretch"):
            store.reload()
            get_tutor.clear()
            st.rerun()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Курсов", len(stats["courses"]))
    m2.metric("Источников", stats["sources"])
    m3.metric("Фрагментов (чанков)", stats["chunks"])
    m4.metric("Символов текста", f"{stats['chars']:,}".replace(",", " "))

    with st.expander("➕ Добавить свой файл (.pdf, .docx, .md, .txt, .tex, .csv, .html) в любой кодировке", expanded=False):
        with st.form("add_material"):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                course = st.text_input("Курс / дисциплина", placeholder="Линейная алгебра")
                title = st.text_input("Название материала", placeholder="Лекция 1. Матрицы и определители")
            with f_col2:
                url = st.text_input("Ссылка (необязательно)", placeholder="https://dispace.nstu.ru/...")
                uploaded = st.file_uploader(
                    "Файл .txt / .md / .tex / .csv / .html / .pdf / .docx (автоопределение UTF-8, CP1251, KOI8-R, UTF-16)",
                    type=["txt", "md", "markdown", "tex", "csv", "html", "htm", "pdf", "docx"],
                )
            pasted = st.text_area("…или вставь текст лекции / методички", height=120)
            add = st.form_submit_button("Сохранить в базу", type="primary", width="stretch")

        if add:
            if not course.strip() or not (uploaded or pasted.strip()):
                st.warning("Укажи название курса и прикрепи файл (или вставь текст).")
            else:
                text = pasted
                if uploaded:
                    uname = uploaded.name.lower()
                    raw_data = bytes(uploaded.getbuffer())
                    if uname.endswith(".pdf"):
                        try:
                            text = extract_pdf_bytes(raw_data)
                        except Exception as exc:
                            st.error(f"Не удалось прочитать PDF: {exc}")
                            text = ""
                    elif uname.endswith(".docx"):
                        try:
                            text = extract_docx_bytes(raw_data)
                        except Exception as exc:
                            st.error(f"Не удалось прочитать DOCX: {exc}")
                            text = ""
                    else:
                        text = decode_bytes_smart(raw_data)

                if text.strip():
                    source = store.add_source(
                        course=course.strip(),
                        title=(title.strip() or (uploaded.name if uploaded else "Материал")),
                        text=text,
                        url=url.strip(),
                    )
                    store.reload()
                    get_tutor.clear()
                    st.success(
                        f"Сохранено: «{source.title}» — {len(source.chunk_ids)} фрагментов ({source.chars} симв.)"
                    )
                    st.rerun()

    st.divider()
    st.markdown("#### Загруженные курсы и методички")
    if not store.sources:
        st.info("База материалов пуста. Скачай курс на вкладке «🎓 DiSpace НГТУ» или загрузи файл выше.")
    else:
        for course_name in store.list_courses():
            course_sources = store.list_sources(course_name)
            total_course_chars = sum(s.chars for s in course_sources)
            total_course_chunks = sum(len(s.chunk_ids) for s in course_sources)
            with st.expander(
                f"📁 {course_name} — {len(course_sources)} ист. · {total_course_chunks} фр. · {total_course_chars:,} симв.".replace(
                    ",", " "
                ),
                expanded=True,
            ):
                for source in course_sources:
                    with st.container(border=True):
                        cols = st.columns([5, 1, 1])
                        with cols[0]:
                            st.markdown(
                                f"**{source.title}** — `{source.chars:,}` симв., `{len(source.chunk_ids)}` фр.".replace(
                                    ",", " "
                                )
                            )
                            if source.url:
                                st.caption(f"Источник: {source.url}")
                        with cols[1]:
                            preview_key = f"prev_toggle_{source.id}"
                            if st.button("👁️ Текст", key=f"btn_prev_{source.id}", width="stretch"):
                                st.session_state[preview_key] = not st.session_state.get(preview_key, False)
                        with cols[2]:
                            if st.button("🗑️ Удалить", key=f"del_{source.id}", width="stretch"):
                                store.remove_source(source.id)
                                store.reload()
                                get_tutor.clear()
                                st.rerun()

                        if st.session_state.get(f"prev_toggle_{source.id}", False):
                            full_text = store.get_source_text(source.id)
                            view_mode = st.radio(
                                "Режим просмотра",
                                ["📐 Форматированный (формулы LaTeX, фото, видео)", "📄 Исходный текст"],
                                horizontal=True,
                                key=f"view_mode_{source.id}",
                                label_visibility="collapsed",
                            )
                            if view_mode.startswith("📐"):
                                with st.container(border=True):
                                    render_rich_content(
                                        full_text[:12000]
                                        + ("\n\n*… [показаны первые 12 000 символов]*" if len(full_text) > 12000 else "")
                                    )
                            else:
                                st.text_area(
                                    "Содержимое источника",
                                    value=full_text[:12000]
                                    + ("\n\n… [показаны первые 12000 символов]" if len(full_text) > 12000 else ""),
                                    height=260,
                                    disabled=True,
                                    key=f"ta_prev_{source.id}",
                                )


# ============================================================= 4. КОНСПЕКТЫ

with tabs[3]:
    st.subheader("📝 Структурированные конспекты со ссылками на источники")
    store = sync_materials()
    if not store.sources:
        st.info("Сначала добавь материалы на вкладке «🎓 DiSpace НГТУ» или «📚 Материалы».")
    else:
        course_names = store.list_courses()
        selected_course = st.selectbox("Курс", course_names, key="notes_course")
        sources = store.list_sources(selected_course) if selected_course else []
        selected = (
            st.selectbox(
                "Материал", sources, format_func=lambda s: s.title, key="notes_source"
            )
            if sources
            else None
        )

        if selected is None:
            st.info("В выбранном курсе пока нет источников.")
        else:
            col1, col2 = st.columns([1, 3])
            with col1:
                rebuild = st.button("✨ Собрать конспект", type="primary", width="stretch")
            with col2:
                st.caption(
                    f"{selected.chars} символов, {len(selected.chunk_ids)} фрагментов. "
                    "Каждый тезис получит ссылку `[#N]` на конкретный фрагмент лекции."
                )

            note_path = store.notes_dir / slugify(selected_course) / f"{slugify(selected.title)}.md"

            if rebuild:
                builder = NoteBuilder(get_tutor().llm, store)
                progress = st.progress(0.0, text="Готовлю фрагменты…")

                def on_progress(position: int, total: int, chunk) -> None:
                    progress.progress(position / max(total, 1), text=f"Обрабатываю фрагмент {position} из {total}…")

                try:
                    result = builder.build(selected, on_progress=on_progress)
                except (LLMError, ValueError) as exc:
                    st.error(f"Не удалось собрать конспект: {exc}")
                else:
                    progress.progress(1.0, text="Готово")
                    st.success(result.summary_line())
                    st.rerun()

            if note_path.exists():
                note_md = note_path.read_text(encoding="utf-8")
                st.download_button(
                    "📥 Скачать конспект (.md)",
                    data=note_md,
                    file_name=note_path.name,
                    mime="text/markdown",
                )
                with st.container(border=True):
                    render_rich_content(note_md)
            else:
                st.info("Конспект для этого материала ещё не собран. Нажми «Собрать конспект».")


# ============================================================== 5. ТРЕНАЖЁР

with tabs[4]:
    st.subheader("🧪 Тренажёр: генерация вариантов с самопроверкой")
    store = sync_materials()

    prefill_topic = st.session_state.get("practice_prefill_topic", "")
    prefill_course = st.session_state.get("practice_prefill_course", "Все курсы")

    practice_courses = ["Все курсы"] + store.list_courses()
    default_course_idx = (
        practice_courses.index(prefill_course)
        if prefill_course in practice_courses
        else 0
    )

    with st.form("practice_form"):
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            topic = st.text_input(
                "Тема тренировки",
                value=prefill_topic,
                placeholder="Например: Определители, метод Крамера, предел последовательности",
            )
        with col2:
            p_course = st.selectbox("Опираться на курс", practice_courses, index=default_course_idx)
        with col3:
            count = st.slider("Заданий", 1, 8, 3)
        with col4:
            difficulty = st.selectbox("Уровень", ["лёгкий", "средний", "сложный"], index=1)

        verify_all = st.toggle(
            "Проверить эталонные решения (SymPy пересчитает ответ, вторая модель проверит ход)",
            value=True,
        )
        generate = st.form_submit_button("Сгенерировать вариант", type="primary", width="stretch")

    if generate and topic.strip():
        with st.spinner("Составляю задания по материалам курса и проверяю эталоны…"):
            try:
                tasks, reports = get_tutor().practice(
                    topic=topic.strip(),
                    count=count,
                    difficulty=difficulty,
                    course=None if p_course == "Все курсы" else p_course,
                    verify=verify_all,
                )
            except LLMError as exc:
                st.error(f"Модель недоступна: {exc}")
                tasks, reports = [], []
        st.session_state.practice = (topic.strip(), tasks, reports, difficulty)

    if st.session_state.get("practice"):
        p_topic, tasks, reports, p_diff = st.session_state.practice
        st.caption(f"Тема: **{p_topic}** · уровень: {p_diff} · заданий: {len(tasks)}")
        if not tasks:
            st.warning("Задания не распознаны в ответе модели. Попробуй сформулировать тему конкретнее.")
        for i, task_item in enumerate(tasks, start=1):
            report = reports[i - 1] if i - 1 < len(reports) else None
            badge = ""
            if report:
                badge = {
                    "confirmed": " · ✅ эталон проверен",
                    "disputed": " · ⛔ эталон спорный",
                    "unverified": " · ⚠️ эталон не проверен",
                }.get(report.verdict, "")
            with st.container(border=True):
                render_rich_content(f"**Задание {i}.** {task_item['statement']}")
                st.caption(f"Уровень: {task_item.get('difficulty') or p_diff}{badge}")
                with st.expander("Показать эталонное решение и ответ"):
                    render_rich_content(f"**Ответ:** {task_item.get('answer') or '—'}")
                    if task_item.get("solution"):
                        render_rich_content(task_item["solution"])
                    if report:
                        for check in report.checks:
                            mark = {"pass": "✅", "fail": "❌", "skipped": "➖", "error": "⚠️"}.get(
                                check.status, "•"
                            )
                            st.caption(f"{mark} {check.layer}: {check.detail}")

        if tasks:
            as_md = "\n\n".join(
                f"**{i}.** {t['statement']}\n\nОтвет: {t.get('answer', '')}"
                for i, t in enumerate(tasks, 1)
            )
            st.download_button(
                "📥 Скачать вариант (.md)",
                as_md,
                file_name="вариант_тренировки.md",
                mime="text/markdown",
            )


# =============================================================== 6. МОДЕЛИ

with tabs[5]:
    st.subheader("⚙️ Модели и маршрутизация")
    llm = get_tutor().llm

    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("**Маршруты задач (`config/settings.json`)**")
        rows = []
        for task_name in llm.router.routing:
            route = llm.router.route(task_name)
            rows.append(
                {
                    "Задача": str(task_name),
                    "Основная модель": str(route.model),
                    "Размышления (think)": "да" if route.think else "нет",
                    "Температура": _fmt_cell(route.temperature),
                    "Контекст (num_ctx)": _fmt_cell(route.num_ctx),
                    "Запасные модели": ", ".join(route.fallback) or "—",
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

        st.markdown("**Независимость перекрёстной проверки**")
        for model_name in llm.router.models:
            verifier = llm.router.verifier(model_name)
            ok = verifier and verifier != model_name
            st.caption(f"{'✅' if ok else '⚠️'} решает `{model_name}` → проверяет `{verifier or '—'}`")

    with col2:
        st.markdown("**Профиль локального сервера Ollama**")
        if status["ok"]:
            st.caption(f"Адрес сервера: `{status['base_url']}` · Доступно моделей: **{len(status['models'])}**")
            if status["models"]:
                st.caption("Установленные модели: " + ", ".join(f"`{m}`" for m in status["models"][:8]))
        else:
            st.caption("Локальный сервер Ollama в текущей сессии не подключён.")
        st.caption(
            "Контекст по умолчанию (`num_ctx = 8192`) сбалансирован для видеокарт от 8 ГБ VRAM и CPU/RAM. "
            "При наличии большего объёма VRAM или других моделей параметры настраиваются в `config/settings.json`."
        )
        if status["loaded"]:
            st.markdown("**Сейчас загружено в память (VRAM / RAM):**")
            for item in status["loaded"]:
                st.caption(f"• `{item['name']}` · {item['size_gb']} ГБ · {item['processor']}")

    st.divider()
    st.markdown("### 🔒 Безопасный удалённый доступ к локальным моделям (Ollama Gateway)")
    st.caption(
        "Позволяет подключаться к локальным моделям Ollama на твоём ПК или сервере "
        "из задеплоенного приложения в Streamlit Cloud, с телефона или ноутбука — изолированно для каждого устройства. "
        "Вместо прямого открытия порта Ollama используется защищённый прокси-шлюз с 256-битным токеном, "
        "защитой от перебора (rate-limit) и жёсткой блокировкой удаления/скачивания моделей."
    )

    gw_state_file = PROJECT_ROOT / "data" / "gateway_state.json"
    gw_token_file = PROJECT_ROOT / "data" / "gateway_token.txt"
    gw_state: dict[str, Any] = {}
    if gw_state_file.exists():
        try:
            import json as _json

            gw_state = _json.loads(gw_state_file.read_text(encoding="utf-8"))
        except Exception:
            gw_state = {}

    gw_token = gw_token_file.read_text(encoding="utf-8").strip() if gw_token_file.exists() else ""
    gw_public_url = str(gw_state.get("public_url") or "")

    # Проверим, отвечает ли локальный шлюз на 127.0.0.1:11435
    gw_alive = False
    if gw_token:
        try:
            r_gw = requests.get(
                "http://127.0.0.1:11435/health",
                headers={"Authorization": f"Bearer {gw_token}"},
                timeout=1.5,
            )
            gw_alive = r_gw.status_code == 200
        except requests.RequestException:
            gw_alive = False

    gw_col1, gw_col2 = st.columns([2, 3])
    with gw_col1:
        if gw_alive and gw_public_url:
            st.success(f"🟢 Защищённый туннель активен\n\n`{gw_public_url}`")
        elif gw_alive:
            st.info("🟡 Локальный шлюз (127.0.0.1:11435) запущен, ожидание HTTPS-туннеля Cloudflare…")
        else:
            st.warning("⚪ Удалённый шлюз на этом ПК сейчас не запущен")

        if st.button("🚀 Запустить / перезапустить защищённый HTTPS-туннель", type="primary", width="stretch"):
            subprocess.Popen(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "secure_ollama_gateway.py")],
                cwd=str(PROJECT_ROOT),
            )
            import time as _time

            with st.spinner("Поднимаю защищённый шлюз (11435) и HTTPS-туннель Cloudflare…"):
                for _ in range(25):
                    _time.sleep(0.5)
                    if gw_state_file.exists():
                        try:
                            import json as _json

                            st_data = _json.loads(gw_state_file.read_text(encoding="utf-8"))
                            if st_data.get("public_url"):
                                break
                        except Exception:
                            pass
            st.rerun()

        if st.button("🔑 Перевыпустить секретный 256-битный токен", width="stretch"):
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "secure_ollama_gateway.py"),
                    "--rotate-token",
                    "--no-tunnel",
                ],
                cwd=str(PROJECT_ROOT),
                timeout=3,
            )
            st.success("Токен перевыпущен! Перезапусти туннель кнопкой выше.")
            st.rerun()

    with gw_col2:
        if gw_token:
            st.markdown("**1. Секретный токен (вводить в боковой панели только на своих устройствах):**")
            st.caption(
                "Токен хранится изолированно в `st.session_state` конкретного браузера/устройства и "
                "не передаётся другим посетителям сайта:"
            )
            st.code(gw_token, language="text")
            st.markdown("**2. Для `Settings → Secrets` в Streamlit Cloud (только URL туннеля без ключа!):**")
            st.caption(
                "Укажи в `Secrets` только публичный URL туннеля, чтобы не вводить его вручную, "
                "а секретный токен `dibot_sk_...` вводи только на своих устройствах:"
            )
            secrets_toml = (
                f'OLLAMA_BASE_URL = "{gw_public_url or "https://ЗАПУСТИ-ТУННЕЛЬ-СЛЕВА.trycloudflare.com"}"\n'
            )
            st.code(secrets_toml, language="toml")
        else:
            st.info(
                "Нажми **«🚀 Запустить / перезапустить защищённый HTTPS-туннель»** слева (на домашнем ПК) — "
                "сгенерируется криптостойкий 256-битный токен `dibot_sk_...` и публичный HTTPS-адрес."
            )

    st.divider()
    if st.button("🩺 Запустить полную диагностику Ollama"):
        with st.spinner("Проверяю доступность всех моделей…"):
            report = llm.health()
        st.code(report.as_text())
