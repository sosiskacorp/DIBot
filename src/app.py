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

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(Path(__file__).resolve().parent)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

if not hasattr(sys, "_dibot_import_lock"):
    sys._dibot_import_lock = threading.Lock()  # type: ignore[attr-defined]

with sys._dibot_import_lock:  # type: ignore[attr-defined]
    import materials as _materials_mod  # noqa: E402
    import llm_client as _llm_client_mod  # noqa: E402
    import notes as _notes_mod  # noqa: E402
    import ingest as _ingest_mod  # noqa: E402
    import verification as _verification_mod  # noqa: E402
    import lms_reader as _lms_reader_mod  # noqa: E402
    import tutor as _tutor_mod  # noqa: E402

    # Перезагружаем модули, если исходники изменились (иначе запущенный сервер держит старые классы,
    # и появляются ошибки вида «'DiSpaceAPI' object has no attribute 'get_course_overview'»)
    _mods_all = (
        _materials_mod,
        _llm_client_mod,
        _notes_mod,
        _ingest_mod,
        _verification_mod,
        _lms_reader_mod,
        _tutor_mod,
    )

    def _mod_mtime(_m: Any) -> float:
        try:
            return os.path.getmtime(_m.__file__)
        except Exception:
            return 0.0

    _cur_mtimes = {_m.__name__: _mod_mtime(_m) for _m in _mods_all}
    _prev_mtimes = getattr(sys, "_dibot_mod_mtimes", None)
    _missing_api = (
        not hasattr(_lms_reader_mod.DiSpaceAPI, "get_course_overview")
        or not hasattr(_tutor_mod.Tutor, "solve_course_assignment")
        or not hasattr(_ingest_mod, "paginate_markdown")
    )
    if _missing_api or (_prev_mtimes is not None and _prev_mtimes != _cur_mtimes):
        for _mod in _mods_all:
            sys.modules[_mod.__name__] = _mod
            importlib.reload(_mod)
    sys._dibot_mod_mtimes = _cur_mtimes  # type: ignore[attr-defined]

from ingest import (  # noqa: E402
    clean_pdf_artifacts,
    decode_bytes_smart,
    extract_docx_bytes,
    extract_pdf_bytes,
    format_math_markdown,
    html_to_text,
    paginate_markdown,
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
from llm_client import CLOUD_PROVIDERS, fetch_cloud_models  # noqa: E402

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
    if "session_model_provider" not in st.session_state:
        st.session_state["session_model_provider"] = "auto"
    if "session_cloud_provider" not in st.session_state:
        st.session_state["session_cloud_provider"] = "groq"
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
        provider=st.session_state.get("session_model_provider", "auto"),
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


@st.dialog("📖 Читалка учебного материала", width="large")
def open_material_reader_dialog(
    title: str,
    course: str,
    url: str = "",
    full_text: str = "",
    text: str = "",
) -> None:
    """Отдельное полноэкранное окно для комфортного чтения длинных лекций и методичек с пагинацией."""
    full_text = clean_pdf_artifacts(full_text or text)
    st.markdown(f"#### {title}")
    st.caption(f"Курс: **{course}** · Всего символов: `{len(full_text):,}`".replace(",", " ") + (f" · [Открыть в DiSpace]({url})" if url else ""))

    dlg_key = slugify(title)[:60]
    size_label = st.radio(
        "Размер страницы",
        ["5 000 симв.", "10 000 симв.", "20 000 симв."],
        index=1,
        horizontal=True,
        key=f"dlg_size_{dlg_key}",
        label_visibility="collapsed",
    )
    page_size = int(size_label.split()[0]) * 1000
    pages = paginate_markdown(full_text, page_size=page_size)
    total_pages = len(pages)
    dlg_page_key = f"dlg_reader_page_{dlg_key}_{page_size}"
    cur_page = max(1, min(int(st.session_state.get(dlg_page_key, 1)), total_pages))

    def _nav(suffix: str) -> None:
        nonlocal cur_page
        c1, c2, c3 = st.columns([1, 2, 1])
        with c1:
            if st.button("⬅️ Назад", disabled=(cur_page <= 1), key=f"dlg_prev_{dlg_key}_{suffix}", width="stretch"):
                cur_page -= 1
                st.session_state[dlg_page_key] = cur_page
                st.rerun(scope="fragment")
        with c2:
            st.markdown(
                f"<div style='text-align:center;padding-top:6px;'><b>Страница {cur_page} из {total_pages}</b></div>",
                unsafe_allow_html=True,
            )
        with c3:
            if st.button("Далее ➡️", disabled=(cur_page >= total_pages), key=f"dlg_next_{dlg_key}_{suffix}", width="stretch"):
                cur_page += 1
                st.session_state[dlg_page_key] = cur_page
                st.rerun(scope="fragment")

    if total_pages > 1:
        _nav("top")
    st.divider()
    render_rich_content(pages[cur_page - 1])
    if total_pages > 1:
        st.divider()
        _nav("bottom")


CHAT_HISTORY_FILE = PROJECT_ROOT / "data" / "chat_sessions.json"  # старый общий файл (только для миграции)
CHAT_HISTORY_DIR = PROJECT_ROOT / "data" / "chat_history"
CHAT_UPLOADS_DIR = PROJECT_ROOT / "data" / "chat_uploads"


def _chat_uploads_dir(owner_key: str) -> Path:
    safe = "".join(ch for ch in owner_key if ch.isalnum())[:32] or "common"
    p = CHAT_UPLOADS_DIR / safe
    p.mkdir(parents=True, exist_ok=True)
    return p


def _dispace_user_from_token(token: str) -> dict[str, Any] | None:
    """Достать user_id из уже проверенного (verify_token) JWT DiSpace."""
    import base64 as _b64
    import hashlib as _hashlib

    try:
        payload_b64 = token.split(".")[1]
        payload = json.loads(_b64.urlsafe_b64decode(payload_b64 + "===").decode("utf-8"))
        uid = str(payload.get("user_id") or payload.get("sub") or "").strip()
    except Exception:
        return None
    if not uid:
        return None
    key = _hashlib.sha256(f"dibot-user:{uid}".encode("utf-8")).hexdigest()[:24]
    return {"id": uid, "key": key}


def _chat_history_file(owner_key: str) -> Path | None:
    """Файл истории конкретного пользователя (имя — хэш его user_id). Для гостей история на диск не пишется."""
    if not owner_key or owner_key == "guest":
        return None
    safe = "".join(ch for ch in owner_key if ch.isalnum())[:32]
    return CHAT_HISTORY_DIR / f"{safe}.json"


def _serialize_chat_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    owner_key = st.session_state.get("chat_owner") or st.session_state.get("dibot_user_key") or "common"
    uploads_dir = _chat_uploads_dir(owner_key)

    for s in sessions:
        turns_clean = []
        for t_idx, t in enumerate(s.get("turns") or []):
            ans_obj: Answer | None = t.get("answer")

            # Сохраняем прикреплённые изображения на диск и записываем пути
            saved_images: list[str] = []
            for img_idx, img in enumerate(t.get("images") or []):
                if isinstance(img, str):
                    saved_images.append(img)
                elif isinstance(img, (bytes, bytearray)):
                    img_file = uploads_dir / f"{s['id']}_t{t_idx}_i{img_idx}.png"
                    try:
                        img_file.write_bytes(bytes(img))
                        rel = str(img_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                        saved_images.append(rel)
                    except Exception:
                        pass

            # Сохраняем видео на диск и записываем пути
            saved_videos: list[str] = []
            for vid_idx, vid in enumerate(t.get("videos") or []):
                if isinstance(vid, str):
                    saved_videos.append(vid)
                elif isinstance(vid, (bytes, bytearray)):
                    vid_file = uploads_dir / f"{s['id']}_t{t_idx}_v{vid_idx}.mp4"
                    try:
                        vid_file.write_bytes(bytes(vid))
                        rel = str(vid_file.relative_to(PROJECT_ROOT)).replace("\\", "/")
                        saved_videos.append(rel)
                    except Exception:
                        pass

            turns_clean.append(
                {
                    "role": t.get("role", "user"),
                    "content": t.get("content", ""),
                    "images": saved_images,
                    "videos": saved_videos,
                    "meta": (
                        {
                            "task": ans_obj.task,
                            "model": ans_obj.model,
                            "seconds": ans_obj.seconds,
                            "sources": ans_obj.sources,
                            "context_used": ans_obj.context_used,
                            "thinking": ans_obj.thinking,
                        }
                        if ans_obj
                        else t.get("meta")
                    ),
                }
            )
        out.append(
            {
                "id": s["id"],
                "title": s["title"],
                "updated_at": s.get("updated_at", ""),
                "turns": turns_clean,
            }
        )
    return out


def _init_chat_sessions() -> None:
    """Загрузить историю чатов ТЕКУЩЕГО пользователя DiSpace (у гостей история живёт только во вкладке)."""
    owner = st.session_state.get("dibot_user_key") or "guest"
    if (
        st.session_state.get("chat_owner") == owner
        and "chat_sessions" in st.session_state
        and st.session_state["chat_sessions"]
    ):
        return
    loaded: list[dict[str, Any]] = []
    hist_file = _chat_history_file(owner)
    if hist_file is not None:
        src_file = hist_file
        # Однократный перенос старой общей истории на локальном ПК владельцу
        if not hist_file.exists() and IS_LOCAL_PC and CHAT_HISTORY_FILE.exists():
            src_file = CHAT_HISTORY_FILE
        if src_file.exists():
            try:
                raw = json.loads(src_file.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    loaded = raw
            except Exception:
                loaded = []
    if not loaded:
        now_str = datetime.now().strftime("%d.%m %H:%M")
        loaded = [{"id": f"chat_{int(time.time())}", "title": "Новый диалог", "updated_at": now_str, "turns": []}]
    st.session_state["chat_owner"] = owner
    st.session_state["chat_sessions"] = loaded
    st.session_state["active_chat_id"] = loaded[0]["id"]
    st.session_state.chat_turns = list(loaded[0].get("turns") or [])


def _save_active_chat_turns() -> None:
    _init_chat_sessions()
    active_id = st.session_state.get("active_chat_id")
    turns = st.session_state.get("chat_turns") or []
    now_str = datetime.now().strftime("%d.%m %H:%M")
    for s in st.session_state["chat_sessions"]:
        if s["id"] == active_id:
            s["turns"] = list(turns)
            s["updated_at"] = now_str
            if turns and (s["title"] == "Новый диалог" or s["title"].startswith("Диалог ")):
                first_user = next((t.get("content", "") for t in turns if t.get("role") == "user"), "")
                first_line = first_user.splitlines()[0].strip() if first_user else ""
                if first_line:
                    s["title"] = first_line[:42] + ("…" if len(first_line) > 42 else "")
            break
    hist_file = _chat_history_file(st.session_state.get("chat_owner") or "guest")
    if hist_file is not None:
        try:
            hist_file.parent.mkdir(parents=True, exist_ok=True)
            hist_file.write_text(
                json.dumps(_serialize_chat_sessions(st.session_state["chat_sessions"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass


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

    formatted = format_math_markdown(clean_pdf_artifacts(text))
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


def render_assistant_turn(turn: dict[str, Any], is_latest: bool = False, turn_idx: int = 0) -> None:
    """Отрисовать сообщение ассистента со всеми деталями (фото-OCR, формулы, видео, ход мыслей, проверка)."""
    ans: Answer | None = turn.get("answer")
    if ans is None:
        meta = turn.get("meta") or {}
        if meta.get("thinking"):
            with st.expander("💭 Ход рассуждений модели", expanded=False):
                st.caption(meta["thinking"])
        render_rich_content(turn.get("content", ""))
        if meta.get("model"):
            badges = [
                f"модель: `{meta.get('model')}`",
                f"задача: `{meta.get('task', 'chat')}`",
                f"контекст: **{meta.get('context_used', 'нет')}**",
                f"время: {float(meta.get('seconds') or 0):.1f} с",
            ]
            st.caption(" · ".join(badges))
        if meta.get("sources"):
            st.caption("📚 Источники: " + " · ".join(meta["sources"][:5]))
        if is_latest:
            col_act1, _ = st.columns([1.2, 4])
            with col_act1:
                if st.button("🔄 Перегенерировать ответ", key=f"btn_retry_{turn_idx}", help="Повторить запрос с теми же файлами и условием"):
                    if st.session_state.chat_turns and st.session_state.chat_turns[-1].get("role") == "assistant":
                        st.session_state.chat_turns.pop()
                        st.session_state["trigger_retry"] = True
                        _save_active_chat_turns()
                        st.rerun()
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

    if is_latest:
        col_act1, _ = st.columns([1.2, 4])
        with col_act1:
            if st.button("🔄 Перегенерировать ответ", key=f"btn_retry_{turn_idx}", help="Повторить запрос с теми же файлами и условием"):
                if st.session_state.chat_turns and st.session_state.chat_turns[-1].get("role") == "assistant":
                    st.session_state.chat_turns.pop()
                    st.session_state["trigger_retry"] = True
                    _save_active_chat_turns()
                    st.rerun()


# ------------------------------------------------------------------ сайдбар

store = sync_materials()
stats = store.stats()
status = ollama_status()
dispace_api = get_session_dispace_api(auto_refresh=False)

# Идентификация пользователя по проверенной сессии DiSpace: история чатов привязывается к нему,
# а функции с моделями доступны только авторизованным (гости могут лишь смотреть материалы).
_current_user = _dispace_user_from_token(dispace_api.token) if dispace_api else None
st.session_state["dibot_user_key"] = _current_user["key"] if _current_user else "guest"
IS_AUTHORIZED = _current_user is not None


def require_auth(feature: str, key: str) -> bool:
    """Показать заглушку для гостя и вернуть False, если пользователь не вошёл в DiSpace."""
    if IS_AUTHORIZED:
        return True
    with st.container(border=True):
        st.warning(
            f"🔒 **{feature}** доступно только после входа в DiSpace НГТУ. "
            "Гостевой режим позволяет только просматривать материалы."
        )
        if st.button("🔑 Войти в DiSpace", key=f"auth_gate_{key}", type="primary"):
            open_dispace_browser_login_dialog(
                importlib.util.find_spec("playwright") is None or sys.platform != "win32"
            )
    return False

def _course_progress(cid: int) -> dict[str, Any] | None:
    """Прогресс курса из DiSpace (кэш в сессии пользователя на 10 минут)."""
    if not dispace_api:
        return None
    cache = st.session_state.setdefault("course_progress_cache", {})
    hit = cache.get(cid)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    try:
        ov = dispace_api.get_course_overview(int(cid))
    except Exception:
        ov = None
    cache[cid] = (time.time(), ov)
    return ov


with st.sidebar:
    st.title("🎓 DIBot")
    st.caption("Учебный ИИ-ассистент студента НГТУ")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Курсов в базе", len(stats["courses"]))
    with col_s2:
        st.metric("Фрагментов", stats["chunks"])

    active_prov = st.session_state.get("session_model_provider", "auto")
    prov_mode = st.radio(
        "Источник модели",
        ["🖥️ Локальная Ollama (GPU)", "☁️ Бесплатные облачные модели"],
        index=1 if active_prov == "cloud" else 0,
        label_visibility="collapsed",
    )
    if prov_mode.startswith("☁️"):
        st.session_state["session_model_provider"] = "cloud"
    else:
        st.session_state["session_model_provider"] = "ollama"

    if st.session_state["session_model_provider"] == "cloud":
        cur_cloud_key = st.session_state.get("session_cloud_key", "")
        if cur_cloud_key:
            st.success(f"☁️ Облачная модель: `{st.session_state.get('session_cloud_model', 'llama-3.3-70b-versatile')}`")
        else:
            st.warning("☁️ Настройте бесплатный облачный API ниже")

        with st.expander("⚙️ Настройка облачной модели", expanded=not bool(cur_cloud_key)):
            st.caption("Подключение бесплатных облачных моделей (Groq, Gemini, OpenRouter, OpenAI) со стримингом.")
            prov_keys = list(CLOUD_PROVIDERS.keys())
            saved_cp = st.session_state.get("session_cloud_provider", "groq")
            cp_idx = prov_keys.index(saved_cp) if saved_cp in prov_keys else 0
            chosen_cp = st.selectbox(
                "Облачный провайдер",
                prov_keys,
                index=cp_idx,
                format_func=lambda k: CLOUD_PROVIDERS[k]["name"],
            )

            # При смене провайдера в селекте синхронизируем URL и рекомендуемую модель
            if chosen_cp != st.session_state.get("last_selected_cp"):
                st.session_state["last_selected_cp"] = chosen_cp
                st.session_state["session_cloud_provider"] = chosen_cp
                st.session_state["session_cloud_url"] = CLOUD_PROVIDERS[chosen_cp]["base_url"]
                st.session_state["session_cloud_model"] = CLOUD_PROVIDERS[chosen_cp]["default_model"]
                st.session_state.pop("cached_cloud_models", None)

            cp_data = CLOUD_PROVIDERS[chosen_cp]

            if cp_data.get("key_url"):
                st.link_button("🔑 Получить бесплатный API-ключ", cp_data["key_url"], use_container_width=True)

            c_key = st.text_input(
                "API-ключ",
                value=st.session_state.get("session_cloud_key", ""),
                type="password",
                placeholder="gsk_... / AIza... / sk-or-... / sk-...",
            )

            # Автоопределение провайдера по формату ключа при вставке
            raw_k = c_key.strip()
            if raw_k:
                if raw_k.startswith("gsk_") and chosen_cp != "groq":
                    st.info("💡 Обнаружен ключ Groq — переключаю на Groq Cloud.")
                    st.session_state["session_cloud_provider"] = "groq"
                    st.session_state["last_selected_cp"] = "groq"
                    st.session_state["session_cloud_url"] = CLOUD_PROVIDERS["groq"]["base_url"]
                    st.session_state["session_cloud_model"] = CLOUD_PROVIDERS["groq"]["default_model"]
                    st.session_state.pop("cached_cloud_models", None)
                    st.rerun()
                elif raw_k.startswith("sk-or-") and chosen_cp != "openrouter":
                    st.info("💡 Обнаружен ключ OpenRouter — переключаю на OpenRouter.")
                    st.session_state["session_cloud_provider"] = "openrouter"
                    st.session_state["last_selected_cp"] = "openrouter"
                    st.session_state["session_cloud_url"] = CLOUD_PROVIDERS["openrouter"]["base_url"]
                    st.session_state["session_cloud_model"] = CLOUD_PROVIDERS["openrouter"]["default_model"]
                    st.session_state.pop("cached_cloud_models", None)
                    st.rerun()
                elif raw_k.startswith("AIza") and chosen_cp != "gemini":
                    st.info("💡 Обнаружен ключ Google AI Studio — переключаю на Gemini.")
                    st.session_state["session_cloud_provider"] = "gemini"
                    st.session_state["last_selected_cp"] = "gemini"
                    st.session_state["session_cloud_url"] = CLOUD_PROVIDERS["gemini"]["base_url"]
                    st.session_state["session_cloud_model"] = CLOUD_PROVIDERS["gemini"]["default_model"]
                    st.session_state.pop("cached_cloud_models", None)
                    st.rerun()

            c_url = st.text_input("Base URL", value=st.session_state.get("session_cloud_url") or cp_data["base_url"])

            # Кнопка запроса актуального списка моделей прямо с сервера провайдера
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                fetch_btn = st.button("🔄 Загрузить модели", width="stretch", help="Запросить список моделей по этому API-ключу")
            with col_b2:
                save_btn = st.button("💾 Сохранить", type="primary", width="stretch")

            if fetch_btn and raw_k:
                with st.spinner("Запрашиваю список моделей с сервера…"):
                    real_models = fetch_cloud_models(c_url, raw_k)
                if real_models:
                    st.session_state["cached_cloud_models"] = real_models
                    st.success(f"✅ Доступно {len(real_models)} реальных моделей!")
                else:
                    st.error("Не удалось получить список моделей. Проверьте ключ и URL.")

            known_models = st.session_state.get("cached_cloud_models") or cp_data.get("models") or []
            cur_mod = st.session_state.get("session_cloud_model") or cp_data.get("default_model") or ""
            if cur_mod in known_models:
                m_idx = known_models.index(cur_mod)
            else:
                m_idx = 0

            selected_mod_item = st.selectbox(
                "Модель",
                known_models + ["Ввести имя вручную…"],
                index=m_idx,
            )
            if selected_mod_item == "Ввести имя вручную…":
                chosen_mod_name = st.text_input("Имя модели", value=cur_mod)
            else:
                chosen_mod_name = selected_mod_item

            if save_btn:
                st.session_state["session_cloud_key"] = raw_k
                st.session_state["session_cloud_url"] = c_url.strip()
                st.session_state["session_cloud_model"] = chosen_mod_name.strip()
                st.success("Параметры облачной модели сохранены!")
                st.rerun()

            if cur_cloud_key:
                if st.button("Отключить облачный API", width="stretch"):
                    st.session_state["session_cloud_key"] = ""
                    st.session_state.pop("cached_cloud_models", None)
                    st.rerun()
    else:
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

    if status["loaded"]:
        for item in status["loaded"]:
            st.caption(f"⚡ В памяти: `{item['name']}` ({item['size_gb']} ГБ)")

    if dispace_api:
        st.success(f"DiSpace НГТУ: вы вошли (ID `{_current_user['id'] if _current_user else '—'}`)")
    else:
        st.warning("👤 Гостевой режим: войдите в DiSpace, чтобы пользоваться моделями и сохранять историю чатов")

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
    do_verify = st.toggle("✅ Проверять решения (SymPy)", value=True)

    st.divider()
    _init_chat_sessions()
    st.markdown("**🗂️ История чатов**")
    st.caption(
        "Сохраняется за вашим аккаунтом DiSpace и доступна на любом устройстве после входа."
        if IS_AUTHORIZED
        else "Гость: история не сохраняется и пропадёт при закрытии вкладки."
    )
    ch_col1, ch_col2 = st.columns([1, 1])
    with ch_col1:
        if st.button("➕ Новый чат", width="stretch", type="primary"):
            new_id = f"chat_{int(time.time() * 1000)}"
            now_str = datetime.now().strftime("%d.%m %H:%M")
            st.session_state["chat_sessions"].insert(
                0,
                {"id": new_id, "title": "Новый диалог", "updated_at": now_str, "turns": []},
            )
            st.session_state["active_chat_id"] = new_id
            st.session_state.chat_turns = []
            _save_active_chat_turns()
            st.rerun()
    with ch_col2:
        if st.button("🗑️ Удалить чат", width="stretch"):
            cur_id = st.session_state.get("active_chat_id")
            st.session_state["chat_sessions"] = [
                s for s in st.session_state["chat_sessions"] if s["id"] != cur_id
            ]
            if not st.session_state["chat_sessions"]:
                new_id = f"chat_{int(time.time() * 1000)}"
                now_str = datetime.now().strftime("%d.%m %H:%M")
                st.session_state["chat_sessions"] = [
                    {"id": new_id, "title": "Новый диалог", "updated_at": now_str, "turns": []}
                ]
            st.session_state["active_chat_id"] = st.session_state["chat_sessions"][0]["id"]
            st.session_state.chat_turns = list(st.session_state["chat_sessions"][0].get("turns") or [])
            _save_active_chat_turns()
            st.rerun()

    session_ids = [s["id"] for s in st.session_state["chat_sessions"]]
    session_map = {s["id"]: s for s in st.session_state["chat_sessions"]}
    cur_active_id = st.session_state.get("active_chat_id") or session_ids[0]
    if cur_active_id not in session_map:
        cur_active_id = session_ids[0]
        st.session_state["active_chat_id"] = cur_active_id

    chosen_chat_id = st.radio(
        "Сохранённые диалоги",
        session_ids,
        index=session_ids.index(cur_active_id),
        format_func=lambda sid: (
            f"💬 {session_map[sid]['title']} "
            f"({len(session_map[sid].get('turns') or []) // 2} отв. · {session_map[sid].get('updated_at', '')})"
        ),
        label_visibility="collapsed",
    )
    if chosen_chat_id != cur_active_id:
        _save_active_chat_turns()
        st.session_state["active_chat_id"] = chosen_chat_id
        st.session_state.chat_turns = list(session_map[chosen_chat_id].get("turns") or [])
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
    _init_chat_sessions()

    # Информационная строка текущего диалога с индикатором токенов (компактизация работает автоматически)
    active_sess = next(
        (s for s in st.session_state.get("chat_sessions", []) if s["id"] == st.session_state.get("active_chat_id")),
        None,
    )
    active_title = active_sess["title"] if active_sess else "Диалог"
    turns_cnt = len(st.session_state.chat_turns)
    est_tokens = sum(len(str(t.get("content", ""))) for t in st.session_state.chat_turns) // 4
    st.caption(f"💬 **{active_title}** · Реплик: {turns_cnt} · ~{est_tokens:,} токенов · ⚡ автосжатие контекста активно".replace(",", " "))

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
        for idx, turn in enumerate(st.session_state.chat_turns):
            with st.chat_message(turn["role"]):
                if turn["role"] == "user":
                    if turn.get("images"):
                        for img_idx, img_ref in enumerate(turn["images"]):
                            img_source = img_ref
                            if isinstance(img_ref, str):
                                p = PROJECT_ROOT / img_ref if not Path(img_ref).is_absolute() else Path(img_ref)
                                if p.exists():
                                    img_source = str(p)
                            with st.container(border=True):
                                st.image(img_source, use_container_width=True)
                                with st.expander(f"🔍 Исходное разрешение фото #{img_idx+1} (1:1)", expanded=False):
                                    st.image(img_source, use_container_width=False)
                    if turn.get("videos"):
                        for vid_ref in turn["videos"]:
                            vid_source = vid_ref
                            if isinstance(vid_ref, str):
                                p = PROJECT_ROOT / vid_ref if not Path(vid_ref).is_absolute() else Path(vid_ref)
                                if p.exists():
                                    vid_source = str(p)
                            st.video(vid_source)
                    if turn.get("content"):
                        render_rich_content(turn["content"])
                else:
                    is_latest = (idx == len(st.session_state.chat_turns) - 1)
                    render_assistant_turn(turn, is_latest=is_latest, turn_idx=idx)

    # Проверка на повторный перезапуск промпта при сбое или обрыве
    is_retry_run = False
    if st.session_state.pop("trigger_retry", False):
        if st.session_state.chat_turns and st.session_state.chat_turns[-1].get("role") == "user":
            is_retry_run = True

    # Поле ввода чата с прикреплением фото, видео и документов в любых кодировках
    if not IS_AUTHORIZED:
        require_auth("Чат с моделью", "chat")
        picked_suggestion = None
    chat_submission = st.chat_input(
        "Напиши вопрос, задачу или прикрепи фото, видео, PDF/DOCX/TXT…"
        if IS_AUTHORIZED
        else "🔒 Войдите в DiSpace, чтобы задать вопрос модели",
        disabled=not IS_AUTHORIZED,
        accept_file="multiple",
        file_type=[
            "png", "jpg", "jpeg", "webp", "gif", "bmp",
            "mp4", "webm", "mov",
            "pdf", "docx", "txt", "md", "tex", "csv", "html",
        ],
    )

    user_text = ""
    attached_images: list[Any] = []
    attached_videos: list[Any] = []
    attached_docs_text: list[str] = []

    if is_retry_run:
        last_turn = st.session_state.chat_turns[-1]
        user_text = last_turn.get("content", "")
        attached_images = list(last_turn.get("images") or [])
        attached_videos = list(last_turn.get("videos") or [])
    elif chat_submission is not None:
        owner_key = st.session_state.get("chat_owner") or st.session_state.get("dibot_user_key") or "common"
        uploads_dir = _chat_uploads_dir(owner_key)
        if isinstance(chat_submission, str):
            user_text = chat_submission.strip()
        else:
            user_text = (getattr(chat_submission, "text", "") or "").strip()
            for f_obj in getattr(chat_submission, "files", None) or []:
                fname = str(getattr(f_obj, "name", "") or "").lower()
                raw_data = bytes(f_obj.getvalue())
                s_name = f"{int(time.time()*1000)}_{slugify(f_obj.name)[:30]}{Path(f_obj.name).suffix}"
                saved_f = uploads_dir / s_name
                try:
                    saved_f.write_bytes(raw_data)
                    rel_path = str(saved_f.relative_to(PROJECT_ROOT)).replace("\\", "/")
                except Exception:
                    rel_path = ""
                file_ref = rel_path or raw_data

                if fname.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")):
                    attached_images.append(file_ref)
                elif fname.endswith((".mp4", ".webm", ".mov")):
                    attached_videos.append(file_ref)
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

    if user_text or attached_images or attached_videos or is_retry_run:
        display_text = (
            user_text
            or ("📷 Разбери и реши задание на прикреплённом фото." if attached_images else "🎬 Прикреплено видео.")
        )

        # Автоматическая компактизация сессии при накоплении контекста (если это необходимо)
        source_turns = st.session_state.chat_turns[:-1] if is_retry_run else st.session_state.chat_turns
        total_prev_chars = sum(len(str(t.get("content", ""))) for t in source_turns)
        if len(source_turns) >= 6 or total_prev_chars >= 4500:
            keep_recent = 3
            if len(source_turns) > keep_recent:
                older = source_turns[:-keep_recent]
                recent = source_turns[-keep_recent:]
                hist_for_compact = []
                for t in older:
                    c = t.get("content", "")
                    if not c and t.get("answer"):
                        c = t["answer"].text
                    if c:
                        hist_for_compact.append({"role": t.get("role", "user"), "content": c})
                if hist_for_compact:
                    compacted = get_tutor().compact_history(hist_for_compact, keep_recent=0)
                    summary_text = ""
                    if compacted:
                        summary_text = compacted[0].get("content", "")
                    if summary_text:
                        synthetic_turn = {"role": "user", "content": summary_text, "images": [], "videos": []}
                        if is_retry_run:
                            st.session_state.chat_turns = [
                                synthetic_turn,
                                *recent,
                                st.session_state.chat_turns[-1],
                            ]
                        else:
                            st.session_state.chat_turns = [
                                synthetic_turn,
                                *recent,
                            ]
                        _save_active_chat_turns()
                        source_turns = st.session_state.chat_turns[:-1] if is_retry_run else st.session_state.chat_turns

        # Собираем историю предыдущих реплик для многошагового контекста
        history_messages: list[dict[str, str]] = []
        recent_turns = source_turns[-8:]
        for idx, prev in enumerate(recent_turns):
            role = prev["role"]
            if role == "user":
                content = prev.get("content") or ""
                ocr_cond = prev.get("ocr_condition")
                if not ocr_cond and idx + 1 < len(recent_turns):
                    next_turn = recent_turns[idx + 1]
                    next_ans = next_turn.get("answer")
                    if next_ans and getattr(next_ans, "extracted_image_text", None):
                        ocr_cond = next_ans.extracted_image_text
                if ocr_cond:
                    content = f"{content}\n[Распознанное условие задачи с прикреплённого фото]:\n{ocr_cond}".strip()
            else:
                ans_obj: Answer | None = prev.get("answer")
                content = ans_obj.text if ans_obj else (prev.get("content") or "")
            if content:
                history_messages.append({"role": role, "content": content})

        if not is_retry_run:
            st.session_state.chat_turns.append(
                {
                    "role": "user",
                    "content": display_text,
                    "images": attached_images,
                    "videos": attached_videos,
                }
            )

        with chat_container:
            if not is_retry_run:
                with st.chat_message("user"):
                    if attached_images:
                        for idx, img_ref in enumerate(attached_images):
                            img_src = img_ref
                            if isinstance(img_ref, str):
                                p = PROJECT_ROOT / img_ref if not Path(img_ref).is_absolute() else Path(img_ref)
                                if p.exists():
                                    img_src = str(p)
                            with st.container(border=True):
                                st.image(img_src, use_container_width=True)
                                with st.expander(f"🔍 Исходное разрешение фото #{idx+1} (1:1)", expanded=False):
                                    st.image(img_src, use_container_width=False)
                    if attached_videos:
                        for vid_ref in attached_videos:
                            vid_src = vid_ref
                            if isinstance(vid_ref, str):
                                p = PROJECT_ROOT / vid_ref if not Path(vid_ref).is_absolute() else Path(vid_ref)
                                if p.exists():
                                    vid_src = str(p)
                            st.video(vid_src)
                    render_rich_content(display_text)

            with st.chat_message("assistant"):
                status_placeholder = st.empty()
                think_placeholder = st.empty()
                stream_placeholder = st.empty()
                status_placeholder.caption(
                    "👁️ Распознаю фото (qwen3-vl) и ищу в материалах…"
                    if attached_images
                    else "⚡ Подключаюсь к модели и ищу в материалах…"
                )
                last_ui_ts = [0.0]

                def _on_stream(kind: str, text_so_far: str, model_name: str) -> None:
                    now = time.perf_counter()
                    if kind != "status" and (now - last_ui_ts[0]) < 0.08:
                        return
                    last_ui_ts[0] = now
                    if kind == "status":
                        status_placeholder.caption(f"⏳ {text_so_far}")
                    elif kind == "thinking":
                        status_placeholder.caption(f"💭 `{model_name}` размышляет над задачей в реальном времени…")
                        with think_placeholder.expander("💭 Размышления модели (в реальном времени)…", expanded=True):
                            st.caption(text_so_far[-3000:])
                    elif kind in ("content", "token"):
                        status_placeholder.caption(f"✍️ `{model_name}` пишет ответ в реальном времени…")
                        think_placeholder.empty()
                        stream_placeholder.markdown(format_math_markdown(text_so_far) + " ▌")

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
                        on_stream=_on_stream,
                    )
                except LLMError as exc:
                    status_placeholder.empty()
                    think_placeholder.empty()
                    stream_placeholder.empty()
                    st.error(f"Ошибка обращения к модели: {exc}")
                else:
                    status_placeholder.empty()
                    think_placeholder.empty()
                    stream_placeholder.empty()
                    assistant_turn = {
                        "role": "assistant",
                        "content": answer.text,
                        "answer": answer,
                    }
                    st.session_state.chat_turns.append(assistant_turn)
                    # Если было распознано фото, сохраняем условие в соответствующий user-ход
                    if answer.extracted_image_text and len(st.session_state.chat_turns) >= 2:
                        prev_user = st.session_state.chat_turns[-2]
                        if prev_user.get("role") == "user":
                            prev_user["ocr_condition"] = answer.extracted_image_text
                    _save_active_chat_turns()
                    render_assistant_turn(assistant_turn, is_latest=True, turn_idx=len(st.session_state.chat_turns) - 1)


# ============================================================ 2. DISPACE НГТУ

with tabs[1]:
    st.subheader("🎓 Интеграция с DiSpace НГТУ (только чтение)")
    st.caption(
        "Все разделы личного кабинета студента НГТУ, структура курсов (включая непройденные темы и фрагменты), "
        "а также интерактивный разбор и решение заданий по материалам лекций с полным тетрадным оформлением."
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
            "📘 Мои курсы и прохождение",
            "🏛️ Дисциплины и журнал",
            "📅 События и вебинары",
            "🗂️ Каталог курсов",
            "🧪 Тесты и контрольные",
            "✉️ Сообщения",
        ],
        default="📘 Мои курсы и прохождение",
        label_visibility="collapsed",
    )

    if not dispace_api:
        st.info("Войди в DiSpace НГТУ кнопкой выше, чтобы просматривать разделы и скачивать курсы.")
    else:
        # ----------------------------------------------------- 2.1 Мои курсы
        if dispace_section == "📘 Мои курсы и прохождение":
            st.markdown(f"#### 📘 Мои курсы и прохождение тем/заданий ([открыть в DiSpace]({MY_COURSES_URL}))")
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
                    c_prog = _course_progress(cid)
                    if c_prog:
                        p_done, p_tot = c_prog.get("passed_themes", 0), c_prog.get("total_themes", 0)
                        p_icon = "✅" if p_tot and p_done >= p_tot else "⏳"
                        prog_summary = f" · {p_icon} тем пройдено: `{p_done}/{p_tot}`"
                    else:
                        prog_summary = ""
                    already_in_store = (
                        len(store.list_sources(c.get("discipline") or c_title)) > 0
                        or len(store.list_sources(c_title)) > 0
                    )
                    status_icon = "📚 В базе" if already_in_store else "📥 Не скачан"

                    with st.container(border=True):
                        st.markdown(f"**#{cid} — {c_title}** ({status_icon}){disc_badge}{prog_summary}")
                        st.caption(f"Ссылка: {c['url']}")

                        b_col1, b_col2 = st.columns([1, 1])
                        with b_col1:
                            if st.button(
                                "📥 Скачать весь курс в Материалы",
                                key=f"imp_course_{cid}",
                                type="primary",
                                width="stretch",
                            ):
                                target_course_name = c.get("discipline") or c_title
                                prog = st.progress(0.0, text=f"Чтение модулей, тем и заданий курса «{c_title}»…")

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
                                        f"Курс «{target_course_name}» (включая учебный план и задания) загружен в Материалы! "
                                        f"Источников: {len(saved)}."
                                    )
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Ошибка при импорте курса: {exc}")

                        with b_col2:
                            if st.button(
                                "📖 Темы, фрагменты и задания курса",
                                key=f"struct_course_{cid}",
                                width="stretch",
                            ):
                                st.session_state["inspect_course_id"] = cid
                                st.session_state["inspect_course_name"] = c.get("discipline") or c_title

            with col_mc2:
                st.markdown("**Импорт по прямой ссылке или ID**")
                with st.form("direct_dispace_import"):
                    direct_url = st.text_input(
                        "Ссылка на курс или тему DiSpace",
                        placeholder="https://dispace.nstu.ru/course/3121/32637",
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

            # --- Интерактивное прохождение курса: все модули, пройденные и непройденные темы, фрагменты и задания ---
            inspect_cid = st.session_state.get("inspect_course_id") or (my_courses[0]["id"] if my_courses else None)
            if inspect_cid:
                st.divider()
                try:
                    overview = dispace_api.get_course_overview(int(inspect_cid))
                except Exception as exc:
                    st.error(f"Не удалось загрузить обзор курса: {exc}")
                    overview = None

                if overview:
                    tot_th = max(overview["total_themes"], 1)
                    done_th = overview["passed_themes"]
                    st.markdown(
                        f"### 📖 Прохождение курса: **{overview['title']}** (`#{inspect_cid}`)"
                    )
                    st.progress(
                        done_th / tot_th,
                        text=f"Пройдено тем: {done_th} из {overview['total_themes']} "
                        f"(осталось пройти: {overview['total_themes'] - done_th})",
                    )

                    theme_filter = st.radio(
                        "Фильтр тем курса",
                        [
                            "Все темы курса (пройденные, начатые и закрытые)",
                            "Только непройденные и начатые темы",
                            "Только пройденные темы",
                        ],
                        horizontal=True,
                        key=f"th_filter_{inspect_cid}",
                        label_visibility="collapsed",
                    )

                    for mod in overview.get("modules") or []:
                        mod_score_str = (
                            f" · балл: **{mod['score']} / {mod['score_max']}**"
                            if mod.get("score_max") is not None
                            else ""
                        )
                        mod_icon = "✅" if mod.get("passed") else "⏳"
                        with st.expander(
                            f"{mod_icon} Модуль: {mod['title']}{mod_score_str}",
                            expanded=not mod.get("passed"),
                        ):
                            for th in mod.get("themes") or []:
                                if theme_filter.startswith("Только непройденные") and th["passed"]:
                                    continue
                                if theme_filter.startswith("Только пройденные") and not th["passed"]:
                                    continue
                                tid = th["id"]
                                sc_str = (
                                    f" · балл: `{_fmt_cell(th['score'])} / {_fmt_cell(th['score_max'])}`"
                                    if th.get("score_max") is not None
                                    else ""
                                )
                                t_col1, t_col2, t_col3 = st.columns([4, 1.4, 1])
                                with t_col1:
                                    st.markdown(
                                        f"{th['status_badge']} — **{th['title']}** (`#{tid}`){sc_str}"
                                    )
                                with t_col2:
                                    if st.button(
                                        "📂 Открыть теорию и задания",
                                        key=f"open_th_{inspect_cid}_{tid}",
                                        type="primary" if not th["passed"] and not th["locked"] else "secondary",
                                        width="stretch",
                                    ):
                                        st.session_state["active_theme_id"] = tid
                                        st.session_state["active_theme_course_id"] = int(inspect_cid)
                                        st.session_state["active_theme_module_title"] = mod["title"]
                                with t_col3:
                                    st.link_button("🌐 В DiSpace", th["url"], use_container_width=True)

                    active_tid = st.session_state.get("active_theme_id")
                    active_tcid = st.session_state.get("active_theme_course_id")
                    if active_tid and active_tcid == int(inspect_cid):
                        st.divider()
                        with st.spinner(f"Загружаю все фрагменты и задания темы #{active_tid}…"):
                            try:
                                th_detail = dispace_api.get_theme_detailed_blocks(int(inspect_cid), int(active_tid))
                            except Exception as exc:
                                st.error(f"Не удалось прочитать фрагменты темы: {exc}")
                                th_detail = None

                        if th_detail:
                            blocks = th_detail.get("blocks") or []
                            passed_blocks = sum(1 for b in blocks if b.get("passed"))
                            q_blocks = [b for b in blocks if b.get("content_type") in ("question", "task")]
                            theory_blocks = [b for b in blocks if b.get("content_type") in ("text", "video")]

                            st.markdown(
                                f"#### 📂 Тема: **{th_detail['title']}** "
                                f"(фрагментов: `{len(blocks)}`, пройдено: `{passed_blocks}/{len(blocks)}`, "
                                f"теория: `{len(theory_blocks)}`, заданий: `{len(q_blocks)}`)"
                            )

                            if th_detail.get("locked") and not blocks:
                                st.warning(
                                    "🔒 Эта тема пока закрыта сервером DiSpace: чтобы открыть её фрагменты и задания, "
                                    f"необходимо сначала завершить предыдущую тему (`#{th_detail.get('restriction_theme_id')}`) в курсе."
                                )
                            elif th_detail.get("test_id") and not blocks:
                                st.info(
                                    f"🧪 Это контрольный тест (`test_id={th_detail['test_id']}`). "
                                    "Нажми кнопку ниже, чтобы сгенерировать тренировочный вариант по этой теме в «Тренажёре»."
                                )
                                if st.button("🏋️ Сгенерировать тренировку по теме теста", key=f"tr_btn_{active_tid}", type="primary"):
                                    st.session_state["practice_prefill_topic"] = th_detail["title"]
                                    st.session_state["practice_prefill_course"] = st.session_state.get("inspect_course_name") or overview["title"]
                                    st.toast("Тема передана на вкладку «🧪 Тренажёр»!", icon="🧪")
                            else:
                                blk_mode = st.radio(
                                    "Отображать фрагменты темы",
                                    [
                                        f"📋 Все фрагменты по порядку ({len(blocks)})",
                                        f"✏️ Только практические задания ({len(q_blocks)})",
                                        f"📖 Только теоретические фрагменты ({len(theory_blocks)})",
                                    ],
                                    horizontal=True,
                                    key=f"blk_filter_{active_tid}",
                                    label_visibility="collapsed",
                                )

                                combined_theory_text = th_detail.get("combined_theory") or ""
                                course_rag_name = st.session_state.get("inspect_course_name") or overview["title"]

                                for b in blocks:
                                    b_type = b["content_type"]
                                    if blk_mode.startswith("✏️") and b_type not in ("question", "task"):
                                        continue
                                    if blk_mode.startswith("📖") and b_type not in ("text", "video"):
                                        continue

                                    b_idx = b["index"]
                                    b_id = b["id"]
                                    p_badge = "✅ Пройден" if b.get("passed") else "⏳ Не пройден"
                                    type_icon = "✏️" if b_type in ("question", "task") else ("🎬" if b_type == "video" else "📄")

                                    with st.container(border=True):
                                        st.markdown(f"##### {type_icon} Фрагмент {b_idx}: {b['title']} · *{p_badge}*")

                                        if b_type in ("text", "video"):
                                            render_rich_content(b.get("text") or "*(пустой фрагмент)*")

                                        elif b_type in ("question", "task"):
                                            q_data = b.get("question_data") or {}
                                            q_text = b.get("text") or "*(условие задания не получено)*"
                                            st.markdown("**📋 Условие задания:**")
                                            render_rich_content(q_text)

                                            opts = q_data.get("options") or []
                                            if opts:
                                                st.markdown("**Варианты ответа:**")
                                                for o_i, opt in enumerate(opts, 1):
                                                    render_rich_content(f"**{o_i}.** {opt['text']}")

                                            meta_bits = []
                                            if q_data.get("max_score") is not None:
                                                meta_bits.append(
                                                    f"Балл: `{_fmt_cell(q_data.get('score'))} / {_fmt_cell(q_data.get('max_score'))}`"
                                                )
                                            if q_data.get("student_answer"):
                                                meta_bits.append(f"Твой ответ в DiSpace: `{q_data['student_answer']}`")
                                            if q_data.get("remaining_attempts") is not None:
                                                meta_bits.append(f"Осталось попыток: `{q_data['remaining_attempts']}`")
                                            if meta_bits:
                                                st.caption(" · ".join(meta_bits))

                                            st.markdown("---")
                                            sol_c1, sol_c2 = st.columns([2, 1])
                                            with sol_c1:
                                                student_try = st.text_input(
                                                    "✍️ Твой ответ или ход решения (необязательно — для проверки твоего решения)",
                                                    value=q_data.get("student_answer") or "",
                                                    placeholder="Введи свой ответ (например: -1/2 или 18*x^(3)), чтобы бот проверил его…",
                                                    key=f"try_input_{active_tid}_{b_id}",
                                                )
                                            with sol_c2:
                                                st.write("")
                                                btn_check_own = st.button(
                                                    "🔍 Проверить мой ответ",
                                                    key=f"btn_chk_{active_tid}_{b_id}",
                                                    width="stretch",
                                                )

                                            btn_notebook = st.button(
                                                "📓 Решить с моделью (полное тетрадное оформление по материалам лекции)",
                                                key=f"btn_nb_{active_tid}_{b_id}",
                                                type="primary",
                                                width="stretch",
                                            )

                                            sol_state_key = f"saved_nb_sol_{active_tid}_{b_id}"
                                            if btn_check_own or btn_notebook:
                                                status_box = st.empty()
                                                think_box = st.empty()
                                                live_box = st.empty()
                                                last_ts = [0.0]

                                                def _on_nb_stream(kind: str, text_so_far: str, m_name: str) -> None:
                                                    now = time.perf_counter()
                                                    if kind != "status" and (now - last_ts[0]) < 0.08:
                                                        return
                                                    last_ts[0] = now
                                                    if kind == "thinking":
                                                        status_box.caption(f"💭 `{m_name}` просчитывает задание…")
                                                        with think_box.expander("💭 Размышления модели (в реальном времени)…", expanded=True):
                                                            st.caption(text_so_far[-2500:])
                                                    elif kind == "content":
                                                        status_box.caption(f"📓 `{m_name}` оформляет тетрадное решение…")
                                                        think_box.empty()
                                                        live_box.markdown(format_math_markdown(text_so_far) + " ▌")

                                                try:
                                                    nb_ans = get_tutor().solve_course_assignment(
                                                        question_text=q_text,
                                                        options=opts,
                                                        theme_theory_text=combined_theory_text,
                                                        course_name=course_rag_name,
                                                        student_attempt=student_try if btn_check_own else "",
                                                        on_stream=_on_nb_stream,
                                                    )
                                                    status_box.empty()
                                                    think_box.empty()
                                                    live_box.empty()
                                                    st.session_state[sol_state_key] = nb_ans
                                                except LLMError as exc:
                                                    status_box.empty()
                                                    think_box.empty()
                                                    live_box.empty()
                                                    st.error(f"Ошибка генерации решения: {exc}")

                                            if st.session_state.get(sol_state_key):
                                                saved_nb: Answer = st.session_state[sol_state_key]
                                                with st.container(border=True):
                                                    st.markdown("##### 📓 Тетрадное решение (на основе лекции курса)")
                                                    render_assistant_turn({"role": "assistant", "content": saved_nb.text, "answer": saved_nb})

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
        import re as _re_mat

        def _course_progress(cid: int) -> dict[str, Any] | None:
            """Прогресс курса из DiSpace (кэш в сессии пользователя на 10 минут)."""
            if not dispace_api:
                return None
            cache = st.session_state.setdefault("course_progress_cache", {})
            hit = cache.get(cid)
            if hit and time.time() - hit[0] < 600:
                return hit[1]
            try:
                ov = dispace_api.get_course_overview(int(cid))
            except Exception:
                ov = None
            cache[cid] = (time.time(), ov)
            return ov

        mat_hdr1, mat_hdr2 = st.columns([4, 1])
        with mat_hdr1:
            if dispace_api:
                st.caption("Отметки: ✅ пройдено · ⏳ в процессе · 🔓 доступно, не пройдено · 🔒 закрыто · 📎 файл/методичка")
            else:
                st.caption("Войдите в DiSpace, чтобы видеть отметки о прохождении тем.")
        with mat_hdr2:
            if dispace_api and st.button("🔄 Обновить прогресс", key="refresh_progress", width="stretch"):
                st.session_state["course_progress_cache"] = {}
                st.rerun()

        for course_name in store.list_courses():
            course_sources = store.list_sources(course_name)
            total_course_chars = sum(s.chars for s in course_sources)
            total_course_chunks = sum(len(s.chunk_ids) for s in course_sources)

            # Статусы тем по ссылкам вида /course/<cid>/<tid>
            theme_status: dict[tuple[int, int], dict[str, Any]] = {}
            course_ids = sorted(
                {
                    int(m.group(1))
                    for s in course_sources
                    if s.url and (m := _re_mat.search(r"/course/(\d+)", s.url))
                }
            )
            progress_bits: list[str] = []
            for cid in course_ids:
                ov = _course_progress(cid)
                if not ov:
                    continue
                for mod in ov.get("modules") or []:
                    for th in mod.get("themes") or []:
                        theme_status[(cid, int(th["id"]))] = th
                done, total = ov.get("passed_themes", 0), ov.get("total_themes", 0)
                icon = "✅" if total and done >= total else "⏳"
                progress_bits.append(f"{icon} пройдено тем {done}/{total}")
            progress_str = (" · " + " · ".join(progress_bits)) if progress_bits else ""

            with st.expander(
                f"📁 {course_name}{progress_str} — {len(course_sources)} ист. · {total_course_chunks} фр. · {total_course_chars:,} симв.".replace(
                    ",", " "
                ),
                expanded=True,
            ):
                for source in course_sources:
                    m_th = _re_mat.search(r"/course/(\d+)/(\d+)", source.url or "")
                    th_info = theme_status.get((int(m_th.group(1)), int(m_th.group(2)))) if m_th else None
                    if th_info:
                        src_badge = th_info.get("status_badge", "").split(" (")[0] + " · "
                    elif source.url and not m_th and "/course/" not in source.url:
                        src_badge = "📎 "
                    else:
                        src_badge = ""
                    with st.container(border=True):
                        cols = st.columns([4, 1, 1, 1])
                        with cols[0]:
                            st.markdown(
                                f"{src_badge}**{source.title}** — `{source.chars:,}` симв., `{len(source.chunk_ids)}` фр.".replace(
                                    ",", " "
                                )
                            )
                            if source.url:
                                st.caption(f"Источник: {source.url}")
                        with cols[1]:
                            if st.button("📖 Читалка", key=f"btn_dlg_{source.id}", width="stretch"):
                                open_material_reader_dialog(
                                    title=source.title,
                                    course=source.course,
                                    url=source.url,
                                    text=store.get_source_text(source.id),
                                )
                        with cols[2]:
                            preview_key = f"prev_toggle_{source.id}"
                            if st.button("👁️ Текст", key=f"btn_prev_{source.id}", width="stretch"):
                                st.session_state[preview_key] = not st.session_state.get(preview_key, False)
                        with cols[3]:
                            if st.button("🗑️ Удалить", key=f"del_{source.id}", width="stretch"):
                                store.remove_source(source.id)
                                store.reload()
                                get_tutor.clear()
                                st.rerun()

                        if st.session_state.get(f"prev_toggle_{source.id}", False):
                            full_text = clean_pdf_artifacts(store.get_source_text(source.id))
                            page_size = 8000
                            mat_pages = paginate_markdown(full_text, page_size=page_size)
                            total_pages = len(mat_pages)
                            page_key = f"mat_page_{source.id}"
                            cur_page = int(st.session_state.get(page_key, 1))
                            cur_page = max(1, min(cur_page, total_pages))

                            ctrl1, ctrl2, ctrl3 = st.columns([3, 2, 2])
                            with ctrl1:
                                view_mode = st.radio(
                                    "Режим просмотра",
                                    ["📐 Форматированный (формулы LaTeX, фото, видео)", "📄 Исходный текст"],
                                    horizontal=True,
                                    key=f"view_mode_{source.id}",
                                    label_visibility="collapsed",
                                )
                            with ctrl2:
                                show_all_pages = st.checkbox(
                                    f"Показать весь текст целиком ({len(full_text):,} симв.)".replace(",", " "),
                                    value=(total_pages == 1),
                                    key=f"show_all_{source.id}",
                                )
                            with ctrl3:
                                if not show_all_pages and total_pages > 1:
                                    nav_b1, nav_b2, nav_b3 = st.columns([1, 2, 1])
                                    with nav_b1:
                                        if st.button(
                                            "⬅️ Назад",
                                            key=f"prev_p_{source.id}",
                                            disabled=(cur_page <= 1),
                                            width="stretch",
                                        ):
                                            st.session_state[page_key] = cur_page - 1
                                            st.rerun()
                                    with nav_b2:
                                        st.caption(f"Стр. **{cur_page}** из **{total_pages}**")
                                    with nav_b3:
                                        if st.button(
                                            "Далее ➡️",
                                            key=f"next_p_{source.id}",
                                            disabled=(cur_page >= total_pages),
                                            width="stretch",
                                        ):
                                            st.session_state[page_key] = cur_page + 1
                                            st.rerun()

                            if show_all_pages:
                                slice_text = full_text
                            else:
                                slice_text = mat_pages[cur_page - 1]

                            if view_mode.startswith("📐"):
                                with st.container(border=True):
                                    render_rich_content(slice_text)
                            else:
                                st.text_area(
                                    "Содержимое источника",
                                    value=slice_text,
                                    height=320,
                                    disabled=True,
                                    key=f"ta_prev_{source.id}_{cur_page}_{show_all_pages}",
                                )

                            if not show_all_pages and total_pages > 1:
                                b_nav1, b_nav2, b_nav3 = st.columns([1, 2, 1])
                                with b_nav1:
                                    if st.button(
                                        "⬅️ Предыдущая страница",
                                        key=f"b_prev_p_{source.id}",
                                        disabled=(cur_page <= 1),
                                        width="stretch",
                                    ):
                                        st.session_state[page_key] = cur_page - 1
                                        st.rerun()
                                with b_nav2:
                                    st.caption(f"Страница {cur_page} из {total_pages} · всего {len(full_text):,} симв.".replace(",", " "))
                                with b_nav3:
                                    if st.button(
                                        "Следующая страница ➡️",
                                        key=f"b_next_p_{source.id}",
                                        disabled=(cur_page >= total_pages),
                                        width="stretch",
                                    ):
                                        st.session_state[page_key] = cur_page + 1
                                        st.rerun()


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
                    "Генерация идёт в реальном времени, каждый тезис получает ссылку `[#N]` на фрагмент."
                )

            note_path = store.notes_dir / slugify(selected_course) / f"{slugify(selected.title)}.md"

            if rebuild and require_auth("Сборка конспекта моделью", "notes"):
                builder = NoteBuilder(get_tutor().llm, store)
                progress = st.progress(0.0, text="Готовлю фрагменты…")
                live_note_box = st.empty()
                _last_note_ui = [0.0]

                def on_progress(position: int, total: int, chunk) -> None:
                    progress.progress(
                        position / max(total, 1),
                        text=f"✍️ Конспектирую в реальном времени: фрагмент {position} из {total}…",
                    )

                def on_note_stream(partial_md: str) -> None:
                    now = time.perf_counter()
                    if now - _last_note_ui[0] >= 0.08:
                        _last_note_ui[0] = now
                        live_note_box.markdown(format_math_markdown(partial_md) + " ▌")

                try:
                    result = builder.build(
                        selected,
                        on_progress=on_progress,
                        on_stream=on_note_stream,
                    )
                except (LLMError, ValueError) as exc:
                    st.error(f"Не удалось собрать конспект: {exc}")
                else:
                    live_note_box.empty()
                    progress.progress(1.0, text="Готово")
                    st.success(result.summary_line())
                    st.rerun()

            if note_path.exists():
                note_md = note_path.read_text(encoding="utf-8")
                dl_c1, dl_c2 = st.columns([1, 1])
                with dl_c1:
                    st.download_button(
                        "📥 Скачать конспект (.md)",
                        data=note_md,
                        file_name=note_path.name,
                        mime="text/markdown",
                        width="stretch",
                    )
                with dl_c2:
                    if st.button("📖 Открыть конспект в окне читалки", key="open_note_dlg", width="stretch"):
                        open_material_reader_dialog(
                            title=f"Конспект: {selected.title}",
                            course=selected_course,
                            url=selected.url,
                            text=note_md,
                        )
                with st.container(border=True):
                    render_rich_content(note_md)
            else:
                st.info("Конспект для этого материала ещё не собран. Нажми «Собрать конспект».")


# ============================================================== 5. ТРЕНАЖЁР

with tabs[4]:
    st.subheader("🧪 Тренажёр: генерация вариантов с самопроверкой и тетрадным решением")
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

        opt_c1, opt_c2 = st.columns(2)
        with opt_c1:
            verify_all = st.toggle(
                "Мгновенная математическая проверка эталонов (SymPy)",
                value=True,
            )
        with opt_c2:
            deep_think_practice = st.toggle(
                "Глубокие размышления (<think>, медленнее, для олимпиадных задач)",
                value=False,
            )
        generate = st.form_submit_button("⚡ Сгенерировать вариант (в реальном времени)", type="primary", width="stretch")

    if generate and topic.strip() and require_auth("Генерация заданий моделью", "practice"):
        stream_status = st.empty()
        stream_preview = st.empty()
        _last_pr_ui = [0.0]

        def _on_practice_stream(kind: str, text_so_far: str, model_name: str) -> None:
            now = time.perf_counter()
            if now - _last_pr_ui[0] < 0.08:
                return
            _last_pr_ui[0] = now
            if kind == "content" and text_so_far.strip():
                stream_status.caption(f"✍️ `{model_name}` генерирует условия и решения в реальном времени…")
                stream_preview.code(text_so_far[-1600:] + " ▌", language="json")
            elif kind == "thinking":
                stream_status.caption(f"💭 `{model_name}` размышляет ({len(text_so_far)} симв.)…")

        try:
            tasks, reports = get_tutor().practice(
                topic=topic.strip(),
                count=count,
                difficulty=difficulty,
                course=None if p_course == "Все курсы" else p_course,
                verify=verify_all,
                on_stream=_on_practice_stream,
                deep_think=deep_think_practice,
            )
        except LLMError as exc:
            st.error(f"Модель недоступна: {exc}")
            tasks, reports = [], []
        finally:
            stream_status.empty()
            stream_preview.empty()
        st.session_state.practice = (
            topic.strip(),
            tasks,
            reports,
            difficulty,
            None if p_course == "Все курсы" else p_course,
        )

    if st.session_state.get("practice"):
        prac_tuple = st.session_state.practice
        p_topic, tasks, reports, p_diff = prac_tuple[0], prac_tuple[1], prac_tuple[2], prac_tuple[3]
        p_selected_course = prac_tuple[4] if len(prac_tuple) > 4 else None
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

                user_prac_ans = st.text_input(
                    f"Твой ответ к заданию {i} (для самопроверки):",
                    key=f"prac_ans_{i}",
                    placeholder="Введи ответ или ход решения…",
                )
                pr_btn1, pr_btn2 = st.columns(2)
                with pr_btn1:
                    check_prac = st.button(
                        "🔍 Проверить мой ответ",
                        key=f"btn_prac_chk_{i}",
                        width="stretch",
                    )
                with pr_btn2:
                    notebook_prac = st.button(
                        "📓 Полное тетрадное решение по лекции",
                        key=f"btn_prac_nb_{i}",
                        type="primary",
                        width="stretch",
                    )

                prac_sol_key = f"prac_nb_sol_{i}"
                want_check = check_prac and user_prac_ans.strip()
                if (want_check or notebook_prac) and require_auth("Решение с моделью", f"prac_{i}"):
                    live_p_box = st.empty()
                    _last_p_t = [0.0]

                    def _on_p_stream(kind: str, text_so_far: str, model_name: str) -> None:
                        now = time.perf_counter()
                        if kind == "content" and now - _last_p_t[0] >= 0.08 and text_so_far.strip():
                            _last_p_t[0] = now
                            live_p_box.markdown(format_math_markdown(text_so_far) + " ▌")

                    try:
                        res_p = get_tutor().solve_course_assignment(
                            question_text=task_item["statement"],
                            theme_theory_text="",
                            course_name=p_selected_course,
                            student_attempt=user_prac_ans.strip() if want_check else "",
                            on_stream=_on_p_stream,
                        )
                        st.session_state[prac_sol_key] = res_p
                    except LLMError as exc:
                        st.error(f"Модель недоступна: {exc}")
                    finally:
                        live_p_box.empty()

                if st.session_state.get(prac_sol_key):
                    sol_turn = st.session_state[prac_sol_key]
                    with st.container(border=True):
                        render_rich_content(getattr(sol_turn, "text", "") or "")

                with st.expander("Показать краткое эталонное решение и ответ из генератора"):
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
