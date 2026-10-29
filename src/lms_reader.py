"""Чтение материалов с dispace.nstu.ru (Dicourse) через браузерную сессию и read-only API.

Принципы:
  * бот НЕ знает пароль и не хранит его. Ты логинишься сам в открытом окне,
    дальше используется сохранённая сессия браузера (токен authToken в localStorage
    и SSO-cookie Keycloak в профиле браузера);
  * только чтение: скрипт забирает учебные тексты, формулы LaTeX и прикреплённые
    методички (.pdf, .docx). Никаких отправок ответов, прохождения тестов и кликов
    по кнопкам сдачи;
  * двойной контур: быстрый REST-клиент DiSpace (api/v1/view/...) с сохранением
    исходного LaTeX ($...$) и автоматическим резервным чтением через Playwright.

Использование:
    python src/lms_reader.py --login                 # войти вручную, сохранить сессию
    python src/lms_reader.py --courses               # список курсов
    python src/lms_reader.py --disciplines           # дисциплины и контрольные (/education)
    python src/lms_reader.py --events                # расписание вебинаров и событий (/events)
    python src/lms_reader.py --read URL              # текст страницы/курса/темы в консоль
    python src/lms_reader.py --links URL             # структура тем/ссылок страницы или курса
    python src/lms_reader.py --import URL --course "Линейная алгебра"
    python src/lms_reader.py --import-course 3095 --course "Линейная алгебра"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ingest import (  # noqa: E402
    decode_tex_cp1251,
    extract_docx_bytes,
    extract_pdf_bytes,
    html_to_text,
)

BASE_URL = "https://dispace.nstu.ru"
MY_COURSES_URL = f"{BASE_URL}/courses/my?filterMode=all&offset=0&sortBy=created_at&sortOrder=desc"
EDUCATION_URL = f"{BASE_URL}/education"
EVENTS_URL = f"{BASE_URL}/events"
CATALOG_URL = f"{BASE_URL}/courses?offset=0"
TESTS_URL = f"{BASE_URL}/tests?offset=0&sortBy=test_end_time&sortOrder=desc"
MESSAGES_URL = f"{BASE_URL}/messages"

DEFAULT_PROFILE = PROJECT_ROOT / "data" / "browser_profile"
DEFAULT_STATE = PROJECT_ROOT / "data" / "browser_state.json"

# Собираем текст из основного содержимого, а не из навигации.
TEXT_JS = r"""
() => {
  const bad = ['NAV','HEADER','FOOTER','ASIDE','SCRIPT','STYLE','NOSCRIPT','SVG','BUTTON'];
  const root = document.querySelector('main') || document.querySelector('[role=main]')
            || document.body;
  const clone = root.cloneNode(true);
  clone.querySelectorAll(bad.join(',')).forEach(el => el.remove());
  const text = clone.innerText || '';
  return {
    title: document.title,
    text: text.replace(/\n{3,}/g, '\n\n').trim(),
    url: location.href,
  };
}
"""

LINKS_JS = r"""
() => Array.from(document.querySelectorAll('a[href]'))
  .map(a => ({
    href: a.href,
    text: ((a.innerText || a.getAttribute('aria-label') || '').trim()).slice(0, 200)
  }))
  .filter(x => x.text.length > 1)
"""


def _session_bootstrap_js(
    session_data: dict[str, dict[str, str]],
    local_data: dict[str, dict[str, str]] | None = None,
) -> str:
    """Скрипт, восстанавливающий localStorage (authToken) и sessionStorage до старта SPA."""
    s_payload = json.dumps(session_data or {}, ensure_ascii=False)
    l_payload = json.dumps(local_data or {}, ensure_ascii=False)
    return (
        "(function(){try{"
        f"const sData={s_payload};const lData={l_payload};"
        "const origin=location.origin;"
        "const sBucket=sData[origin];"
        "if(sBucket){for(const k in sBucket){try{sessionStorage.setItem(k,sBucket[k]);}catch(e){}}}"
        "const lBucket=lData[origin];"
        "if(lBucket){for(const k in lBucket){try{if(!localStorage.getItem(k)){localStorage.setItem(k,lBucket[k]);}}catch(e){}}}"
        "}catch(e){}})();"
    )


def load_auth_token(state_file: Path = DEFAULT_STATE) -> str | None:
    """Извлечь JWT-токен DiSpace (ключ authToken в localStorage) из browser_state.json."""
    if not state_file.exists():
        return None
    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    for origin_entry in state.get("origins") or []:
        if "dispace.nstu.ru" not in str(origin_entry.get("origin", "")):
            continue
        for item in origin_entry.get("localStorage") or []:
            if item.get("name") == "authToken":
                raw_val = item.get("value")
                if not raw_val or raw_val == "null":
                    return None
                try:
                    parsed = json.loads(raw_val)
                    return str(parsed) if parsed else None
                except json.JSONDecodeError:
                    return str(raw_val).strip('"')
    return None


@dataclass
class MaterialItem:
    title: str
    text: str
    url: str
    kind: str = "lecture"  # lecture | pdf | docx | task


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    links: list[dict[str, str]] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.text) > 100


class LoginRequired(RuntimeError):
    pass


# --------------------------------------------------------- Read-Only REST API


class DiSpaceAPI:
    """Клиент только для чтения учебных материалов DiSpace по сохранённому токену."""

    def __init__(self, token: str, base_url: str = BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) DIBot-StudyReader/1.0",
            }
        )

    def verify_token(self) -> bool:
        try:
            resp = self.session.post(
                f"{self.base_url}/api/v1/auth/verify",
                json={"token": self.token},
                timeout=10,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        resp = self.session.get(url, params=params, timeout=20)
        if resp.status_code == 401:
            raise LoginRequired("Сессия DiSpace истекла (401). Выполни: python src/lms_reader.py --login")
        resp.raise_for_status()
        return resp.json()

    def list_courses(self) -> list[dict[str, Any]]:
        """Получить список курсов из 'Мои курсы' и прикреплённых к дисциплинам."""
        courses_by_id: dict[int, dict[str, Any]] = {}

        try:
            my_data = self._get(
                "api/v1/view/courses",
                params={
                    "limit": 50,
                    "offset": 0,
                    "sort_by": "created_at",
                    "sort_order": "desc",
                    "mode": "my_courses",
                },
            )
            for item in (my_data or {}).get("data") or []:
                cid = item.get("id")
                if not cid:
                    continue
                courses_by_id[int(cid)] = {
                    "id": int(cid),
                    "title": str(item.get("title") or f"Курс #{cid}").strip(),
                    "url": f"{self.base_url}/course/{cid}",
                    "status": item.get("status") or "",
                    "discipline": "",
                }
        except requests.RequestException:
            pass

        try:
            disciplines = self._get("api/v1/view/lms/disciplines") or []
            for disc in disciplines:
                course = disc.get("course")
                if isinstance(course, dict) and course.get("id"):
                    cid = int(course["id"])
                    entry = courses_by_id.setdefault(
                        cid,
                        {
                            "id": cid,
                            "title": str(course.get("title") or f"Курс #{cid}").strip(),
                            "url": f"{self.base_url}/course/{cid}",
                            "status": course.get("status") or "published",
                            "discipline": "",
                        },
                    )
                    entry["discipline"] = str(disc.get("title") or "").strip()
        except requests.RequestException:
            pass

        return sorted(courses_by_id.values(), key=lambda c: c["title"])

    def list_disciplines(self) -> list[dict[str, Any]]:
        """Список учебных дисциплин студента (/education) и контрольных мероприятий."""
        raw = self._get("api/v1/view/lms/disciplines") or []
        result: list[dict[str, Any]] = []
        for item in raw:
            did = item.get("id")
            etype = item.get("entity_type") or ("discipline" if item.get("is_custom_group") else "stream")
            course = item.get("course") or {}
            lectors = [
                f"{p.get('last_name', '')} {p.get('first_name', '')[:1]}.{p.get('patronymic', '')[:1]}.".strip()
                for p in (item.get("lectors") or []) + (item.get("teachers") or [])
                if isinstance(p, dict) and p.get("last_name")
            ]
            control_events: list[dict[str, Any]] = []
            if did:
                try:
                    ce_list = self._get(f"api/v1/lms/control_events/student/discipline/{did}") or []
                    for ce in ce_list:
                        if isinstance(ce, dict):
                            control_events.append(
                                {
                                    "id": ce.get("id"),
                                    "name": ce.get("name") or "Задание",
                                    "description": html_to_text(ce.get("description") or ""),
                                    "files": ce.get("files") or [],
                                    "score": ce.get("score"),
                                    "ects_limit": ce.get("ects_limit"),
                                }
                            )
                except requests.RequestException:
                    pass

            result.append(
                {
                    "id": did,
                    "title": str(item.get("title") or "").strip(),
                    "entity_type": etype,
                    "url": f"{self.base_url}/education/{etype}/{did}/description",
                    "semesters": item.get("semesters") or [],
                    "course_id": course.get("id"),
                    "course_title": course.get("title"),
                    "course_url": f"{self.base_url}/course/{course['id']}" if course.get("id") else "",
                    "lectors": list(dict.fromkeys(lectors)),
                    "control_events": control_events,
                }
            )
        return result

    def list_events(self, month: int | None = None, year: int | None = None) -> list[dict[str, Any]]:
        """Расписание событий, онлайн-лекций и вебинаров (/events)."""
        now = dt.datetime.now()
        m = month or now.month
        y = year or now.year
        raw = self._get("api/v1/lms/events", params={"month": m, "year": y, "tz_offset": 7}) or {}
        events: list[dict[str, Any]] = []
        if isinstance(raw, dict):
            for day_key, day_items in raw.items():
                if str(day_key).startswith(("-", "+")) or not isinstance(day_items, list):
                    continue
                for ev in day_items:
                    if not isinstance(ev, dict):
                        continue
                    events.append(
                        {
                            "day": f"{int(day_key):02d}.{m:02d}.{y}",
                            "title": ev.get("title") or ev.get("name") or "Событие",
                            "type": ev.get("type") or "",
                            "time": ev.get("start_time") or ev.get("time") or "",
                            "discipline": ev.get("discipline_title") or "",
                            "url": ev.get("join_url") or ev.get("url") or f"{self.base_url}/events",
                        }
                    )
        return events

    def get_course(self, course_id: int) -> dict[str, Any]:
        return self._get(f"api/v1/view/course/{course_id}", params={"from_catalog": "false"})

    def get_course_scores(self, course_id: int) -> dict[str, Any]:
        """Получить баллы и прогресс по модулям/темам курса (/api/v1/view/course/{id}/scores)."""
        try:
            return self._get(f"api/v1/view/course/{course_id}/scores") or {}
        except requests.RequestException:
            return {}

    def get_discipline_journal(self, discipline_id: int) -> list[dict[str, Any]]:
        """Получить журнал оценок дисциплины (/api/v1/view/lms/journal/{id})."""
        try:
            raw = self._get(f"api/v1/view/lms/journal/{discipline_id}") or {}
        except requests.RequestException:
            return []
        columns = raw.get("columns") if isinstance(raw, dict) else []
        out: list[dict[str, Any]] = []
        for col in columns or []:
            if not isinstance(col, dict):
                continue
            marks = col.get("student_marks") or []
            m0 = marks[0] if marks and isinstance(marks[0], dict) else {}
            out.append(
                {
                    "id": col.get("id"),
                    "title": str(col.get("title") or "Контроль").strip(),
                    "type": str(col.get("type") or ""),
                    "ects_limit": col.get("ects_limit"),
                    "ects_mark": m0.get("ects_mark"),
                    "text_mark": m0.get("text_mark"),
                    "date": col.get("date") or "",
                }
            )
        return out

    def list_webinar_records(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Список записей прошедших вебинаров (/api/v1/lms/records)."""
        try:
            raw = self._get("api/v1/lms/records", params={"limit": limit, "offset": offset}) or {}
        except requests.RequestException:
            return []
        items = raw.get("data") if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
        out: list[dict[str, Any]] = []
        for r in items or []:
            if not isinstance(r, dict):
                continue
            out.append(
                {
                    "id": r.get("id"),
                    "title": str(r.get("title") or r.get("name") or "Запись вебинара").strip(),
                    "date": str(r.get("date") or r.get("created_at") or ""),
                    "url": str(r.get("url") or r.get("record_url") or ""),
                    "discipline": str(r.get("discipline_title") or ""),
                }
            )
        return out

    def list_catalog_courses(
        self,
        search: str = "",
        limit: int = 24,
        offset: int = 0,
        favorite_only: bool = False,
    ) -> dict[str, Any]:
        """Каталог курсов НГТУ (/courses?offset=0 -> /api/v1/catalog/courses)."""
        q = search.strip().lower()
        fetch_limit = 200 if q else limit
        params: dict[str, Any] = {
            "favorite_only": "true" if favorite_only else "false",
            "limit": fetch_limit,
            "offset": 0 if q else offset,
        }

        raw = self._get("api/v1/catalog/courses", params=params) or {}
        items = (raw.get("data") if isinstance(raw, dict) else []) or []
        total = int(
            (
                raw.get("pagination")
                or raw.get("count")
                or len(items)
            )
            if isinstance(raw, dict)
            else len(items)
        )

        courses: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            cid = int(item["id"])
            title = str(item.get("title") or f"Курс #{cid}").strip()
            desc = html_to_text(str(item.get("description") or ""))
            authors: list[str] = []
            for entry in (item.get("coauthors") or item.get("authors") or []):
                if not isinstance(entry, dict):
                    continue
                person = entry.get("author") if isinstance(entry.get("author"), dict) else entry
                if person.get("last_name"):
                    authors.append(
                        f"{person.get('last_name', '')} {person.get('first_name', '')[:1]}.{person.get('patronymic', '')[:1]}.".strip()
                    )
            haystack = f"{title} {desc} {' '.join(authors)} {cid}".lower()
            if q and q not in haystack:
                continue
            courses.append(
                {
                    "id": cid,
                    "title": title,
                    "description": desc[:400],
                    "authors": authors,
                    "weeks": item.get("count_study_weeks"),
                    "education_form": str(item.get("education_form") or ""),
                    "publication_type": str(item.get("publication_type") or ""),
                    "url": f"{self.base_url}/course/{cid}/description",
                }
            )
        return {"count": total, "courses": courses[:limit]}

    def list_tests(self) -> dict[str, Any]:
        """Список назначенных тестов (/tests) и тестовых тем внутри курсов студента."""
        assigned: list[dict[str, Any]] = []
        try:
            raw = self._get(
                "api/v1/test/assigntest/assignings",
                params={
                    "sort_by": "test_end_time",
                    "sort_order": "desc",
                    "offset": 0,
                    "limit": 50,
                },
            ) or {}
            for item in (raw.get("response_assignings") if isinstance(raw, dict) else []) or []:
                if not isinstance(item, dict):
                    continue
                assigned.append(
                    {
                        "id": item.get("id") or item.get("assigning_id"),
                        "test_id": item.get("test_id"),
                        "title": str(item.get("test_title") or item.get("title") or "Тест").strip(),
                        "start_time": str(item.get("test_start_time") or ""),
                        "end_time": str(item.get("test_end_time") or ""),
                        "score": item.get("score") or item.get("result"),
                        "status": str(item.get("status") or ""),
                        "url": f"{self.base_url}/tests",
                    }
                )
        except requests.RequestException:
            pass

        course_tests: list[dict[str, Any]] = []
        for c in self.list_courses():
            cid = c["id"]
            try:
                c_data = self.get_course(cid)
            except requests.RequestException:
                continue
            c_title = str(c_data.get("title") or c["title"]).strip()
            for mod in c_data.get("modules") or []:
                if mod.get("hidden"):
                    continue
                m_title = str(mod.get("title") or "").strip()
                for th in mod.get("themes") or []:
                    if th.get("hidden"):
                        continue
                    test_id = th.get("test_id")
                    th_type = str(th.get("type") or "")
                    th_title = str(th.get("title") or "").strip()
                    if test_id or th_type == "test" or "тест" in th_title.lower() or "контрольн" in th_title.lower():
                        tid = th.get("id")
                        course_tests.append(
                            {
                                "course_id": cid,
                                "course_title": c_title,
                                "module_title": m_title,
                                "theme_id": tid,
                                "theme_title": th_title,
                                "test_id": test_id,
                                "type": "тест" if (test_id or th_type == "test") else "контрольная/задание",
                                "url": f"{self.base_url}/course/{cid}/{tid}" if tid else f"{self.base_url}/course/{cid}",
                            }
                        )

        return {"assigned_tests": assigned, "course_tests": course_tests}

    def list_messages(self, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """Список диалогов в сообщениях (/messages -> /api/v1/lms/messages)."""
        raw = self._get("api/v1/lms/messages", params={"limit": limit, "offset": offset}) or {}
        items = (raw.get("data") if isinstance(raw, dict) else []) or []
        dialogues: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            d_settings = item.get("dialogue_settings") or {}
            conv = d_settings.get("conversant") or item.get("sender") or {}
            conv_id = conv.get("id")
            conv_name = (
                f"{conv.get('last_name', '')} {conv.get('first_name', '')} {conv.get('patronymic', '')}".strip()
                or f"Собеседник #{conv_id}"
            )
            content = item.get("content") or {}
            msg_text = html_to_text(str(content.get("message_text") or ""))
            files = content.get("files") or []
            dialogues.append(
                {
                    "id": item.get("id"),
                    "conversant_id": int(conv_id) if conv_id else None,
                    "conversant_name": conv_name,
                    "conversant_role": "Преподаватель" if conv.get("is_teacher") else "Пользователь",
                    "last_message": msg_text,
                    "date": str(content.get("date") or ""),
                    "theme": str(content.get("theme") or ""),
                    "count_unread": int(item.get("count_unread") or 0),
                    "files": files,
                    "url": f"{self.base_url}/messages",
                }
            )
        return dialogues

    def get_dialogue(self, conversant_id: int, limit: int = 30, offset: int = 0) -> list[dict[str, Any]]:
        """Сообщения конкретного диалога (/api/v1/lms/messages/dialogue)."""
        raw = (
            self._get(
                "api/v1/lms/messages/dialogue",
                params={"conversant_id": conversant_id, "limit": limit, "offset": offset},
            )
            or {}
        )
        items = (raw.get("data") if isinstance(raw, dict) else []) or []
        messages: list[dict[str, Any]] = []
        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            sender = item.get("sender") or {}
            sender_name = (
                f"{sender.get('last_name', '')} {sender.get('first_name', '')[:1]}.{sender.get('patronymic', '')[:1]}.".strip()
                or "Отправитель"
            )
            content = item.get("content") or {}
            text = html_to_text(str(content.get("message_text") or ""))
            messages.append(
                {
                    "id": item.get("id"),
                    "sender_id": sender.get("id"),
                    "sender_name": sender_name,
                    "is_teacher": bool(sender.get("is_teacher")),
                    "text": text,
                    "date": str(content.get("date") or ""),
                    "theme": str(content.get("theme") or ""),
                    "files": content.get("files") or [],
                }
            )
        return messages

    def import_course_to_store(
        self,
        store: Any,
        course_id: int,
        course_name: str | None = None,
        include_files: bool = True,
        on_progress: Any = None,
    ) -> list[Any]:
        """Скачать все модули, темы и файлы курса и сохранить напрямую в MaterialStore."""
        detected_title, items = self.extract_course_materials(course_id, include_files=include_files)
        target_course = (course_name or detected_title).strip()
        saved_sources: list[Any] = []
        total = len(items)
        for idx, it in enumerate(items, start=1):
            if on_progress:
                try:
                    on_progress(idx, total, it.title)
                except Exception:
                    pass
            source = store.add_source(
                course=target_course,
                title=it.title,
                text=it.text,
                url=it.url,
            )
            saved_sources.append(source)
        store.reload()
        return saved_sources

    def import_control_event_to_store(
        self,
        store: Any,
        discipline_title: str,
        control_event: dict[str, Any],
    ) -> list[Any]:
        """Скачать условие и прикреплённые файлы контрольного мероприятия в MaterialStore."""
        saved_sources: list[Any] = []
        ce_name = str(control_event.get("name") or "Контрольное задание").strip()
        ce_desc = str(control_event.get("description") or "").strip()
        if len(ce_desc) >= 30:
            s = store.add_source(
                course=discipline_title,
                title=f"{ce_name} — описание",
                text=ce_desc,
                url=f"{self.base_url}/education",
            )
            saved_sources.append(s)
        for f_info in control_event.get("files") or []:
            f_text = self._download_file_text(f_info)
            if len(f_text) >= 40:
                f_title = str(f_info.get("title") or "Файл").strip()
                s = store.add_source(
                    course=discipline_title,
                    title=f"{ce_name}: {f_title}",
                    text=f_text,
                    url=str(f_info.get("url") or f"{self.base_url}/education"),
                )
                saved_sources.append(s)
        store.reload()
        return saved_sources

    def get_theme(self, theme_id: int) -> dict[str, Any]:
        return self._get(f"api/v1/view/course/theme/{theme_id}")

    def get_block(self, block_id: int) -> dict[str, Any]:
        return self._get(f"api/v1/view/course/block/{block_id}")

    def _download_file_text(self, file_info: dict[str, Any]) -> str:
        """Скачать прикреплённый файл (.pdf, .docx, .txt, .md) и извлечь текст."""
        f_url = file_info.get("url")
        f_title = str(file_info.get("title") or "")
        f_type = str(file_info.get("file_type") or "").lower()
        if not f_url:
            return ""
        ext = Path(f_title).suffix.lower() or (f".{f_type}" if f_type else "")
        if ext not in {".pdf", ".docx", ".txt", ".md"}:
            return ""
        try:
            resp = requests.get(f_url, timeout=45)
            resp.raise_for_status()
            data = resp.content
        except requests.RequestException:
            return ""

        try:
            if ext == ".pdf":
                return extract_pdf_bytes(data)
            if ext == ".docx":
                return extract_docx_bytes(data)
            for enc in ("utf-8", "utf-8-sig", "cp1251"):
                try:
                    return decode_tex_cp1251(data.decode(enc)).strip()
                except UnicodeDecodeError:
                    continue
        except Exception:
            return ""
        return ""

    def extract_theme_materials(
        self,
        course_id: int,
        theme_id: int,
        module_title: str = "",
        theme_title: str = "",
        include_files: bool = True,
    ) -> list[MaterialItem]:
        """Собрать все текстовые блоки (с LaTeX) и прикреплённые файлы одной темы."""
        theme = self.get_theme(theme_id)
        t_title = theme_title or str(theme.get("title") or f"Тема {theme_id}").strip()
        prefix = f"{module_title} — {t_title}" if module_title else t_title
        theme_url = f"{self.base_url}/course/{course_id}/{theme_id}"

        text_parts: list[str] = []
        if theme.get("description"):
            desc = html_to_text(str(theme["description"]))
            if desc:
                text_parts.append(desc)

        file_items: list[MaterialItem] = []
        seen_file_urls: set[str] = set()

        for block_stub in theme.get("blocks") or []:
            bid = block_stub.get("id")
            ctype = block_stub.get("content_type")
            if not bid:
                continue
            try:
                block = self.get_block(int(bid))
            except requests.RequestException:
                block = block_stub

            if ctype == "text" or block.get("text_page"):
                tp = block.get("text_page") or {}
                b_title = str(tp.get("title") or "").strip()
                b_html = str(tp.get("content") or "")
                b_text = html_to_text(b_html)
                if b_text:
                    header = f"### {b_title}\n\n" if b_title and b_title.lower() != t_title.lower() else ""
                    text_parts.append(f"{header}{b_text}")

            elif ctype == "task" or block.get("task"):
                task = block.get("task") or {}
                t_name = str(task.get("title") or "Задание").strip()
                t_body = html_to_text(str(task.get("content") or task.get("description") or ""))
                if t_body:
                    text_parts.append(f"### Задание: {t_name}\n\n{t_body}")

            if include_files:
                for f_info in block.get("files") or []:
                    f_url = str(f_info.get("url") or "")
                    if not f_url or f_url in seen_file_urls:
                        continue
                    seen_file_urls.add(f_url)
                    f_text = self._download_file_text(f_info)
                    if len(f_text) >= 60:
                        f_title = str(f_info.get("title") or "Файл").strip()
                        file_items.append(
                            MaterialItem(
                                title=f"{prefix}: {f_title}",
                                text=f_text,
                                url=f_url,
                                kind=str(f_info.get("file_type") or "file"),
                            )
                        )

        items: list[MaterialItem] = []
        combined_text = "\n\n".join(text_parts).strip()
        if len(combined_text) >= 40:
            items.append(
                MaterialItem(
                    title=prefix,
                    text=combined_text,
                    url=theme_url,
                    kind="lecture",
                )
            )
        items.extend(file_items)
        return items

    def extract_course_materials(
        self, course_id: int, include_files: bool = True
    ) -> tuple[str, list[MaterialItem]]:
        """Собрать все модули, темы и файлы курса. Возвращает (название_курса, список_материалов)."""
        course = self.get_course(course_id)
        course_title = str(course.get("title") or f"Курс {course_id}").strip()
        all_items: list[MaterialItem] = []
        seen_urls: set[str] = set()

        # Описание курса, если содержательное
        desc_parts = [
            html_to_text(str(course.get(field) or ""))
            for field in ("description", "goals", "thematic", "requirements")
        ]
        desc_text = "\n\n".join(p for p in desc_parts if p).strip()
        if len(desc_text) >= 120:
            all_items.append(
                MaterialItem(
                    title=f"{course_title} — Описание курса",
                    text=desc_text,
                    url=f"{self.base_url}/course/{course_id}/description",
                    kind="lecture",
                )
            )

        for mod in course.get("modules") or []:
            if mod.get("hidden"):
                continue
            mod_title = str(mod.get("title") or "").strip()
            for theme in mod.get("themes") or []:
                if theme.get("hidden"):
                    continue
                tid = theme.get("id")
                if not tid:
                    continue
                theme_items = self.extract_theme_materials(
                    course_id=course_id,
                    theme_id=int(tid),
                    module_title=mod_title,
                    theme_title=str(theme.get("title") or "").strip(),
                    include_files=include_files,
                )
                for item in theme_items:
                    # Дедупликация одинаковых файлов, прикреплённых к разным блокам курса
                    dedup_key = (
                        f"file:{item.title.split(':', 1)[-1].strip()}:{len(item.text)}"
                        if item.kind != "lecture"
                        else item.url
                    )
                    if dedup_key in seen_urls:
                        continue
                    seen_urls.add(dedup_key)
                    all_items.append(item)

        return course_title, all_items


# ------------------------------------------------------- Разбор URL DiSpace


def parse_dispace_target(target: str) -> dict[str, Any]:
    """Распознать, на что указывает ссылка или ID: курс, тема, дисциплина или обычная страница."""
    raw = target.strip()
    if raw.isdigit():
        return {"kind": "course", "course_id": int(raw)}

    parsed = urlparse(raw)
    path = parsed.path or "/"

    m_theme = re.search(r"^/course/(\d+)/(\d+)(?:/(\d+))?", path)
    if m_theme:
        return {
            "kind": "theme",
            "course_id": int(m_theme.group(1)),
            "theme_id": int(m_theme.group(2)),
            "block_id": int(m_theme.group(3)) if m_theme.group(3) else None,
        }

    m_course = re.search(r"^/course/(\d+)", path)
    if m_course:
        return {"kind": "course", "course_id": int(m_course.group(1))}

    m_edu = re.search(r"^/education/([^/]+)/(\d+)", path)
    if m_edu:
        return {
            "kind": "discipline",
            "entity_type": m_edu.group(1),
            "discipline_id": int(m_edu.group(2)),
        }

    return {"kind": "page", "url": raw if raw.startswith("http") else f"{BASE_URL}/{raw.lstrip('/')}"}


# ----------------------------------------------------- Браузерный контур


class Reader:
    def __init__(
        self,
        profile_dir: str | Path | None = None,
        state_file: str | Path | None = None,
        headless: bool = False,
        slow_mo: int = 0,
    ):
        self.profile_dir = Path(profile_dir) if profile_dir else DEFAULT_PROFILE
        self.state_file = Path(state_file) if state_file else DEFAULT_STATE
        self.session_file = self.state_file.with_name("browser_session_storage.json")
        self.headless = headless
        self.slow_mo = slow_mo
        self._playwright = None
        self._context = None
        self._payloads: list[dict[str, Any]] = []
        self._capture = False

    def start(self) -> None:
        from playwright.sync_api import sync_playwright

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            slow_mo=self.slow_mo,
            locale="ru-RU",
            viewport={"width": 1440, "height": 900},
        )
        self._context.on("response", self._on_response)

        session_state = self._load_session_state()
        local_state = self._load_local_state()
        if session_state or local_state:
            self._context.add_init_script(
                script=_session_bootstrap_js(session_state, local_state)
            )

    def close(self) -> None:
        if self._context:
            self._context.close()
        if self._playwright:
            self._playwright.stop()
        self._context = None
        self._playwright = None

    def __enter__(self) -> "Reader":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @property
    def page(self):
        if not self._context:
            raise RuntimeError("Reader не запущен: вызовите start()")
        pages = self._context.pages
        return pages[0] if pages else self._context.new_page()

    def _on_response(self, response) -> None:
        if not self._capture:
            return
        try:
            content_type = response.headers.get("content-type", "")
            if "json" not in content_type:
                return
            if int(response.headers.get("content-length") or 0) > 2_000_000:
                return
            data = response.json()
        except Exception:
            return
        if isinstance(data, (dict, list)):
            self._payloads.append({"url": response.url, "data": data})

    def open(self, url: str = MY_COURSES_URL) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        self._settle()
        self._wait_for_oauth_callback()

    def _settle(self, extra: float = 1.2) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        time.sleep(extra)

    def _wait_for_oauth_callback(self, timeout: float = 20.0) -> None:
        """Если URL содержит ?code=... или session_state=..., дождаться обмена кода на JWT."""
        deadline = time.time() + timeout
        saw_callback = False
        while time.time() < deadline:
            try:
                cur_url = self.page.url
            except Exception:
                break
            parsed = urlparse(cur_url)
            qs = parse_qs(parsed.query)
            if "dispace.nstu.ru" in parsed.netloc.lower() and ("code" in qs or "session_state" in qs):
                saw_callback = True
                time.sleep(1.0)
                continue
            if saw_callback:
                self._settle(1.0)
                self.save_state()
            break

    def try_sso_refresh(self, timeout: float = 20.0) -> bool:
        """Попробовать бесшовно обновить authToken через сохранённую SSO-сессию Keycloak."""
        try:
            self.open(MY_COURSES_URL)
        except Exception:
            return False

        if self.is_logged_in():
            self.save_state()
            return True

        try:
            if "/login" in self.page.url:
                btn = self.page.get_by_text("Войти по логину НГТУ")
                if btn.count() > 0:
                    btn.first.click()
                    deadline = time.time() + timeout
                    while time.time() < deadline:
                        time.sleep(1.0)
                        self._wait_for_oauth_callback(timeout=5.0)
                        if self.is_logged_in():
                            self.save_state()
                            return True
                        # Если открылась форма пароля на login2.nstu.ru — нужен ручной вход
                        if "login2.nstu.ru" in self.page.url:
                            if self.page.locator('input[type="password"]').count() > 0:
                                return False
        except Exception:
            return False

        return self.is_logged_in()

    def is_logged_in(self) -> bool:
        """Проверить вход: должен быть валидный authToken в localStorage на dispace.nstu.ru."""
        if not self._context:
            return False
        return any(self._page_is_logged_in(page) for page in self._context.pages)

    def _page_is_logged_in(self, page) -> bool:
        try:
            url = page.url
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            path = parsed.path or "/"
            qs = parse_qs(parsed.query)
            has_password = page.locator('input[type="password"]').count() > 0
        except Exception:
            return False

        if has_password:
            return False
        if "dispace.nstu.ru" not in host:
            return False
        if path.startswith("/login"):
            return False
        # Пока в URL висит одноразовый code=... от Keycloak, обмен на токен ещё не завершён!
        if "code" in qs or "session_state" in qs:
            return False

        try:
            auth_token = page.evaluate("() => localStorage.getItem('authToken')")
        except Exception:
            auth_token = None

        if auth_token and str(auth_token).strip() not in ("", "null", '""'):
            return True
        return False

    def wait_for_login(self, timeout: int = 300) -> bool:
        """Ждать, пока пользователь войдёт руками (или сработает SSO). Возвращает True при успехе."""
        if self.try_sso_refresh(timeout=12.0):
            print("Вход выполнен автоматически по сохранённой SSO-сессии НГТУ.")
            return True

        print("\n" + "=" * 70)
        print("Открылось окно браузера. Войди в диспейс вручную (логин и пароль).")
        print("Пароль вводится только в браузере — скрипт его не видит и не хранит.")
        print("Ожидание входа: до", timeout // 60, "мин.")
        print("=" * 70 + "\n")
        started = time.time()
        while time.time() - started < timeout:
            self._wait_for_oauth_callback(timeout=5.0)
            if self.is_logged_in():
                time.sleep(1.0)
                self.save_state()
                print("Вход выполнен, токен authToken и сессия сохранены.")
                return True
            time.sleep(2.0)
        print("Время ожидания истекло — вход не выполнен.")
        return False

    def save_state(self) -> Path:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self._context:
            self._context.storage_state(path=str(self.state_file))
            self._save_session_state()
        return self.state_file

    def _save_session_state(self) -> None:
        session: dict[str, dict[str, str]] = {}
        for page in self._context.pages if self._context else []:
            try:
                origin = page.evaluate("() => location.origin")
            except Exception:
                continue
            if not str(origin).startswith("http"):
                continue
            try:
                data = page.evaluate(
                    "() => Object.fromEntries(Object.entries(sessionStorage))"
                )
            except Exception:
                continue
            if data:
                session[str(origin)] = {str(k): str(v) for k, v in data.items()}
        if not session:
            return
        self.session_file.write_text(
            json.dumps(session, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load_session_state(self) -> dict[str, dict[str, str]]:
        if not self.session_file.exists():
            return {}
        try:
            data = json.loads(self.session_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return data if isinstance(data, dict) else {}

    def _load_local_state(self) -> dict[str, dict[str, str]]:
        if not self.state_file.exists():
            return {}
        try:
            state = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        result: dict[str, dict[str, str]] = {}
        for origin_entry in state.get("origins") or []:
            origin = str(origin_entry.get("origin") or "")
            items = origin_entry.get("localStorage") or []
            if origin and items:
                result[origin] = {
                    str(i["name"]): str(i["value"])
                    for i in items
                    if isinstance(i, dict) and "name" in i and "value" in i
                }
        return result

    def _evaluate(self, expression: str, retries: int = 4) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                return self.page.evaluate(expression)
            except Exception as exc:
                message = str(exc).lower()
                if (
                    "execution context was destroyed" not in message
                    and "navigat" not in message
                ):
                    raise
                last_error = exc
                time.sleep(1.0 + attempt)
                try:
                    self.page.wait_for_load_state("domcontentloaded", timeout=15_000)
                except Exception:
                    pass
        if last_error:
            raise last_error
        raise RuntimeError("не удалось выполнить JS на странице")

    def read(self, url: str) -> PageContent:
        self._payloads.clear()
        self._capture = True
        try:
            self.open(url)
            try:
                info = self._evaluate(TEXT_JS)
                title = info.get("title", "")
                text = info.get("text", "")
            except Exception:
                title, text = "", ""
            links = self._evaluate(LINKS_JS)
            content = PageContent(
                url=self.page.url,
                title=str(title).strip(),
                text=_clean_text(str(text)),
                links=[link for link in links if _same_site(link["href"])],
                payloads=list(self._payloads),
            )
            return content
        finally:
            self._capture = False


def get_api_client(
    state_file: Path = DEFAULT_STATE,
    profile_dir: Path = DEFAULT_PROFILE,
    auto_refresh: bool = True,
) -> DiSpaceAPI | None:
    """Получить готовый REST-клиент DiSpace (с автообновлением токена через SSO при необходимости)."""
    token = load_auth_token(state_file)
    if token:
        api = DiSpaceAPI(token)
        if api.verify_token():
            return api

    if not auto_refresh:
        return None

    # Попытка тихого обновления токена через сохранённый профиль браузера
    try:
        with Reader(profile_dir=profile_dir, state_file=state_file, headless=True) as reader:
            if reader.try_sso_refresh(timeout=15.0):
                new_token = load_auth_token(state_file)
                if new_token:
                    api = DiSpaceAPI(new_token)
                    if api.verify_token():
                        return api
    except Exception:
        pass
    return None


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _same_site(url: str) -> bool:
    return "dispace" in url and "nstu.ru" in url


# --------------------------------------------------------------------- CLI


def _store():
    from materials import MaterialStore

    return MaterialStore()


def _build_notes_for_sources(store, source_ids: list[str]) -> None:
    from llm_client import LLMClient, LLMError
    from notes import NoteBuilder, verify_citations

    llm = LLMClient()
    builder = NoteBuilder(llm, store)
    for sid in source_ids:
        source = store.sources.get(sid)
        if not source:
            continue
        print(f"\nСтрою конспект для: {source.title}...")

        def progress(position: int, total: int, chunk) -> None:
            print(f"    фрагмент {position}/{total}", flush=True)

        try:
            result = builder.build(source, on_progress=progress)
        except (LLMError, ValueError) as exc:
            print(f"  ! конспект не построен: {exc}")
            continue

        valid = {c.index + 1 for c in store.chunks.values() if c.source_id == sid}
        _, bad = verify_citations(result.text, valid)
        print(f"  -> {result.path}")
        print(f"     {result.summary_line()}")
        if bad:
            print(f"     ВНИМАНИЕ: ссылки на несуществующие фрагменты: {bad}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Чтение материалов диспейса (только чтение)")
    parser.add_argument("--login", action="store_true", help="открыть браузер и войти вручную")
    parser.add_argument("--courses", action="store_true", help="список курсов (/courses/my и /education)")
    parser.add_argument("--disciplines", action="store_true", help="список дисциплин и заданий (/education)")
    parser.add_argument("--journal", type=int, metavar="DISC_ID", help="журнал оценок дисциплины (/education)")
    parser.add_argument("--events", action="store_true", help="расписание вебинаров и лекций (/events)")
    parser.add_argument("--catalog", nargs="?", const="", metavar="QUERY", help="каталог курсов НГТУ (/courses)")
    parser.add_argument("--tests", action="store_true", help="назначенные тесты и тесты в курсах (/tests)")
    parser.add_argument("--messages", action="store_true", help="диалоги и сообщения (/messages)")
    parser.add_argument("--links", metavar="URL", help="показать ссылки или структуру курса/страницы")
    parser.add_argument("--read", metavar="URL", help="прочитать страницу, курс или тему")
    parser.add_argument("--import", dest="import_url", metavar="URL", help="прочитать и сохранить в хранилище")
    parser.add_argument("--import-course", metavar="URL_OR_ID", help="сохранить все материалы и файлы курса")
    parser.add_argument("--course", help="название курса для сохранения")
    parser.add_argument("--title", help="название материала")
    parser.add_argument("--notes", action="store_true", help="сразу построить конспекты после импорта")
    parser.add_argument("--headless", action="store_true", help="без окна браузера (нужна готовая сессия)")
    parser.add_argument("--timeout", type=int, default=300, help="ожидание входа, сек")
    args = parser.parse_args(argv)

    # 1. Вход в браузере
    if args.login:
        with Reader(headless=args.headless) as reader:
            reader.open(MY_COURSES_URL)
            if not reader.wait_for_login(timeout=args.timeout):
                return 2
            reader.save_state()
            print(f"Сессия: {reader.state_file}")

        api = get_api_client(auto_refresh=False)
        if api:
            courses = api.list_courses()
            if courses:
                print(f"\nНайдено курсов ({len(courses)}):")
                for c in courses:
                    disc_note = f"  [дисциплина: {c['discipline']}]" if c.get("discipline") else ""
                    print(f"  • #{c['id']} {c['title']}{disc_note}\n    {c['url']}")
            disciplines = api.list_disciplines()
            if disciplines:
                print(f"\nДисциплины семестра ({len(disciplines)}):")
                for d in disciplines:
                    c_info = f" -> курс #{d['course_id']} ({d['course_title']})" if d.get("course_id") else " (без курса)"
                    print(f"  • {d['title']}{c_info}\n    {d['url']}")
        return 0

    # Для остальных операций сначала пробуем быстрый REST API
    api = get_api_client(auto_refresh=True)

    # 2. Список курсов
    if args.courses:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        courses = api.list_courses()
        if not courses:
            print("Курсы не найдены.")
            return 1
        print(f"Мои курсы ({len(courses)}):")
        for c in courses:
            disc_note = f"  [дисциплина: {c['discipline']}]" if c.get("discipline") else ""
            print(f"  #{c['id']}  {c['title']}{disc_note}\n    {c['url']}")
        return 0

    # 3. Список дисциплин (/education)
    if args.disciplines:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        disciplines = api.list_disciplines()
        if not disciplines:
            print("Дисциплины не найдены.")
            return 1
        print(f"Учебный процесс — дисциплины ({len(disciplines)}):")
        for d in disciplines:
            lect = f" ({', '.join(d['lectors'])})" if d.get("lectors") else ""
            c_info = f"\n    Прикреплённый курс: #{d['course_id']} «{d['course_title']}» ({d['course_url']})" if d.get("course_id") else "\n    Курс не прикреплён"
            ce_count = len(d.get("control_events") or [])
            ce_info = f"\n    Контрольных мероприятий: {ce_count}" if ce_count else ""
            print(f"  • [{d['id']}] {d['title']}{lect}\n    {d['url']}{c_info}{ce_info}")
            for ce in d.get("control_events") or []:
                files_str = ", ".join(f.get("title", "") for f in (ce.get("files") or []))
                f_note = f" [файлы: {files_str}]" if files_str else ""
                print(f"      - {ce['name']}{f_note}")
        return 0

    # 4. События и вебинары (/events)
    if args.events:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        events = api.list_events()
        records = api.list_webinar_records()
        if not events and not records:
            print("В текущем месяце событий и записей вебинаров не найдено.")
            return 0
        if events:
            print(f"События текущего месяца ({len(events)}):")
            for ev in events:
                t_str = f" {ev['time']}" if ev.get("time") else ""
                d_str = f" [{ev['discipline']}]" if ev.get("discipline") else ""
                print(f"  • {ev['day']}{t_str}{d_str}: {ev['title']}")
        if records:
            print(f"\nЗаписи вебинаров ({len(records)}):")
            for r in records:
                print(f"  • {r['title']} ({r['date']}) {r['url']}")
        return 0

    # 4b. Журнал оценок дисциплины (--journal)
    if args.journal:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        cols = api.get_discipline_journal(args.journal)
        if not cols:
            print(f"Журнал дисциплины #{args.journal} пуст или недоступен.")
            return 0
        print(f"Журнал оценок дисциплины #{args.journal} ({len(cols)} колонок):")
        for col in cols:
            mark = col.get("text_mark") or col.get("ects_mark") or "—"
            lim = f" (макс. {col['ects_limit']})" if col.get("ects_limit") is not None else ""
            print(f"  • {col['title']}{lim}: {mark}")
        return 0

    # 4c. Каталог курсов (--catalog)
    if args.catalog is not None:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        cat = api.list_catalog_courses(search=args.catalog, limit=20)
        courses = cat.get("courses") or []
        print(f"Каталог курсов НГТУ (всего в базе: {cat.get('count', 0)}, показано: {len(courses)}):")
        for c in courses:
            auth = f" ({', '.join(c['authors'])})" if c.get("authors") else ""
            print(f"  • #{c['id']} {c['title']}{auth}\n    {c['url']}")
        return 0

    # 4d. Тесты и контрольные (--tests)
    if args.tests:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        t_data = api.list_tests()
        assigned = t_data.get("assigned_tests") or []
        c_tests = t_data.get("course_tests") or []
        print(f"Назначенные сессии тестирования (/tests): {len(assigned)}")
        for t in assigned:
            print(f"  • {t['title']} [{t['status']}] балл: {t.get('score') or '—'}")
        print(f"\nТесты и контрольные внутри курсов студента: {len(c_tests)}")
        for ct in c_tests:
            print(f"  • [{ct['course_title']}] {ct['module_title']} -> {ct['theme_title']} ({ct['type']})\n    {ct['url']}")
        return 0

    # 4e. Сообщения и диалоги (--messages)
    if args.messages:
        if not api:
            print("Нужен вход. Выполни: python src/lms_reader.py --login")
            return 1
        dialogs = api.list_messages(limit=20)
        if not dialogs:
            print("Диалогов не найдено.")
            return 0
        print(f"Диалоги в DiSpace ({len(dialogs)}):")
        for d in dialogs:
            unread = f" [непрочитано: {d['count_unread']}]" if d.get("count_unread") else ""
            snippet = (d.get("last_message") or "").replace("\n", " ")[:90]
            print(f"  • [{d['conversant_id']}] {d['conversant_name']} ({d['conversant_role']}){unread}")
            print(f"    {d['date']}: {snippet}")
        return 0

    # 5. Ссылки / структура курса или страницы (--links)
    if args.links:
        target = parse_dispace_target(args.links)
        if api and target["kind"] == "course":
            course = api.get_course(target["course_id"])
            cid = course.get("id")
            print(f"{course.get('title')} (#{cid})\n{BASE_URL}/course/{cid}\n{'-' * 60}")
            count = 0
            for mod in course.get("modules") or []:
                if mod.get("hidden"):
                    continue
                print(f"\nМодуль: {mod.get('title')}")
                for th in mod.get("themes") or []:
                    if th.get("hidden"):
                        continue
                    tid = th.get("id")
                    t_url = f"{BASE_URL}/course/{cid}/{tid}"
                    print(f"  {str(th.get('title') or '')[:55]:55} {t_url}")
                    count += 1
            print(f"{'-' * 60}\nтем: {count}")
            return 0 if count else 1

        if api and args.links.rstrip("/").endswith("/courses/my") or "courses/my" in args.links:
            courses = api.list_courses()
            print(f"Мои курсы | DiSpace\n{MY_COURSES_URL}\n{'-' * 60}")
            for c in courses:
                print(f"  {c['title'][:55]:55} {c['url']}")
            print(f"{'-' * 60}\nкурсов: {len(courses)}")
            return 0 if courses else 1

        with Reader(headless=args.headless) as reader:
            content = reader.read(args.links)
            print(f"{content.title}\n{content.url}\n{'-' * 60}")
            for link in content.links:
                print(f"  {(link['text'] or '—')[:60]:60} {link['href']}")
            print(f"{'-' * 60}\nссылок: {len(content.links)}")
            return 0 if content.links else 1

    # 6. Чтение страницы / темы / курса (--read)
    if args.read:
        target = parse_dispace_target(args.read)
        if api and target["kind"] == "theme":
            items = api.extract_theme_materials(target["course_id"], target["theme_id"])
            if items:
                combined = "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(
                    f"# {it.title}\nИсточник: {it.url}\n\n{it.text}" for it in items
                )
                print(combined[:6000])
                print(f"\n{'-' * 60}\nматериалов: {len(items)}, символов: {sum(len(i.text) for i in items)}")
                return 0

        if api and target["kind"] == "course":
            c_title, items = api.extract_course_materials(target["course_id"], include_files=False)
            print(f"{c_title}\n{BASE_URL}/course/{target['course_id']}\n{'-' * 60}")
            for it in items:
                print(f"\n## {it.title} ({len(it.text)} симв.)\n{it.text[:600]}...")
            print(f"\n{'-' * 60}\nтем с текстом: {len(items)}")
            return 0 if items else 1

        with Reader(headless=args.headless) as reader:
            content = reader.read(args.read)
            print(f"{content.title}\n{content.url}\n{'-' * 60}")
            print(content.text[:5000])
            print(f"{'-' * 60}\nсимволов: {len(content.text)}, ссылок: {len(content.links)}")
            if "/login" in content.url or len(content.text) < 100:
                print(
                    "\nПохоже, нужен вход. Выполни: python src/lms_reader.py --login\n"
                    "затем повтори чтение (сессия сохранится)."
                )
            return 0 if content.ok else 1

    # 7. Импорт материала или всего курса в локальную базу знаний
    if args.import_url or args.import_course:
        store = _store()
        saved_ids: list[str] = []

        if args.import_url:
            target = parse_dispace_target(args.import_url)
            if api and target["kind"] == "theme":
                course_obj = api.get_course(target["course_id"])
                course_name = args.course or str(course_obj.get("title") or "Курс").strip()
                items = api.extract_theme_materials(target["course_id"], target["theme_id"])
                for it in items:
                    source = store.add_source(
                        course=course_name,
                        title=args.title or it.title,
                        text=it.text,
                        url=it.url,
                    )
                    saved_ids.append(source.id)
                    print(f"  + [{source.id}] {source.title}: {source.chars} симв., {len(source.chunk_ids)} фр.")
            elif api and target["kind"] == "course":
                args.import_course = str(target["course_id"])
            else:
                if not args.course:
                    parser.error("для импорта произвольной страницы нужен --course")
                with Reader(headless=args.headless) as reader:
                    content = reader.read(args.import_url)
                if not content.ok:
                    print("страница пустая или не загрузилась — проверь вход и ссылку")
                    return 1
                source = store.add_source(
                    course=args.course,
                    title=args.title or content.title or "Материал",
                    text=content.text,
                    url=content.url,
                )
                saved_ids.append(source.id)
                print(
                    f"сохранено: [{source.id}] {source.title}, "
                    f"{source.chars} симв., {len(source.chunk_ids)} фрагментов"
                )

        if args.import_course:
            if not api:
                print("Нужен вход в DiSpace. Выполни: python src/lms_reader.py --login")
                return 1
            target = parse_dispace_target(args.import_course)
            course_id: int | None = None
            discipline_name: str = ""

            if target["kind"] == "course":
                course_id = target["course_id"]
            elif target["kind"] == "theme":
                course_id = target["course_id"]
            elif target["kind"] == "discipline":
                disc_list = api.list_disciplines()
                for d in disc_list:
                    if d["id"] == target["discipline_id"]:
                        discipline_name = d["title"]
                        course_id = d.get("course_id")
                        break

            if not course_id:
                print("Не удалось определить ID курса по переданной ссылке.")
                return 1

            detected_title, items = api.extract_course_materials(course_id, include_files=True)
            course_name = args.course or discipline_name or detected_title
            print(f"Курс: «{course_name}» (#{course_id}) — найдено материалов: {len(items)}")
            if not items:
                print("Материалы не найдены или курс пуст.")
                return 1

            for it in items:
                source = store.add_source(
                    course=course_name,
                    title=it.title,
                    text=it.text,
                    url=it.url,
                )
                saved_ids.append(source.id)
                print(
                    f"  + [{source.id}] {source.title} ({it.kind}): "
                    f"{source.chars} симв., {len(source.chunk_ids)} фрагментов"
                )

        if args.notes and saved_ids:
            _build_notes_for_sources(store, saved_ids)

        return 0 if saved_ids else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
