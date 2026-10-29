"""Чтение материалов с dispace.nstu.ru (Dicourse) через браузерную сессию.

Принципы:
  * бот НЕ знает пароль и не хранит его. Ты логинишься сам в открытом окне,
    дальше используется сохранённая сессия браузера;
  * только чтение: скрипт открывает страницы и снимает текст. Никаких отправок
    ответов, прохождения тестов и кликов по кнопкам сдачи;
  * сайт — SPA, поэтому берём отрисованный текст и попутно запоминаем JSON,
    который подгружает само приложение (это надёжнее, чем угадывать API).

Использование:
    python src/lms_reader.py --login                 # войти вручную, сохранить сессию
    python src/lms_reader.py --courses                # список курсов
    python src/lms_reader.py --read URL               # текст страницы в консоль
    python src/lms_reader.py --import URL --course "Линейная алгебра"
    python src/lms_reader.py --import-course URL --course "Линейная алгебра"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_URL = "https://dispace.nstu.ru"
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
  .map(a => ({href: a.href, text: (a.innerText || '').trim().slice(0, 200)}))
  .filter(x => x.text.length > 2)
"""


def _session_bootstrap_js(data: dict[str, dict[str, str]]) -> str:
    """Скрипт, который кладёт сохранённый sessionStorage до старта приложения."""
    payload = json.dumps(data, ensure_ascii=False)
    return (
        "(function(){try{"
        f"const data={payload};"
        "const origin=location.origin;"
        "const bucket=data[origin];"
        "if(bucket){for(const key in bucket){"
        "try{sessionStorage.setItem(key,bucket[key]);}catch(e){}}"
        "}catch(e){}})();"
    )


@dataclass
class PageContent:
    url: str
    title: str
    text: str
    links: list[dict[str, str]] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.text) > 200


class LoginRequired(RuntimeError):
    pass


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
        # sessionStorage живёт только внутри вкладки и не попадает в storage_state.
        # Для SPA с JWT в sessionStorage сохраняем его отдельно.
        self.session_file = self.state_file.with_name("browser_session_storage.json")
        self.headless = headless
        self.slow_mo = slow_mo
        self._playwright = None
        self._context = None
        self._payloads: list[dict[str, Any]] = []
        self._capture = False

    # ------------------------------------------------------------- жизненный цикл

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

        # Восстанавливаем sessionStorage до загрузки скриптов приложения:
        # если токен лежит там, SPA увидит его при старте.
        session_state = self._load_session_state()
        if session_state:
            self._context.add_init_script(script=_session_bootstrap_js(session_state))

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

    # --------------------------------------------------------------- сеть

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

    # ------------------------------------------------------------- навигация

    def open(self, url: str = BASE_URL) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        self._settle()

    def _settle(self, extra: float = 1.2) -> None:
        try:
            self.page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        time.sleep(extra)

    def is_logged_in(self) -> bool:
        """Проверить вход. SPA может хранить токен в localStorage, а не в cookie."""
        if not self._context:
            return False
        return any(self._page_is_logged_in(page) for page in self._context.pages)

    def _page_is_logged_in(self, page) -> bool:
        from urllib.parse import urlparse

        try:
            url = page.url
            host = urlparse(url).netloc.lower()
            path = urlparse(url).path or "/"
            has_password = page.locator('input[type="password"]').count() > 0
        except Exception:
            return False

        # Форма пароля на экране — точно не вошли.
        if has_password:
            return False

        # Вход через SSO идёт на другом домене. Токен там ничего не доказывает:
        # сессии диспейса ещё может не быть. Ждём возврата именно на диспейс.
        if "dispace.nstu.ru" not in host:
            return False
        if path.startswith("/login"):
            return False

        # После входа SPA уводит на личные разделы.
        if re.search(r"^/(courses|course|lk|profile|dashboard)", path):
            return True

        try:
            storage = page.evaluate("() => ({...localStorage, ...sessionStorage})")
        except Exception:
            storage = {}
        storage_keys = [
            key
            for key in (storage or {})
            if any(marker in key.lower() for marker in ("token", "jwt", "session", "auth"))
        ]
        if storage_keys:
            return True

        cookies = self._context.cookies() if self._context else []
        return any(
            any(marker in cookie["name"].lower() for marker in ("token", "jwt", "session", "auth"))
            for cookie in cookies
        )

    def wait_for_login(self, timeout: int = 300) -> bool:
        """Ждать, пока пользователь войдёт руками. Возвращает True при успехе."""
        print("\n" + "=" * 70)
        print("Открылось окно браузера. Войди в диспейс вручную (логин и пароль).")
        print("Пароль вводится только в браузере — скрипт его не видит и не хранит.")
        print("Ожидание входа: до", timeout // 60, "мин.")
        print("=" * 70 + "\n")
        started = time.time()
        while time.time() - started < timeout:
            if self.is_logged_in():
                print("Вход выполнен, сессия сохранена.")
                self.save_state()
                return True
            time.sleep(3)
        print("Время ожидания истекло — вход не выполнен.")
        return False

    def save_state(self) -> Path:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        if self._context:
            self._context.storage_state(path=str(self.state_file))
            self._save_session_state()
        return self.state_file

    def _save_session_state(self) -> None:
        """Сохранить sessionStorage по источникам (для SPA с токеном во вкладке)."""
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

    # ------------------------------------------------------------- контент

    def _evaluate(self, expression: str, retries: int = 4) -> Any:
        """Выполнить JS, переживая переходы SPA между маршрутами.

        После входа приложение может ещё раз перейти с /login на /courses/my.
        В этот момент контекст страницы уничтожается, и обычный evaluate
        падает с «Execution context was destroyed». Повторяем после паузы.
        """
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
                title, text = self._evaluate(
                    "() => [document.title, document.body ? document.body.innerText : '']"
                )
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

    def list_courses(self) -> list[dict[str, str]]:
        self.open(f"{BASE_URL}/courses/my")
        links = self._evaluate(LINKS_JS)
        courses: dict[str, dict[str, str]] = {}
        for link in links:
            href = link["href"]
            if re.search(r"/course/[^/?#]+", href):
                courses.setdefault(href, {"url": href, "title": link["text"]})
        return sorted(courses.values(), key=lambda c: c["title"])

    def course_materials(self, course_url: str) -> list[dict[str, str]]:
        self.open(course_url)
        links = self._evaluate(LINKS_JS)
        base = course_url.rstrip("/")
        found: dict[str, dict[str, str]] = {}
        for link in links:
            href = link["href"]
            if not _same_site(href) or href.rstrip("/") == base:
                continue
            if re.search(r"/(course|test|task|lesson)", href):
                found.setdefault(href, {"url": href, "title": link["text"]})
        return sorted(found.values(), key=lambda c: c["title"])


def _clean_text(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.strip() for line in text.splitlines()).strip()


def _same_site(url: str) -> bool:
    return "dispace" in url and "nstu.ru" in url


# --------------------------------------------------------------------- CLI


def _store():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from materials import MaterialStore

    return MaterialStore()


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Чтение материалов диспейса (только чтение)")
    parser.add_argument("--login", action="store_true", help="открыть браузер и войти вручную")
    parser.add_argument("--courses", action="store_true", help="список курсов")
    parser.add_argument("--links", metavar="URL", help="показать ссылки страницы (отладка навигации)")
    parser.add_argument("--read", metavar="URL", help="прочитать страницу")
    parser.add_argument("--import", dest="import_url", metavar="URL", help="прочитать и сохранить в хранилище")
    parser.add_argument("--import-course", metavar="URL", help="сохранить все материалы курса")
    parser.add_argument("--course", help="название курса для сохранения")
    parser.add_argument("--title", help="название материала")
    parser.add_argument("--headless", action="store_true", help="без окна браузера (нужна готовая сессия)")
    parser.add_argument("--timeout", type=int, default=300, help="ожидание входа, сек")
    args = parser.parse_args(argv)

    with Reader(headless=args.headless) as reader:
        if args.login:
            reader.open()
            if not reader.wait_for_login(timeout=args.timeout):
                return 2
            print(f"Сессия: {reader.state_file}")

            # Сразу проверяем, что сессия рабочая, и заодно показываем курсы.
            try:
                courses = reader.list_courses()
            except Exception as exc:
                print(f"Не удалось прочитать список курсов: {exc}")
                courses = []
            if courses:
                print("\nКурсы:")
                for course in courses:
                    print(f"  {course['title']}\n    {course['url']}")
            else:
                print(
                    "\nКурсы не распознаны. Открой страницу курсов и запусти:\n"
                    "  python src/lms_reader.py --links \"<URL страницы>\""
                )
            reader.save_state()
            return 0

        if args.courses:
            for course in reader.list_courses():
                print(f"  {course['title']}\n    {course['url']}")
            return 0

        if args.read:
            content = reader.read(args.read)
            print(f"{content.title}\n{content.url}\n{'-' * 60}")
            print(content.text[:5000])
            print(f"{'-' * 60}\nсимволов: {len(content.text)}, ссылок: {len(content.links)}")
            if "/login" in content.url or len(content.text) < 200:
                print(
                    "\nПохоже, нужен вход. Выполни: python src/lms_reader.py --login\n"
                    "затем повтори чтение (сессия сохранится)."
                )
            return 0 if content.ok else 1

        if args.links:
            content = reader.read(args.links)
            print(f"{content.title}\n{content.url}\n{'-' * 60}")
            for link in content.links:
                print(f"  {(link['text'] or '—')[:60]:60} {link['href']}")
            print(f"{'-' * 60}\nссылок: {len(content.links)}")
            if "/login" in content.url:
                print("Страница входа: сначала выполни --login.")
            return 0 if content.links else 1

        if args.import_url or args.import_course:
            store = _store()
            if not args.course:
                parser.error("для импорта нужен --course")

            if args.import_url:
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
                print(f"сохранено: [{source.id}] {source.title}, "
                      f"{source.chars} симв., {len(source.chunk_ids)} фрагментов")
                return 0

            materials = reader.course_materials(args.import_course)
            print(f"найдено материалов: {len(materials)}")
            if not materials:
                print("ссылки не распознаны. Проверь вручную через --read и передай URL.")
                return 1
            for item in materials:
                try:
                    content = reader.read(item["url"])
                except Exception as exc:
                    print(f"  ! {item['title']}: {exc}")
                    continue
                if not content.ok:
                    print(f"  - пропуск (мало текста): {item['title']}")
                    continue
                source = store.add_source(
                    course=args.course,
                    title=item["title"] or content.title or "Материал",
                    text=content.text,
                    url=item["url"],
                )
                print(f"  + [{source.id}] {source.title}: {len(source.chunk_ids)} фрагментов")
            return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
