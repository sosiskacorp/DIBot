"""Веб-поиск: SearXNG как основной бэкенд, DuckDuckGo как резервный.

SearXNG отдаёт JSON только если в его settings.yml включено
`search.formats: [html, json]`. Если сервер не поднят или JSON закрыт,
клиент честно переключается на резерв и сообщает, чем отвечал.
"""

from __future__ import annotations

import html
import json
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


class SearchError(RuntimeError):
    pass


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    engine: str = ""
    published: str = ""

    def as_context(self, index: int) -> str:
        return f"[web:{index}] {self.title}\n{self.url}\n{self.snippet}"


class SearXNG:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 10.0,
        language: str = "ru",
        categories: str = "general",
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.language = language
        self.categories = categories
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    def health(self) -> tuple[bool, str]:
        try:
            resp = self.session.get(
                f"{self.base_url}/search",
                params={"q": "test", "format": "json"},
                timeout=min(self.timeout, 6),
            )
        except requests.RequestException as exc:
            return False, f"нет связи: {exc.__class__.__name__}"
        if resp.status_code == 403:
            return False, "JSON-формат закрыт в settings.yml (403)"
        if resp.status_code >= 400:
            return False, f"HTTP {resp.status_code}"
        try:
            resp.json()
        except ValueError:
            return False, "ответ не JSON — вероятно, format: json не включён"
        return True, "ok"

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        try:
            resp = self.session.get(
                f"{self.base_url}/search",
                params={
                    "q": query,
                    "format": "json",
                    "language": self.language,
                    "categories": self.categories,
                },
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise SearchError(f"SearXNG недоступен: {exc}") from exc

        if resp.status_code == 403:
            raise SearchError("SearXNG отвечает 403: включите json в search.formats")
        if resp.status_code >= 400:
            raise SearchError(f"SearXNG: HTTP {resp.status_code}")

        try:
            data = resp.json()
        except ValueError as exc:
            raise SearchError("SearXNG вернул не JSON") from exc

        results = []
        for item in data.get("results", [])[:limit]:
            results.append(
                SearchResult(
                    title=item.get("title", "").strip(),
                    url=item.get("url", ""),
                    snippet=(item.get("content") or "").strip(),
                    engine=", ".join(item.get("engines", []) or []),
                    published=item.get("publishedDate") or "",
                )
            )
        return results


class DuckDuckGo:
    """Резерв без API-ключей: HTML-выдача."""

    ENDPOINT = "https://html.duckduckgo.com/html/"

    def __init__(self, timeout: float = 12.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def search(self, query: str, limit: int = 8) -> list[SearchResult]:
        try:
            resp = self.session.post(
                self.ENDPOINT, data={"q": query, "kl": "ru-ru"}, timeout=self.timeout
            )
        except requests.RequestException as exc:
            raise SearchError(f"DuckDuckGo недоступен: {exc}") from exc

        if resp.status_code >= 400:
            raise SearchError(f"DuckDuckGo: HTTP {resp.status_code}")

        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[SearchResult] = []
        for node in soup.select(".result"):
            link = node.select_one("a.result__a")
            if not link:
                continue
            url = _clean_ddg_url(link.get("href", ""))
            if not url:
                continue
            snippet_node = node.select_one(".result__snippet")
            results.append(
                SearchResult(
                    title=link.get_text(" ", strip=True),
                    url=url,
                    snippet=snippet_node.get_text(" ", strip=True) if snippet_node else "",
                    engine="duckduckgo",
                )
            )
            if len(results) >= limit:
                break
        if not results:
            raise SearchError("DuckDuckGo вернул пустую выдачу (возможна блокировка)")
        return results


def _clean_ddg_url(href: str) -> str:
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target)
    return href


class WebSearch:
    """Фасад: пробует SearXNG, при неудаче — DuckDuckGo."""

    def __init__(self, config: dict[str, Any] | None = None):
        config = config or {}
        self.searxng = SearXNG(
            base_url=config.get("searxng_url", "http://localhost:8080"),
            timeout=float(config.get("timeout", 10)),
            language=config.get("language", "ru"),
        )
        self.fallback = DuckDuckGo(timeout=float(config.get("timeout", 12)))
        self.prefer = config.get("prefer", "searxng")
        self.last_backend = ""

    def health(self) -> dict[str, str]:
        ok, detail = self.searxng.health()
        return {"searxng": f"{'ok' if ok else 'нет'} — {detail}"}

    def search(self, query: str, limit: int = 8) -> tuple[list[SearchResult], str]:
        order = ["searxng", "duckduckgo"]
        if self.prefer != "searxng":
            order.reverse()

        errors = []
        for backend in order:
            try:
                if backend == "searxng":
                    results = self.searxng.search(query, limit)
                    if results:
                        self.last_backend = "searxng"
                        return results, "searxng"
                    errors.append("searxng: пусто")
                else:
                    results = self.fallback.search(query, limit)
                    self.last_backend = "duckduckgo"
                    return results, "duckduckgo"
            except SearchError as exc:
                errors.append(str(exc))
        raise SearchError("; ".join(errors))

    def context(self, query: str, limit: int = 5, max_chars: int = 3000) -> tuple[str, list[SearchResult]]:
        results, backend = self.search(query, limit=limit)
        blocks, used = [], 0
        for i, result in enumerate(results, start=1):
            block = result.as_context(i)
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks), results

    def fetch(self, url: str, max_chars: int = 6000) -> str:
        """Прочитать страницу и вернуть чистый текст."""
        resp = requests.get(
            url, timeout=15, headers={"User-Agent": USER_AGENT}, allow_redirects=True
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        text = html.unescape(text)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 2]
        return "\n".join(lines)[:max_chars]


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    search = WebSearch()
    print("Здоровье:", json.dumps(search.health(), ensure_ascii=False))
    started = time.perf_counter()
    try:
        results, backend = search.search("линейная алгебра определитель матрицы", limit=5)
        print(f"бэкенд: {backend}, {time.perf_counter() - started:.1f} с")
        for r in results:
            print(f"  - {r.title[:70]}\n    {r.url}\n    {r.snippet[:100]}")
    except SearchError as exc:
        print(f"ОШИБКА: {exc}")
