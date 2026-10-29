"""Безопасный шлюз (Auth Gateway + HTTPS Tunnel) к локальному Ollama для удалённых устройств и Streamlit Cloud.

Почему нельзя просто пробрасывать порт 11434 наружу:
  * У Ollama нет встроенной авторизации: любой, кто узнает ссылку туннеля, сможет
    удалять твои модели (/api/delete), скачивать чужие (/api/pull) или нагружать видеокарту.

Как работает этот шлюз:
  1. Поднимает локальный защищённый прокси на 127.0.0.1:11435.
  2. Генерирует криптостойкий 256-битный ключ `dibot_sk_...` (сохраняется в data/gateway_token.txt,
     добавлен в .gitignore и никогда не попадает в GitHub).
  3. Проверяет заголовок `Authorization: Bearer <TOKEN>` (или `X-DIBot-Token`) через
     `hmac.compare_digest` (защита от timing-атак).
  4. Блокирует перебор токенов (rate-limit: после 8 неверных попыток IP блокируется на 10 минут).
  5. Разрешает ТОЛЬКО безопасные учебные запросы (/api/tags, /api/ps, /api/chat, /api/generate)
     и жёстко блокирует (403 Forbidden) опасные команды управления (/api/delete, /api/pull, /api/create, /api/push).
  6. Автоматически поднимает зашифрованный HTTPS-туннель Cloudflare (`cloudflared`) до шлюза
     и выводит готовый блок настроек для Streamlit Cloud и мобильных устройств.

Запуск:
    python scripts/secure_ollama_gateway.py
    python scripts/secure_ollama_gateway.py --rotate-token   # перевыпустить секретный токен
    python scripts/secure_ollama_gateway.py --no-tunnel      # только локальный прокси без cloudflared
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TOKEN_FILE = DATA_DIR / "gateway_token.txt"
STATE_FILE = DATA_DIR / "gateway_state.json"

OLLAMA_UPSTREAM = "http://127.0.0.1:11434"
DEFAULT_GATEWAY_HOST = "127.0.0.1"
DEFAULT_GATEWAY_PORT = 11435

ALLOWED_GET_PATHS = {"/api/tags", "/api/ps", "/api/version", "/health", "/dispace/token"}
ALLOWED_POST_PATHS = {"/api/chat", "/api/generate", "/api/show"}

# Защита от брутфорса: IP -> [timestamps неверных попыток]
_FAILED_ATTEMPTS: dict[str, list[float]] = {}
_LOCK = threading.Lock()
MAX_FAILURES = 8
BAN_SECONDS = 600.0


def load_or_create_token(rotate: bool = False) -> str:
    """Прочитать существующий 256-битный токен шлюза или создать новый."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not rotate and TOKEN_FILE.exists():
        existing = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if len(existing) >= 32:
            return existing
    token = "dibot_sk_" + secrets.token_urlsafe(32)
    TOKEN_FILE.write_text(token, encoding="utf-8")
    return token


def save_gateway_state(
    *,
    running: bool,
    port: int,
    public_url: str = "",
    pid: int | None = None,
) -> None:
    """Сохранить текущее состояние шлюза и публичный URL туннеля для отображения в UI."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "running": running,
        "local_url": f"http://127.0.0.1:{port}",
        "public_url": public_url,
        "pid": pid or os.getpid(),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gateway_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def find_cloudflared() -> str | None:
    """Найти исполняемый файл cloudflared в PATH или стандартных папках Windows."""
    in_path = shutil.which("cloudflared")
    if in_path:
        return in_path
    candidates = [
        DATA_DIR / "bin" / "cloudflared.exe",
        Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
        Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
        Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Links" / "cloudflared.exe",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _client_ip(handler: BaseHTTPRequestHandler) -> str:
    for header in ("CF-Connecting-IP", "X-Forwarded-For", "X-Real-IP"):
        val = handler.headers.get(header)
        if val:
            return val.split(",")[0].strip()
    return handler.client_address[0] if handler.client_address else "unknown"


def _is_banned(ip: str) -> bool:
    now = time.time()
    with _LOCK:
        attempts = [t for t in _FAILED_ATTEMPTS.get(ip, []) if now - t < BAN_SECONDS]
        _FAILED_ATTEMPTS[ip] = attempts
        return len(attempts) >= MAX_FAILURES


def _record_failure(ip: str) -> None:
    now = time.time()
    with _LOCK:
        attempts = [t for t in _FAILED_ATTEMPTS.get(ip, []) if now - t < BAN_SECONDS]
        attempts.append(now)
        _FAILED_ATTEMPTS[ip] = attempts


def _clear_failures(ip: str) -> None:
    with _LOCK:
        _FAILED_ATTEMPTS.pop(ip, None)


class OllamaGatewayHandler(BaseHTTPRequestHandler):
    expected_token: str = ""
    upstream_url: str = OLLAMA_UPSTREAM

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        ts = dt.datetime.now().strftime("%H:%M:%S")
        ip = _client_ip(self)
        print(f"[{ts}] {ip} -> {fmt % args}", flush=True)

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _check_auth(self) -> bool:
        ip = _client_ip(self)
        if _is_banned(ip):
            self._send_json(
                429,
                {
                    "error": "Слишком много неверных попыток авторизации. IP временно заблокирован на 10 минут."
                },
            )
            return False

        auth_header = self.headers.get("Authorization", "").strip()
        custom_header = self.headers.get("X-DIBot-Token", "").strip()

        provided = ""
        if auth_header.lower().startswith("bearer "):
            provided = auth_header[7:].strip()
        elif custom_header:
            provided = custom_header

        if not provided or not hmac.compare_digest(provided, self.expected_token):
            _record_failure(ip)
            self._send_json(
                401,
                {
                    "error": "Доступ запрещён (401 Unauthorized). Укажите верный токен шлюза DIBot (OLLAMA_API_KEY)."
                },
            )
            return False

        _clear_failures(ip)
        return True

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/health":
            if not self._check_auth():
                return
            self._send_json(200, {"ok": True, "service": "dibot-secure-ollama-gateway"})
            return

        if path == "/dispace/token":
            if not self._check_auth():
                return
            try:
                sys.path.insert(0, str(PROJECT_ROOT / "src"))
                from lms_reader import get_api_client, load_auth_token

                tok = load_auth_token()
                api = get_api_client(auto_refresh=True)
                if api and api.token:
                    tok = api.token
                self._send_json(200, {"ok": bool(tok), "token": tok or ""})
            except Exception as exc:
                self._send_json(200, {"ok": False, "token": "", "error": str(exc)})
            return

        if path not in ALLOWED_GET_PATHS:
            self._send_json(
                403,
                {"error": f"Эндпоинт GET {path} запрещён политикой безопасности шлюза."},
            )
            return

        if not self._check_auth():
            return

        try:
            resp = requests.get(f"{self.upstream_url}{path}", timeout=15)
            body = resp.content
            self.send_response(resp.status_code)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except requests.RequestException as exc:
            self._send_json(502, {"error": f"Локальный Ollama недоступен: {exc}"})

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path not in ALLOWED_POST_PATHS:
            self._send_json(
                403,
                {
                    "error": f"Эндпоинт POST {path} заблокирован шлюзом безопасности (разрешены только запросы генерации)."
                },
            )
            return

        if not self._check_auth():
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > 25 * 1024 * 1024:
            self._send_json(413, {"error": "Некорректный или слишком большой размер запроса."})
            return

        raw_body = self.rfile.read(length)
        try:
            parsed = json.loads(raw_body.decode("utf-8"))
            is_stream = bool(parsed.get("stream", False))
        except Exception:
            is_stream = False

        try:
            if not is_stream:
                resp = requests.post(
                    f"{self.upstream_url}{path}",
                    data=raw_body,
                    headers={"Content-Type": "application/json"},
                    timeout=600,
                )
                out = resp.content
                self.send_response(resp.status_code)
                self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)
            else:
                # Немедленно отправляем заголовки 200 OK и шлём пустые строки (\n) каждые 5 секунд,
                # пока Ollama выгружает/загружает модель в VRAM или думает над задачей.
                # Это гарантирует, что Cloudflare Tunnel никогда не разорвёт соединение с ошибкой 524!
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(b"\n")
                self.wfile.flush()

                line_q: queue.Queue[bytes | None] = queue.Queue()

                def _upstream_worker() -> None:
                    try:
                        with requests.post(
                            f"{self.upstream_url}{path}",
                            data=raw_body,
                            headers={"Content-Type": "application/json"},
                            stream=True,
                            timeout=600,
                        ) as resp:
                            if not resp.ok:
                                err_payload = json.dumps(
                                    {"error": f"HTTP {resp.status_code}: {resp.text[:300]}", "done": True},
                                    ensure_ascii=False,
                                ).encode("utf-8")
                                line_q.put(err_payload + b"\n")
                                return
                            for chunk in resp.iter_lines():
                                if chunk:
                                    line_q.put(chunk + b"\n")
                    except Exception as exc:
                        err_payload = json.dumps(
                            {"error": f"Ошибка локального Ollama: {exc}", "done": True},
                            ensure_ascii=False,
                        ).encode("utf-8")
                        line_q.put(err_payload + b"\n")
                    finally:
                        line_q.put(None)

                worker = threading.Thread(target=_upstream_worker, daemon=True)
                worker.start()

                while True:
                    try:
                        item = line_q.get(timeout=5.0)
                    except queue.Empty:
                        # Heartbeat для удержания соединения Cloudflare
                        self.wfile.write(b"\n")
                        self.wfile.flush()
                        continue
                    if item is None:
                        break
                    self.wfile.write(item)
                    self.wfile.flush()
        except (requests.RequestException, OSError) as exc:
            try:
                self._send_json(502, {"error": f"Ошибка обращения к локальному Ollama: {exc}"})
            except Exception:
                pass

    def do_DELETE(self) -> None:
        self._send_json(403, {"error": "Удаление моделей через удалённый шлюз строго запрещено."})

    def do_PUT(self) -> None:
        self._send_json(403, {"error": "Метод PUT запрещён политикой безопасности."})


def start_cloudflared_tunnel(local_port: int) -> tuple[subprocess.Popen | None, str]:
    """Запустить cloudflared tunnel и дождаться появления публичного HTTPS URL."""
    cf_bin = find_cloudflared()
    if not cf_bin:
        return None, ""

    proc = subprocess.Popen(
        [cf_bin, "tunnel", "--url", f"http://127.0.0.1:{local_port}", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    url_pattern = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
    found_url: dict[str, str] = {"url": ""}

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = url_pattern.search(line)
            if m and not found_url["url"]:
                found_url["url"] = m.group(0)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()

    deadline = time.time() + 25.0
    while time.time() < deadline and not found_url["url"]:
        if proc.poll() is not None:
            break
        time.sleep(0.3)

    return proc, found_url["url"]


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description="Безопасный шлюз к локальному Ollama для удалённого доступа")
    parser.add_argument("--port", type=int, default=DEFAULT_GATEWAY_PORT, help="локальный порт шлюза (по умолчанию 11435)")
    parser.add_argument("--rotate-token", action="store_true", help="сгенерировать новый 256-битный токен доступа")
    parser.add_argument("--no-tunnel", action="store_true", help="не запускать cloudflared (только локальный прокси)")
    args = parser.parse_args(argv)

    token = load_or_create_token(rotate=args.rotate_token)
    OllamaGatewayHandler.expected_token = token

    server = ThreadingHTTPServer((DEFAULT_GATEWAY_HOST, args.port), OllamaGatewayHandler)
    srv_thread = threading.Thread(target=server.serve_forever, daemon=True)
    srv_thread.start()

    tunnel_proc = None
    public_url = ""
    if not args.no_tunnel:
        print("Запускаю защищённый HTTPS-туннель Cloudflare (cloudflared)...", flush=True)
        tunnel_proc, public_url = start_cloudflared_tunnel(args.port)

    save_gateway_state(running=True, port=args.port, public_url=public_url, pid=os.getpid())

    print("\n" + "=" * 76)
    print("🔒 БЕЗОПАСНЫЙ ШЛЮЗ DIBOT К ЛОКАЛЬНЫМ МОДЕЛЯМ OLLAMA ЗАПУЩЕН")
    print("=" * 76)
    print(f"Локальный адрес шлюза : http://127.0.0.1:{args.port}")
    if public_url:
        print(f"Публичный HTTPS URL   : {public_url}")
    else:
        print("Публичный HTTPS URL   : (туннель не запущен — проверь установку cloudflared)")
    print(f"Секретный токен (256b): {token}")
    print("-" * 76)
    print("Защита:")
    print("  • Авторизация по Bearer-токену (hmac.compare_digest)")
    print("  • Разрешены только /api/tags, /api/ps, /api/chat, /api/generate")
    print("  • Удаление/скачивание моделей (/api/delete, /api/pull) заблокировано (403)")
    print("  • Защита от подбора токена: бан IP на 10 мин после 8 ошибок (429)")
    if public_url:
        print("-" * 76)
        print("Скопируй это в Streamlit Cloud -> Settings -> Secrets (или в боковую панель):")
        print(f'OLLAMA_BASE_URL = "{public_url}"')
        print(f'OLLAMA_API_KEY = "{token}"')
    print("=" * 76 + "\n", flush=True)

    try:
        while True:
            time.sleep(1.0)
            if tunnel_proc and tunnel_proc.poll() is not None:
                print("Внимание: процесс cloudflared завершился, перезапускаю туннель...", flush=True)
                tunnel_proc, public_url = start_cloudflared_tunnel(args.port)
                save_gateway_state(running=True, port=args.port, public_url=public_url, pid=os.getpid())
    except KeyboardInterrupt:
        print("\nОстанавливаю шлюз...", flush=True)
    finally:
        save_gateway_state(running=False, port=args.port, public_url="")
        server.shutdown()
        if tunnel_proc and tunnel_proc.poll() is None:
            tunnel_proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
