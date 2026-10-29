"""Умный лаунчер DIBot для Windows ПК.

Автоматически:
  1. Проверяет наличие Python и зависимостей (при необходимости доустанавливает).
  2. Проверяет и при необходимости запускает локальный сервер Ollama.
  3. Проверяет наличие моделей Ollama (deepseek-r1, qwen3.5, qwen2.5-coder, qwen3-vl).
  4. Проверяет актуальность базы учебных материалов DiSpace.
  5. Запускает Streamlit и автоматически открывает браузер на http://localhost:8501.
  6. При необходимости поднимает безопасный HTTPS-туннель для доступа со смартфона.
  7. Умеет в 1 клик создать удобный ярлык на Рабочем столе Windows.

Запуск:
    python run.py                  # Интерактивное меню с автозапуском через 3 секунды
    python run.py --quick          # Мгновенный запуск без меню
    python run.py --tunnel         # Запуск вместе с Cloudflare HTTPS-туннелем для телефона
    python run.py --shortcut       # Создать ярлык на Рабочем столе и выйти
    python run.py --health         # Только диагностика системы
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = PROJECT_ROOT / "data"

OLLAMA_API = "http://127.0.0.1:11434"
DEFAULT_PORT = 8501

RECOMMENDED_MODELS = [
    ("deepseek-r1:8b", "🧮 Математика, пошаговые доказательства и расчёты"),
    ("qwen3.5:9b", "💬 Универсал, теория, конспекты и диалог"),
    ("qwen2.5-coder:7b", "💻 Код, Python, LaTeX и SymPy"),
    ("qwen3-vl:8b", "👁️ Распознавание фото задач и формул"),
]


def print_banner() -> None:
    print("\n" + "=" * 68)
    print("   🎓 DIBot — Интеллектуальный учебный ассистент студента НГТУ")
    print("=" * 68 + "\n")


def check_python_environment() -> bool:
    """Проверка версии Python и обязательных пакетов."""
    print("[1/5] 🐍 Проверка окружения Python...")
    if sys.version_info < (3, 10):
        print(f"  ❌ Требуется Python 3.10+, текущая версия: {sys.version.split()[0]}")
        return False
    print(f"  ✓ Python {sys.version.split()[0]} ({sys.executable})")

    req_file = PROJECT_ROOT / "requirements.txt"
    missing: list[str] = []
    for pkg in ("streamlit", "requests", "sympy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"  ⚠️ Отсутствуют пакеты: {', '.join(missing)}. Устанавливаю из requirements.txt...")
        if req_file.exists():
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(req_file)]
            res = subprocess.run(cmd)
            if res.returncode != 0:
                print("  ❌ Ошибка установки зависимостей pip.")
                return False
        else:
            cmd = [sys.executable, "-m", "pip", "install", *missing]
            subprocess.run(cmd)
        print("  ✓ Зависимости успешно установлены.")
    else:
        print("  ✓ Все базовые библиотеки (streamlit, requests, sympy) на месте.")
    return True


def is_ollama_running() -> bool:
    """Проверить, отвечает ли Ollama по HTTP."""
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags", headers={"User-Agent": "DIBot-Launcher"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def find_ollama_executable() -> str | None:
    """Найти путь к ollama.exe."""
    found = shutil.which("ollama")
    if found:
        return found
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
        if candidate.exists():
            return str(candidate)
    return None


def ensure_ollama_running() -> bool:
    """Проверить и при необходимости запустить службу Ollama."""
    print("[2/5] 🦙 Проверка локального сервиса Ollama...")
    if is_ollama_running():
        print("  ✓ Ollama активен и принимает запросы (http://127.0.0.1:11434).")
        return True

    ollama_exe = find_ollama_executable()
    if not ollama_exe:
        print("  ⚠️ Ollama не найден в PATH и AppData. Если вы используете только облачные модели, это нормально.")
        return False

    print("  ⏳ Ollama не запущен. Запускаю фоновую службу Ollama...")
    try:
        # Запускаем ollama serve скрытно в фоне
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        subprocess.Popen(
            [ollama_exe, "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=creationflags,
        )
    except Exception as exc:
        print(f"  ❌ Не удалось запустить Ollama: {exc}")
        return False

    # Ожидание готовности
    for i in range(15):
        time.sleep(0.5)
        if is_ollama_running():
            print(f"  ✓ Ollama успешно запущен и готов к работе (за {round((i+1)*0.5, 1)} с).")
            return True

    print("  ⚠️ Ollama запущен, но пока не ответил на пинг. Продолжаю запуск...")
    return False


def get_available_ollama_models() -> list[str]:
    """Получить список установленных моделей в Ollama."""
    try:
        req = urllib.request.Request(f"{OLLAMA_API}/api/tags")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def check_models_status() -> None:
    """Вывести статус установленных моделей."""
    print("[3/5] 🧠 Проверка локальных моделей нейросетей...")
    available = get_available_ollama_models()
    if not available:
        print("  ℹ️ Локальные модели не обнаружены или Ollama недоступен.")
        print("     Вы сможете пользоваться быстрыми облачными моделями (Groq, OpenRouter, Gemini).")
        return

    clean_avail = [m.split(":")[0] + (":" + m.split(":")[1] if ":" in m else "") for m in available]
    for model_id, desc in RECOMMENDED_MODELS:
        base_id = model_id.split(":")[0]
        is_here = any(model_id in a or (base_id in a and ":latest" in a) for a in clean_avail)
        mark = "✓" if is_here else "—"
        status = "установлена" if is_here else "не найдена (опционально)"
        print(f"  {mark} `{model_id:<17}` : {status:<24} [{desc}]")


def check_materials_and_dispace() -> None:
    """Проверка наличия базы учебных материалов и токена DiSpace."""
    print("[4/5] 📚 Проверка базы курсов и DiSpace НГТУ...")
    index_file = DATA_DIR / "materials" / "index.json"
    if index_file.exists():
        try:
            data = json.loads(index_file.read_text(encoding="utf-8"))
            sources = data.get("sources", [])
            chunks = data.get("chunks", [])
            courses = sorted({s.get("course", "") for s in sources if s.get("course")})
            print(f"  ✓ База знаний активна: {len(sources)} источников, {len(chunks)} фрагментов лекций.")
            print(f"  ✓ Дисциплины: {', '.join(courses[:4])}{' и др.' if len(courses) > 4 else ''}")
        except Exception:
            print("  ✓ База знаний обнаружена.")
    else:
        print("  ℹ️ База материалов пуста. Загрузите файлы в веб-интерфейсе или подключите DiSpace.")

    token_file = DATA_DIR / "dispace_token.txt"
    if token_file.exists() and token_file.read_text(encoding="utf-8").strip():
        print("  ✓ Авторизация DiSpace: активна сохранённая сессия студента.")
    else:
        print("  ℹ️ Вход в DiSpace: можно войти в 1 клик в веб-интерфейсе.")


def create_desktop_shortcut() -> bool:
    """Создать красивый ярлык на Рабочем столе Windows через PowerShell WScript.Shell."""
    if sys.platform != "win32":
        print("Создание ярлыка поддерживается только в Windows.")
        return False

    desktop_dir = Path(os.environ.get("USERPROFILE", "")) / "Desktop"
    if not desktop_dir.exists():
        desktop_dir = Path.home() / "Desktop"
    shortcut_path = desktop_dir / "DIBot.lnk"

    target_bat = PROJECT_ROOT / "start.bat"
    if not target_bat.exists():
        target_bat.write_text(
            "@echo off\n"
            "chcp 65001 >nul\n"
            "cd /d \"%~dp0\"\n"
            "python run.py\n",
            encoding="utf-8",
        )

    ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{shortcut_path}')
$Shortcut.TargetPath = '{target_bat}'
$Shortcut.WorkingDirectory = '{PROJECT_ROOT}'
$Shortcut.Description = 'DIBot - Интеллектуальный учебный ассистент НГТУ'
$Shortcut.WindowStyle = 1
$Shortcut.Save()
"""
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-Command", ps_script], capture_output=True, text=True)
        if res.returncode == 0 and shortcut_path.exists():
            print(f"\n  🎉 Ярлык успешно создан на Рабочем столе:\n     -> {shortcut_path}")
            return True
        print(f"  ❌ Ошибка PowerShell: {res.stderr}")
    except Exception as exc:
        print(f"  ❌ Не удалось создать ярлык: {exc}")
    return False


def pull_missing_models() -> None:
    """Помощник докачки недостающих моделей Ollama."""
    ollama_exe = find_ollama_executable() or "ollama"
    print("\n📦 Загрузка моделей нейросетей через Ollama:")
    for idx, (m_id, desc) in enumerate(RECOMMENDED_MODELS, 1):
        print(f"  [{idx}] Скачать `{m_id}` ({desc})")
    print("  [A] Скачать все рекомендуемые модели по очереди")
    print("  [0] Назад в меню")

    choice = input("\nВыберите номер модели для загрузки: ").strip().lower()
    if choice in ("0", "q", "exit"):
        return
    if choice == "a":
        for m_id, _ in RECOMMENDED_MODELS:
            print(f"\n>>> Загрузка {m_id}...")
            subprocess.run([ollama_exe, "pull", m_id])
        return

    try:
        num = int(choice)
        if 1 <= num <= len(RECOMMENDED_MODELS):
            selected_model = RECOMMENDED_MODELS[num - 1][0]
            print(f"\n>>> Загрузка {selected_model}...")
            subprocess.run([ollama_exe, "pull", selected_model])
    except ValueError:
        print("Неверный ввод.")


def run_health_check() -> None:
    """Полная диагностика всех компонентов системы."""
    print("\n" + "=" * 60)
    print("   🛠️ Диагностический отчёт DIBot")
    print("=" * 60)
    check_python_environment()
    ensure_ollama_running()
    check_models_status()
    check_materials_and_dispace()
    print("=" * 60)
    try:
        input("\nНажмите Enter, чтобы вернуться в меню...")
    except (EOFError, KeyboardInterrupt):
        pass


def wait_for_server(url: str, timeout: float = 12.0) -> bool:
    """Ожидание ответа веб-сервера."""
    started = time.perf_counter()
    while time.perf_counter() - started < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if resp.status in (200, 404):
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def start_dibot(*, port: int = DEFAULT_PORT, with_tunnel: bool = False) -> None:
    """Запуск Streamlit и открытие браузера."""
    print(f"\n[5/5] 🚀 Запуск веб-интерфейса DIBot на порту {port}...")

    tunnel_proc: subprocess.Popen[Any] | None = None
    if with_tunnel:
        gateway_script = PROJECT_ROOT / "scripts" / "secure_ollama_gateway.py"
        if gateway_script.exists():
            print("  🌐 Поднимаю защищённый шлюз и HTTPS-туннель Cloudflare...")
            tunnel_proc = subprocess.Popen(
                [sys.executable, str(gateway_script)],
                cwd=str(PROJECT_ROOT),
            )

    streamlit_app = PROJECT_ROOT / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--theme.base",
        "dark",
    ]

    print(f"  -> Локальный адрес: http://localhost:{port}")
    print("  -> Запуск сервера Streamlit...")

    # Фоновый поток для открытия браузера, как только сервер станет доступен
    def _open_browser() -> None:
        target_url = f"http://localhost:{port}"
        if wait_for_server(target_url, timeout=15.0):
            print(f"  ✓ Сервер готов! Открываю в браузере: {target_url}")
            webbrowser.open(target_url)

    threading.Thread(target=_open_browser, daemon=True).start()

    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n🛑 Завершение работы DIBot по сигналу пользователя (Ctrl+C)...")
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass


def check_and_autoclean() -> None:
    """Фоновая проверка и быстрая автоочистка мусора."""
    try:
        sys.path.insert(0, str(SRC_DIR))
        from cleaner import auto_clean_if_needed

        res = auto_clean_if_needed(PROJECT_ROOT)
        if res and res.bytes_freed > 0:
            print(f"  ✓ Автоочистка: удалено {res.files_deleted} устаревших файлов ({res.formatted_freed}).")
    except Exception:
        pass


def run_clean_interactive() -> None:
    """Интерактивная очистка кэша и памяти из меню лаунчера."""
    sys.path.insert(0, str(SRC_DIR))
    from cleaner import get_storage_stats, clean_garbage, unload_ollama_vram, format_bytes

    print("\n" + "=" * 60)
    print("   🧹 Очистка кэша, мусора и памяти DIBot")
    print("=" * 60)
    stats = get_storage_stats(PROJECT_ROOT)
    for cat in stats.values():
        print(f"  • {cat.title}: {cat.formatted_size} ({cat.file_count} файлов/объектов)")
    print("-" * 60)
    print("  [1] 🧹 Быстрая очистка (кэш страниц браузера, pycache, логи) [Enter]")
    print("  [2] ⚡ Освободить видеопамять (выгрузить модели из VRAM GPU)")
    print("  [3] 🗑️ Глубокая очистка (файлы и загрузки старше 7 дней)")
    print("  [0] Назад в меню")

    choice = input("\nВыберите действие [по умолчанию 1]: ").strip()
    if choice in ("0", "q", "exit"):
        return
    if choice == "2":
        unloaded = unload_ollama_vram()
        if unloaded:
            print(f"  ✓ Модели выгружены из VRAM: {', '.join(unloaded)}")
        else:
            print("  ✓ Видеопамять уже свободна.")
    elif choice == "3":
        res = clean_garbage(max_age_days=7.0, unload_vram=False, preserve_active_chat_uploads=True)
        print(f"  ✓ Очищено {res.formatted_freed} (удалено {res.files_deleted} файлов).")
    else:
        res = clean_garbage(categories=["browser_cache", "python_cache", "logs"], max_age_days=0.0)
        print(f"  ✓ Очищено {res.formatted_freed} (удалено {res.files_deleted} файлов).")
    input("\nНажмите Enter, чтобы продолжить...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Умный лаунчер DIBot")
    parser.add_argument("-q", "--quick", action="store_true", help="Быстрый запуск без отображения меню")
    parser.add_argument("-t", "--tunnel", action="store_true", help="Запуск вместе с защищённым HTTPS-туннелем")
    parser.add_argument("-s", "--shortcut", action="store_true", help="Создать ярлык на Рабочем столе и выйти")
    parser.add_argument("--health", action="store_true", help="Выполнить полную диагностику системы")
    parser.add_argument("--clean", action="store_true", help="Очистить временный кэш, мусор и выйти")
    parser.add_argument("-p", "--port", type=int, default=DEFAULT_PORT, help="Порт для Streamlit (по умолч. 8501)")
    args = parser.parse_args()

    print_banner()

    if args.shortcut:
        create_desktop_shortcut()
        return

    if args.clean:
        sys.path.insert(0, str(SRC_DIR))
        from cleaner import clean_garbage

        res = clean_garbage(categories=["browser_cache", "python_cache", "logs"], max_age_days=0.0)
        print(f"✓ Кэш очищен: освобождено {res.formatted_freed} (удалено {res.files_deleted} файлов).")
        return

    # Базовая валидация среды
    check_python_environment()
    ensure_ollama_running()
    check_models_status()
    check_materials_and_dispace()
    check_and_autoclean()

    if args.health:
        run_health_check()
        return

    if args.quick:
        start_dibot(port=args.port, with_tunnel=args.tunnel)
        return

    # Интерактивное меню
    print("\n" + "-" * 60)
    print("  Выберите действие:")
    print("  [1] 🚀 Запустить DIBot (Локально: авто-открытие браузера)  [Enter]")
    print("  [2] 🌐 Запустить с доступом для смартфона (Cloudflare HTTPS)")
    print("  [3] 📦 Проверить и докачать модели нейросетей (Ollama pull)")
    print("  [4] 🖥️ Создать удобный ярлык на Рабочем столе Windows")
    print("  [5] 🛠️ Диагностика системы (Health Check)")
    print("  [6] 🧹 Очистить временный кэш, мусор и память VRAM")
    print("  [0] ❌ Выход")
    print("-" * 60)

    try:
        user_choice = input("Ваш выбор [по умолчанию 1]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nОтмена.")
        return

    if not user_choice or user_choice == "1":
        start_dibot(port=args.port, with_tunnel=False)
    elif user_choice == "2":
        start_dibot(port=args.port, with_tunnel=True)
    elif user_choice == "3":
        pull_missing_models()
    elif user_choice == "4":
        create_desktop_shortcut()
        input("\nНажмите Enter для запуска DIBot...")
        start_dibot(port=args.port, with_tunnel=False)
    elif user_choice == "5":
        run_health_check()
    elif user_choice == "6":
        run_clean_interactive()
        main()
    elif user_choice == "0":
        print("До встречи!")
    else:
        print("Неизвестный выбор, запускаю DIBot по умолчанию...")
        start_dibot(port=args.port, with_tunnel=False)


if __name__ == "__main__":
    main()
