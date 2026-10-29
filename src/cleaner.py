"""Модуль управления кэшем, мусором и видеопамятью (VRAM) для DIBot.

Предоставляет функции анализа дискового пространства, очистки временных файлов
(кэш браузера DiSpace, старые загрузки чата, лог-файлы, байт-код Python) и выгрузки
неактивных моделей из GPU памяти — как по требованию пользователя, так и автоматически.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папки профиля браузера Chromium, которые представляют собой временный кэш и безопасны для удаления
SAFE_BROWSER_CACHE_SUBDIRS = (
    ("Default", "Cache"),
    ("Default", "Code Cache"),
    ("Default", "GPUCache"),
    ("Default", "DawnGraphiteCache"),
    ("Default", "DawnWebGPUCache"),
    ("BrowserMetrics",),
    ("Crashpad", "reports"),
    ("Crashpad", "attachments"),
    ("GrShaderCache",),
    ("ShaderCache",),
    ("GPUPersistentCache",),
    ("component_crx_cache",),
    ("extensions_crx_cache",),
)

# Папки и файлы, которые КАТЕГОРИЧЕСКИ НЕЛЬЗЯ УДАЛЯТЬ (в них хранятся куки сессии авторизации DiSpace)
PROTECTED_BROWSER_ITEMS = {
    "Cookies",
    "Cookies-journal",
    "Login Data",
    "Login Data-journal",
    "Local Storage",
    "IndexedDB",
    "WebStorage",
    "Session Storage",
    "Sessions",
    "Network",
}


def format_bytes(size_bytes: int | float) -> str:
    """Форматировать размер в человекочитаемый вид (Б, КБ, МБ, ГБ)."""
    val = float(size_bytes)
    if val < 1024:
        return f"{int(val)} Б"
    elif val < 1024 * 1024:
        return f"{val / 1024:.1f} КБ"
    elif val < 1024 * 1024 * 1024:
        return f"{val / (1024 * 1024):.1f} МБ"
    else:
        return f"{val / (1024 * 1024 * 1024):.2f} ГБ"


@dataclass
class CacheCategory:
    """Информация об одной категории кэша или временных данных."""

    name: str
    title: str
    size_bytes: int = 0
    file_count: int = 0
    description: str = ""
    safe_to_delete: bool = True

    @property
    def formatted_size(self) -> str:
        return format_bytes(self.size_bytes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "size_bytes": self.size_bytes,
            "formatted_size": self.formatted_size,
            "file_count": self.file_count,
            "description": self.description,
            "safe_to_delete": self.safe_to_delete,
        }


@dataclass
class CleanupResult:
    """Результат выполнения операции очистки."""

    bytes_freed: int = 0
    files_deleted: int = 0
    categories_cleaned: list[str] = field(default_factory=list)
    vram_models_unloaded: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    error_messages: list[str] = field(default_factory=list)

    @property
    def formatted_freed(self) -> str:
        return format_bytes(self.bytes_freed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bytes_freed": self.bytes_freed,
            "formatted_freed": self.formatted_freed,
            "files_deleted": self.files_deleted,
            "categories_cleaned": self.categories_cleaned,
            "vram_models_unloaded": self.vram_models_unloaded,
            "details": self.details,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).strftime("%d.%m.%Y %H:%M:%S"),
            "errors": self.error_messages,
        }


def _get_dir_size_and_count(
    path: Path,
    max_age_days: float = 0.0,
    check_mtime: bool = False,
) -> tuple[int, int]:
    """Рекурсивно подсчитать размер и количество файлов в папке."""
    if not path.exists():
        return 0, 0
    total_size = 0
    total_count = 0
    cutoff = time.time() - (max_age_days * 86400) if max_age_days > 0 else 0.0

    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    st = entry.stat()
                    if check_mtime and cutoff > 0 and st.st_mtime > cutoff:
                        continue
                    total_size += st.st_size
                    total_count += 1
                except OSError:
                    continue
    except OSError:
        pass
    return total_size, total_count


def _get_safe_browser_cache_paths(profile_dir: Path) -> list[Path]:
    """Получить список существующих безопасных путей к кэшу Chromium."""
    paths: list[Path] = []
    if not profile_dir.exists():
        return paths

    for parts in SAFE_BROWSER_CACHE_SUBDIRS:
        target = profile_dir.joinpath(*parts)
        if target.exists():
            paths.append(target)
    return paths


def _find_active_chat_referenced_files(history_dir: Path) -> set[str]:
    """Найти все имена файлов, которые реально используются в сохранённых чатах."""
    referenced = set()
    if not history_dir.exists():
        return referenced
    for h_file in history_dir.glob("*.json"):
        try:
            content = h_file.read_text(encoding="utf-8")
            data = json.loads(content)
            turns = data.get("turns") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for t in turns or []:
                for img in t.get("images") or []:
                    if isinstance(img, str):
                        referenced.add(Path(img).name)
                        referenced.add(img)
                for vid in t.get("videos") or []:
                    if isinstance(vid, str):
                        referenced.add(Path(vid).name)
                        referenced.add(vid)
        except Exception:
            continue
    return referenced


def get_loaded_ollama_models(base_url: str = "http://localhost:11434", timeout: float = 2.0) -> list[dict[str, Any]]:
    """Получить список моделей, которые сейчас загружены в VRAM / RAM локального Ollama."""
    url = f"{base_url.rstrip('/')}/api/ps"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "DIBot-Cleaner"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("models") or []
            out = []
            for m in models:
                size_b = int(m.get("size") or m.get("size_vram") or 0)
                out.append(
                    {
                        "name": m.get("name", ""),
                        "model": m.get("model", ""),
                        "size_bytes": size_b,
                        "size_gb": round(size_b / (1024**3), 2),
                        "processor": "GPU" if m.get("size_vram") else "CPU/RAM",
                        "expires_at": m.get("expires_at", ""),
                    }
                )
            return out
    except Exception:
        return []


def unload_ollama_vram(base_url: str = "http://localhost:11434", timeout: float = 5.0) -> list[str]:
    """Выгрузить все активные модели из VRAM/RAM Ollama, установив keep_alive = 0."""
    loaded = get_loaded_ollama_models(base_url, timeout=timeout)
    unloaded = []
    clean_url = base_url.rstrip("/")
    for item in loaded:
        name = item.get("name")
        if not name:
            continue
        try:
            req_data = json.dumps({"model": name, "keep_alive": 0}).encode("utf-8")
            req = urllib.request.Request(
                f"{clean_url}/api/generate",
                data=req_data,
                headers={"Content-Type": "application/json", "User-Agent": "DIBot-Cleaner"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    unloaded.append(name)
        except Exception:
            pass
    return unloaded


def get_storage_stats(
    project_root: Path = PROJECT_ROOT,
    ollama_url: str = "http://localhost:11434",
) -> dict[str, CacheCategory]:
    """Собрать полную статистику по дисковому пространству и кэшу DIBot."""
    categories: dict[str, CacheCategory] = {}

    # 1. Кэш браузера DiSpace (Chromium в data/browser_profile)
    b_profile = project_root / "data" / "browser_profile"
    b_size, b_count = 0, 0
    for p in _get_safe_browser_cache_paths(b_profile):
        sz, cnt = _get_dir_size_and_count(p)
        b_size += sz
        b_count += cnt
    categories["browser_cache"] = CacheCategory(
        name="browser_cache",
        title="Кэш браузера DiSpace (Chromium)",
        size_bytes=b_size,
        file_count=b_count,
        description="Сетевой и шейдерный кэш страниц DiSpace НГТУ. Очистка не сбрасывает вход в систему.",
        safe_to_delete=True,
    )

    # 2. Загруженные файлы чата (data/chat_uploads)
    u_dir = project_root / "data" / "chat_uploads"
    u_size, u_count = _get_dir_size_and_count(u_dir)
    categories["chat_uploads"] = CacheCategory(
        name="chat_uploads",
        title="Загруженные изображения и файлы чата",
        size_bytes=u_size,
        file_count=u_count,
        description="Фотографии и документы, прикреплённые пользователем в диалогах.",
        safe_to_delete=True,
    )

    # 3. Байт-код Python (__pycache__, *.pyc)
    pyc_size = 0
    pyc_count = 0
    for root in (project_root / "src", project_root / "scripts"):
        if root.exists():
            for f in root.rglob("*.pyc"):
                try:
                    pyc_size += f.stat().st_size
                    pyc_count += 1
                except OSError:
                    pass
    categories["python_cache"] = CacheCategory(
        name="python_cache",
        title="Байт-код Python (__pycache__)",
        size_bytes=pyc_size,
        file_count=pyc_count,
        description="Скомпилированные файлы скриптов Python (.pyc). Пересоздаются автоматически при запуске.",
        safe_to_delete=True,
    )

    # 4. Логи (logs)
    l_dir = project_root / "logs"
    l_size, l_count = _get_dir_size_and_count(l_dir)
    categories["logs"] = CacheCategory(
        name="logs",
        title="Журналы работы (logs)",
        size_bytes=l_size,
        file_count=l_count,
        description="Файлы логов и отладочных записей сервисов.",
        safe_to_delete=True,
    )

    # 5. Видеопамять GPU (VRAM Ollama)
    loaded_models = get_loaded_ollama_models(ollama_url)
    vram_size = sum(m.get("size_bytes", 0) for m in loaded_models)
    categories["vram"] = CacheCategory(
        name="vram",
        title="Видеопамять GPU (активные модели Ollama)",
        size_bytes=vram_size,
        file_count=len(loaded_models),
        description=(
            f"Загружено в память моделей: {len(loaded_models)} "
            f"({', '.join(m['name'] for m in loaded_models) or 'нет'})"
        ),
        safe_to_delete=True,
    )

    return categories


def clean_garbage(
    categories: Sequence[str] | None = None,
    max_age_days: float = 0.0,
    unload_vram: bool = False,
    preserve_active_chat_uploads: bool = True,
    project_root: Path = PROJECT_ROOT,
    ollama_url: str = "http://localhost:11434",
) -> CleanupResult:
    """Выполнить безопасную очистку выбранных категорий мусора и кэша.

    Параметры:
      - categories: список категорий ("browser_cache", "chat_uploads", "python_cache", "logs", "vram").
                    Если None, очищаются все файловые категории.
      - max_age_days: удалять файлы старше скольких дней (0 = удалять весь временный кэш без ограничений).
      - unload_vram: выгрузить ли неактивные модели из видеопамяти.
      - preserve_active_chat_uploads: не удалять загрузки, на которые ссылаются сохранённые сессии чата.
    """
    to_clean = set(categories) if categories is not None else {"browser_cache", "chat_uploads", "python_cache", "logs"}
    result = CleanupResult()
    cutoff_time = time.time() - (max_age_days * 86400) if max_age_days > 0 else 0.0

    # 1. Кэш браузера DiSpace
    if "browser_cache" in to_clean:
        b_profile = project_root / "data" / "browser_profile"
        cat_freed = 0
        cat_files = 0
        for cache_path in _get_safe_browser_cache_paths(b_profile):
            if cache_path.is_file():
                try:
                    st = cache_path.stat()
                    if cutoff_time == 0.0 or st.st_mtime < cutoff_time:
                        cat_freed += st.st_size
                        cache_path.unlink(missing_ok=True)
                        cat_files += 1
                except OSError as exc:
                    result.error_messages.append(f"Файл заблокирован ({cache_path.name}): {exc}")
            elif cache_path.is_dir():
                for root_dir, _dirs, files in os.walk(cache_path, topdown=False):
                    for fname in files:
                        fpath = Path(root_dir) / fname
                        # Дополнительная проверка на защищённые имена
                        if fname in PROTECTED_BROWSER_ITEMS:
                            continue
                        try:
                            st = fpath.stat()
                            if cutoff_time == 0.0 or st.st_mtime < cutoff_time:
                                sz = st.st_size
                                fpath.unlink(missing_ok=True)
                                cat_freed += sz
                                cat_files += 1
                        except OSError:
                            pass
                    # Пробуем удалить опустевшие вложенные папки
                    try:
                        if Path(root_dir) != cache_path:
                            os.rmdir(root_dir)
                    except OSError:
                        pass
        result.bytes_freed += cat_freed
        result.files_deleted += cat_files
        result.categories_cleaned.append("browser_cache")
        result.details["browser_cache"] = {"freed": format_bytes(cat_freed), "files": cat_files}

    # 2. Загрузки чата (chat_uploads)
    if "chat_uploads" in to_clean:
        u_dir = project_root / "data" / "chat_uploads"
        cat_freed = 0
        cat_files = 0
        referenced = set()
        if preserve_active_chat_uploads:
            h_dir = project_root / "data" / "chat_history"
            referenced = _find_active_chat_referenced_files(h_dir)

        if u_dir.exists():
            for root_dir, _dirs, files in os.walk(u_dir, topdown=False):
                for fname in files:
                    fpath = Path(root_dir) / fname
                    # Если файл задействован в активных сохранённых чатах, сохраняем его
                    if preserve_active_chat_uploads and (fname in referenced or str(fpath) in referenced):
                        continue
                    try:
                        st = fpath.stat()
                        # Если задан max_age_days, удаляем только старше порога, иначе все неиспользуемые
                        if cutoff_time == 0.0 or st.st_mtime < cutoff_time:
                            sz = st.st_size
                            fpath.unlink(missing_ok=True)
                            cat_freed += sz
                            cat_files += 1
                    except OSError:
                        pass
                try:
                    if Path(root_dir) != u_dir:
                        os.rmdir(root_dir)
                except OSError:
                    pass
        result.bytes_freed += cat_freed
        result.files_deleted += cat_files
        result.categories_cleaned.append("chat_uploads")
        result.details["chat_uploads"] = {"freed": format_bytes(cat_freed), "files": cat_files}

    # 3. Байт-код Python (__pycache__)
    if "python_cache" in to_clean:
        cat_freed = 0
        cat_files = 0
        for search_root in (project_root / "src", project_root / "scripts"):
            if not search_root.exists():
                continue
            for pyc in search_root.rglob("*.pyc"):
                try:
                    sz = pyc.stat().st_size
                    pyc.unlink(missing_ok=True)
                    cat_freed += sz
                    cat_files += 1
                except OSError:
                    pass
            for pydir in search_root.rglob("__pycache__"):
                try:
                    shutil.rmtree(pydir, ignore_errors=True)
                except OSError:
                    pass
        result.bytes_freed += cat_freed
        result.files_deleted += cat_files
        result.categories_cleaned.append("python_cache")
        result.details["python_cache"] = {"freed": format_bytes(cat_freed), "files": cat_files}

    # 4. Логи (logs)
    if "logs" in to_clean:
        l_dir = project_root / "logs"
        cat_freed = 0
        cat_files = 0
        if l_dir.exists():
            for log_file in l_dir.glob("*.log"):
                try:
                    st = log_file.stat()
                    if cutoff_time == 0.0 or st.st_mtime < cutoff_time:
                        sz = st.st_size
                        log_file.unlink(missing_ok=True)
                        cat_freed += sz
                        cat_files += 1
                except OSError:
                    pass
        result.bytes_freed += cat_freed
        result.files_deleted += cat_files
        result.categories_cleaned.append("logs")
        result.details["logs"] = {"freed": format_bytes(cat_freed), "files": cat_files}

    # 5. Видеопамять GPU (VRAM)
    if unload_vram or "vram" in to_clean:
        unloaded = unload_ollama_vram(ollama_url)
        result.vram_models_unloaded = unloaded
        if unloaded:
            result.categories_cleaned.append("vram")
            result.details["vram"] = {"unloaded_models": unloaded}

    return result


def auto_clean_if_needed(
    project_root: Path = PROJECT_ROOT,
    max_browser_cache_mb: float = 120.0,
    max_age_days: float = 7.0,
    interval_hours: float = 6.0,
    ollama_url: str = "http://localhost:11434",
) -> CleanupResult | None:
    """Автоматическая фоновая очистка мусора при превышении лимитов или интервала времени.

    Проверяет:
      1. Прошло ли более `interval_hours` с момента последней очистки;
      2. Превышает ли кэш браузера порог `max_browser_cache_mb`.
    Если условие выполнено, безопасно очищает кэш браузера, старые логи и неиспользуемые загрузки.
    """
    marker_file = project_root / "data" / ".last_autoclean"
    now = time.time()
    last_cleaned = 0.0

    if marker_file.exists():
        try:
            last_cleaned = float(marker_file.read_text(encoding="utf-8").strip())
        except Exception:
            last_cleaned = 0.0

    stats = get_storage_stats(project_root, ollama_url)
    b_cache_mb = stats["browser_cache"].size_bytes / (1024 * 1024)
    hours_since_last = (now - last_cleaned) / 3600.0

    # Проверяем условия запуска
    should_run = (hours_since_last >= interval_hours) or (b_cache_mb >= max_browser_cache_mb)

    if not should_run:
        return None

    # Выполняем автоочистку
    res = clean_garbage(
        categories=["browser_cache", "chat_uploads", "python_cache", "logs"],
        max_age_days=max_age_days,
        unload_vram=False,
        preserve_active_chat_uploads=True,
        project_root=project_root,
        ollama_url=ollama_url,
    )

    try:
        marker_file.parent.mkdir(parents=True, exist_ok=True)
        marker_file.write_text(str(now), encoding="utf-8")
    except Exception:
        pass

    return res


def main() -> None:
    """Точка входа командной строки для управления кэшем."""
    parser = argparse.ArgumentParser(description="Утилита очистки кэша, мусора и VRAM для DIBot")
    parser.add_argument("--status", action="store_true", help="Показать статистику занятого места и кэша")
    parser.add_argument("--clean", action="store_true", help="Очистить временный кэш (браузер, pycache, логи)")
    parser.add_argument("--all", action="store_true", help="Полная очистка всего кэша и старых файлов")
    parser.add_argument("--vram", action="store_true", help="Выгрузить неактивные модели из видеопамяти (VRAM)")
    parser.add_argument("--auto", action="store_true", help="Выполнить проверку и автоочистку по порогам")
    parser.add_argument("--days", type=float, default=7.0, help="Порог возраста файлов в днях (по умолчанию: 7)")
    args = parser.parse_args()

    if args.auto:
        print("🔍 Проверка необходимости автоматической очистки…")
        res = auto_clean_if_needed(max_age_days=args.days)
        if res and res.bytes_freed > 0:
            print(f"✅ Автоочистка завершена: освобождено {res.formatted_freed}, удалено {res.files_deleted} файлов.")
        else:
            print("ℹ️ Автоочистка не требуется (кэш в пределах нормы).")
        return

    if args.vram:
        print("⚡ Выгрузка моделей из памяти GPU…")
        unloaded = unload_ollama_vram()
        if unloaded:
            print(f"✅ Выгружено из VRAM: {', '.join(unloaded)}")
        else:
            print("ℹ️ В VRAM не найдено активных моделей.")
        return

    if args.clean or args.all:
        days = 0.0 if args.all else args.days
        print(f"🧹 Очистка временного мусора (файлы старше {days} дн.)…")
        res = clean_garbage(
            max_age_days=days,
            unload_vram=args.all,
            preserve_active_chat_uploads=True,
        )
        print(f"✅ Готово! Освобождено диска: {res.formatted_freed} (удалено файлов: {res.files_deleted})")
        if res.vram_models_unloaded:
            print(f"⚡ Модели выгружены из VRAM: {', '.join(res.vram_models_unloaded)}")
        for cat, det in res.details.items():
            print(f"  • {cat}: освобождено {det.get('freed', '0 Б')}")
        return

    # По умолчанию или при --status выводим красивую сводку
    print("\n📊 Статистика использования кэша и памяти DIBot:\n" + "=" * 55)
    stats = get_storage_stats()
    total_disk_cache = 0
    for cat_key, cat in stats.items():
        if cat_key != "vram":
            total_disk_cache += cat.size_bytes
        print(f"📦 {cat.title}:")
        print(f"   Размер: {cat.formatted_size} ({cat.file_count} файлов/объектов)")
        print(f"   Описание: {cat.description}")
        print("-" * 55)
    print(f"💾 Всего дискового кэша: {format_bytes(total_disk_cache)}\n")


if __name__ == "__main__":
    main()
