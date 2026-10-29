"""Клиент Ollama с автоматическим выбором модели.

Отличия от прежней версии, которая молча возвращала строку "Error: ...":
  * конфиг ищется относительно корня проекта, а не по захардкоженному D:/DIBot;
  * модель выбирается роутером по задаче (см. model_router.py);
  * отсутствующая модель даёт понятную ошибку со списком доступных;
  * таймаут настраивается (локальные модели на CPU думают минутами);
  * есть health-check и потоковая выдача токенов для UI.

Пример:
    from llm_client import LLMClient
    llm = LLMClient()
    print(llm.generate("Найди предел lim x->0 sin(x)/x", task="math"))
"""

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import requests

from model_router import DEFAULT_CONFIG_PATH, Route, Router, classify

log = logging.getLogger("dibot.llm")

Messages = Sequence[Mapping[str, Any]]


class LLMError(RuntimeError):
    """Общая ошибка слоя моделей."""


class OllamaUnavailable(LLMError):
    """Сервер Ollama не отвечает."""


class ModelUnavailable(LLMError):
    """Запрошенной модели нет в локальном хранилище."""


@dataclass
class ChatResult:
    """Ответ модели вместе с телеметрией — она нужна для оценки времени."""

    text: str
    model: str
    task: str
    thinking: str = ""
    prompt_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    done_reason: str = ""

    @property
    def tokens_per_second(self) -> float:
        return self.output_tokens / self.seconds if self.seconds > 0 else 0.0


@dataclass
class HealthReport:
    reachable: bool
    base_url: str
    available: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.reachable and not self.problems

    def as_text(self) -> str:
        lines = [f"Ollama: {self.base_url} — {'доступен' if self.reachable else 'НЕ доступен'}"]
        lines.append(f"Модели: {', '.join(self.available) if self.available else 'нет'}")
        for p in self.problems:
            lines.append(f"  [!] {p}")
        for n in self.notes:
            lines.append(f"  [i] {n}")
        lines.append("Итог: " + ("всё готово" if self.ok else "есть проблемы"))
        return "\n".join(lines)


CLOUD_PROVIDERS: dict[str, dict[str, Any]] = {
    "groq": {
        "name": "⚡ Groq (бесплатно, мгновенный стриминг)",
        "base_url": "https://api.groq.com/openai/v1",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
            "deepseek-r1-distill-llama-70b",
        ],
        "default_model": "llama-3.3-70b-versatile",
        "key_url": "https://console.groq.com/keys",
    },
    "gemini": {
        "name": "✨ Google Gemini API (бесплатный тариф)",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-2.0-flash-lite",
        ],
        "default_model": "gemini-2.0-flash",
        "key_url": "https://aistudio.google.com/app/apikey",
    },
    "openrouter": {
        "name": "🌐 OpenRouter (бесплатные модели :free)",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "deepseek/deepseek-r1:free",
            "meta-llama/llama-3.3-70b-instruct:free",
            "google/gemini-2.0-flash-exp:free",
            "qwen/qwen-2.5-72b-instruct:free",
            "mistralai/mistral-7b-instruct:free",
        ],
        "default_model": "deepseek/deepseek-r1:free",
        "key_url": "https://openrouter.ai/keys",
    },
    "openai": {
        "name": "🤖 OpenAI (официальный ChatGPT / GPT-4o)",
        "base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-4o-mini",
            "gpt-4o",
            "o3-mini",
        ],
        "default_model": "gpt-4o-mini",
        "key_url": "https://platform.openai.com/api-keys",
    },
    "custom": {
        "name": "🛠️ Свой OpenAI-совместимый сервер",
        "base_url": "http://localhost:8000/v1",
        "models": ["gpt-4o-mini", "custom-model"],
        "default_model": "gpt-4o-mini",
        "key_url": "",
    },
}


def fetch_cloud_models(base_url: str, api_key: str, timeout: float = 5.0) -> list[str]:
    """Запросить реальный актуальный список моделей с сервера провайдера по API-ключу."""
    if not base_url or not api_key:
        return []
    clean_url = base_url.strip().rstrip("/")
    endpoint = f"{clean_url}/models"
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    if "openrouter" in clean_url.lower():
        headers["HTTP-Referer"] = "https://github.com/sosiskacorp/DIBot"
        headers["X-Title"] = "DIBot Education Assistant"
    try:
        resp = requests.get(endpoint, headers=headers, timeout=timeout)
        if resp.ok:
            data = resp.json()
            items = data.get("data") or data.get("models") or []
            models: list[str] = []
            for it in items:
                mid = str(it.get("id") or it.get("name") or "").strip()
                if mid:
                    models.append(mid)
            if "openrouter" in clean_url.lower():
                # Поднимаем бесплатные :free модели наверх списка
                free_m = [m for m in models if ":free" in m]
                other_m = [m for m in models if ":free" not in m]
                return sorted(free_m) + other_m[:40]
            return sorted(models)
    except Exception:
        pass
    return []


class LLMClient:
    """Единая точка входа к локальным и облачным моделям."""

    def __init__(
        self,
        config_path: str | Path | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        ollama_api_key: str | None = None,
        cloud_api_key: str | None = None,
        cloud_base_url: str | None = None,
        cloud_model: str | None = None,
        provider: str | None = None,
    ):
        config = self._load_config(config_path)
        self.config = config

        self.provider = (provider or os.environ.get("DIBOT_PROVIDER", "auto")).lower()
        ollama_cfg = config.get("ollama") or {}
        env_ollama = os.environ.get("OLLAMA_BASE_URL", "").strip()
        self.base_url = (base_url or env_ollama or ollama_cfg.get("base_url") or "http://localhost:11434").rstrip("/")
        self.ollama_api_key = (
            (ollama_api_key or "").strip()
            or os.environ.get("OLLAMA_API_KEY", "").strip()
            or os.environ.get("DIBOT_GATEWAY_TOKEN", "").strip()
            or str(ollama_cfg.get("api_key") or "").strip()
        )
        self.timeout = float(timeout or ollama_cfg.get("request_timeout") or 600)
        self.keep_alive = ollama_cfg.get("keep_alive", "5m")
        self._cache_ttl = float(ollama_cfg.get("model_cache_ttl") or 30)

        self.cloud_api_key = (cloud_api_key if cloud_api_key is not None else os.environ.get("OPENAI_API_KEY", "")).strip()
        self.cloud_base_url = (
            (cloud_base_url or "").strip()
            or os.environ.get("OPENAI_BASE_URL", "").strip()
            or "https://api.groq.com/openai/v1"
        ).rstrip("/")
        self.cloud_model = (
            (cloud_model or "").strip()
            or os.environ.get("OPENAI_MODEL", "").strip()
            or "llama-3.3-70b-versatile"
        )

        # Старое имя ключа оставлено для модулей, которые ещё не переписаны.
        self.default_model = config.get("default_model")

        self.router = Router(config)
        self._models_cache: dict[str, list[str]] | None = None
        self._models_cached_at = 0.0
        self._session = requests.Session()
        if self.ollama_api_key:
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.ollama_api_key}",
                    "X-DIBot-Token": self.ollama_api_key,
                }
            )

    # ------------------------------------------------------------- конфиг

    @staticmethod
    def _load_config(config_path: str | Path | None) -> dict[str, Any]:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if not path.is_absolute():
            path = DEFAULT_CONFIG_PATH.parent.parent / path
        if not path.exists():
            raise FileNotFoundError(
                f"Не найден конфиг: {path}\n"
                f"Ожидается файл настроек проекта (обычно config/settings.json)."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------- модели

    def available_models(self, refresh: bool = False) -> list[str]:
        """Имена моделей, реально доступных серверу Ollama."""
        fresh = (time.time() - self._models_cached_at) < self._cache_ttl
        if self._models_cache is not None and fresh and not refresh:
            return sorted(self._models_cache)

        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise OllamaUnavailable(
                f"Нет связи с Ollama по адресу {self.base_url}. "
                f"Запущен ли сервер? Детали: {exc}"
            ) from exc

        names = [m["name"] for m in resp.json().get("models", []) if "name" in m]
        self._models_cache = {name: [] for name in names}
        self._models_cached_at = time.time()
        return sorted(names)

    def has_model(self, model: str) -> bool:
        try:
            return model in self.available_models()
        except OllamaUnavailable:
            return False

    def require_model(self, model: str) -> None:
        available = self.available_models()
        if model not in available:
            raise ModelUnavailable(
                f"Модели {model!r} нет в локальном хранилище.\n"
                f"Доступно: {', '.join(available) or 'ничего'}.\n"
                f"Вариант: ollama pull {model}  (или поправьте routing в config/settings.json)"
            )

    # ------------------------------------------------------------- запросы

    def chat(
        self,
        messages: Messages,
        *,
        task: str | None = None,
        model: str | None = None,
        text_hint: str | None = None,
        think: bool | None = None,
        options: Mapping[str, Any] | None = None,
        images: Sequence[str | Path | bytes] | None = None,
        json_format: bool | str | None = None,
        timeout: float | None = None,
        max_attempts: int = 2,
        on_stream: Any = None,
    ) -> ChatResult:
        """Отправить диалог и получить ответ.

        Модель выбирается так: явный model -> задача task -> угадывание по
        text_hint. Если модель недоступна или сервер моргнул, берётся следующая
        из цепочки запасных.
        """
        prepared = _normalise_messages(messages)
        if images:
            prepared[-1]["images"] = [_encode_image(i) for i in images]

        route: Route = self.router.route(
            task=task,
            text=text_hint if task is None else None,
            has_images=bool(images),
        )
        if model:
            chain = (model, *[m for m in route.chain if m != model])
        else:
            chain = route.chain

        think = route.think if think is None else think
        merged_options = dict(route.options)
        merged_options.update(options or {})

        # Прямое перенаправление на облачный API, если выбран облачный провайдер или модель
        if (
            self.provider == "cloud"
            or (model and (model.startswith("cloud:") or any(model.startswith(p + ":") for p in CLOUD_PROVIDERS)))
        ):
            target_model = model
            if target_model and target_model.startswith("cloud:"):
                target_model = target_model[6:]
            return self._post_cloud_chat(
                messages=prepared,
                options=merged_options,
                json_format=json_format,
                timeout=timeout or 120,
                task=route.task,
                model_override=target_model,
                on_stream=on_stream,
            )

        last_error: Exception | None = None
        for candidate in chain:
            try:
                self.require_model(candidate)
                cand_messages = prepared
                cand_spec = self.router.model_spec(candidate)
                is_vision = cand_spec.get("role") == "vision" or "vl" in candidate.lower() or "vision" in candidate.lower()
                if not is_vision and any("images" in m for m in prepared):
                    cand_messages = []
                    for m in prepared:
                        m_copy = dict(m)
                        if "images" in m_copy:
                            del m_copy["images"]
                            if "[Внимание: изображение" not in str(m_copy.get("content", "")):
                                m_copy["content"] = "[Внимание: к сообщению прикреплено изображение, но резервная модель текстовая]\n" + str(m_copy.get("content", ""))
                        cand_messages.append(m_copy)

                result = self._post_chat(
                    model=candidate,
                    messages=cand_messages,
                    think=think,
                    options=merged_options,
                    json_format=json_format,
                    timeout=timeout or self.timeout,
                    max_attempts=max_attempts,
                    task=route.task,
                    on_stream=on_stream,
                )
                if candidate != chain[0]:
                    log.warning("Задача %s: основная модель недоступна, ответила %s", route.task, candidate)
                return result
            except ModelUnavailable as exc:
                last_error = exc
                log.warning("%s", exc)
                continue
            except OllamaUnavailable as exc:
                last_error = exc
                if self.cloud_api_key:
                    return self._post_cloud_chat(
                        messages=prepared,
                        options=merged_options,
                        json_format=json_format,
                        timeout=timeout or 120,
                        task=route.task,
                        on_stream=on_stream,
                    )
                raise
            except requests.RequestException as exc:
                last_error = exc
                err_detail = ""
                if hasattr(exc, "response") and exc.response is not None:
                    err_detail = f" (детали: {exc.response.text[:300]})"
                log.warning("Сбой запроса к %s: %s%s", candidate, exc, err_detail)
                print(f"[OLLAMA_ERROR] {candidate}: {exc}{err_detail}", flush=True)
                continue

        if self.cloud_api_key:
            return self._post_cloud_chat(
                messages=prepared,
                options=merged_options,
                json_format=json_format,
                timeout=timeout or 120,
                task=route.task,
            )

        raise ModelUnavailable(
            f"Ни одна модель из цепочки {chain} не ответила. Последняя ошибка: {last_error}"
        )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "Ты полезный ассистент.",
        *,
        task: str | None = None,
        model: str | None = None,
        options: Mapping[str, Any] | None = None,
        think: bool | None = None,
        json_format: bool | str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Совместимый со старым кодом вызов: промпт на входе, текст на выходе."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        result = self.chat(
            messages,
            task=task,
            model=model,
            text_hint=prompt,
            options=options,
            think=think,
            json_format=json_format,
            timeout=timeout,
        )
        return result.text

    def stream(
        self,
        messages: Messages,
        *,
        task: str | None = None,
        model: str | None = None,
        think: bool | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> Iterator[str]:
        """Потоковая выдача токенов — для интерфейса."""
        route = self.router.route(task=task)
        chosen = model or route.model
        self.require_model(chosen)

        merged = dict(route.options)
        merged.update(options or {})

        payload = {
            "model": chosen,
            "messages": _normalise_messages(messages),
            "stream": True,
            "think": route.think if think is None else think,
            "options": merged,
            "keep_alive": self.keep_alive,
        }
        with self._session.post(
            f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                piece = (chunk.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if chunk.get("done"):
                    break

    def _post_chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        think: bool,
        options: Mapping[str, Any],
        json_format: bool | str | None,
        timeout: float,
        max_attempts: int,
        task: str,
        on_stream: Any = None,
    ) -> ChatResult:
        """Отправить запрос к Ollama в потоковом режиме (stream=True).

        Потоковый режим критически важен при работе через HTTPS-туннели (Cloudflare):
        байты идут непрерывно по мере генерации токенов, поэтому прокси никогда
        не обрывает соединение по таймауту ожидания (HTTP 524) на долгих рассуждениях.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            "think": think,
            "options": dict(options),
            "keep_alive": self.keep_alive,
        }
        if json_format:
            payload["format"] = json_format

        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            text_parts: list[str] = []
            think_parts: list[str] = []
            resp_model = model
            prompt_tokens = 0
            output_tokens = 0
            done_reason = ""

            try:
                with self._session.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=(15, timeout),
                ) as resp:
                    if resp.status_code == 404:
                        raise ModelUnavailable(
                            f"Ollama не знает модель {model!r}: {_error_message(resp)}"
                        )
                    if not resp.ok:
                        err_msg = _error_message(resp)
                        raise LLMError(f"Ollama HTTP {resp.status_code} ({model}): {err_msg}")

                    for raw_line in resp.iter_lines(chunk_size=1, decode_unicode=True):
                        if not raw_line:
                            continue
                        line = raw_line.strip() if isinstance(raw_line, str) else raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("error"):
                            err_msg = str(chunk["error"])
                            if "not found" in err_msg.lower():
                                raise ModelUnavailable(f"Ollama не знает модель {model!r}: {err_msg}")
                            raise LLMError(f"Ошибка Ollama ({model}): {err_msg}")

                        if chunk.get("model"):
                            resp_model = str(chunk["model"])
                        msg = chunk.get("message") or {}
                        if msg.get("thinking"):
                            think_parts.append(str(msg["thinking"]))
                            if on_stream is not None:
                                try:
                                    on_stream("thinking", "".join(think_parts), resp_model)
                                except Exception:
                                    pass
                        if msg.get("content"):
                            text_parts.append(str(msg["content"]))
                            if on_stream is not None:
                                try:
                                    on_stream("content", "".join(text_parts), resp_model)
                                except Exception:
                                    pass
                        if chunk.get("done"):
                            prompt_tokens = int(chunk.get("prompt_eval_count") or prompt_tokens)
                            output_tokens = int(chunk.get("eval_count") or output_tokens)
                            done_reason = str(chunk.get("done_reason") or "stop")
                            break
                break
            except requests.ConnectionError as exc:
                if attempt >= max_attempts:
                    raise OllamaUnavailable(
                        f"Ollama недоступен ({self.base_url}) после {attempt} попыток: {exc}"
                    ) from exc
                time.sleep(1.5 * attempt)

        elapsed = time.perf_counter() - started
        final_text = "".join(text_parts).strip()
        final_thinking = "".join(think_parts).strip()
        if not final_text and final_thinking:
            cleaned = re.sub(r"^\s*<think>\s*", "", final_thinking, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*</think>\s*$", "", cleaned, flags=re.IGNORECASE).strip()
            final_text = cleaned
            final_thinking = ""

        return ChatResult(
            text=final_text,
            model=resp_model,
            task=task,
            thinking=final_thinking,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            seconds=round(elapsed, 3),
            done_reason=done_reason,
        )

    def _post_cloud_chat(
        self,
        *,
        messages: list[dict[str, Any]],
        options: Mapping[str, Any],
        json_format: bool | str | None,
        timeout: float,
        task: str,
        model_override: str | None = None,
        on_stream: Any = None,
    ) -> ChatResult:
        """Облачный OpenAI-совместимый провайдер (Groq / OpenRouter / Gemini / Custom) со стримингом токенов."""
        oai_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role") or "user"
            content = msg.get("content") or ""
            imgs = msg.get("images") or []
            if imgs:
                parts: list[dict[str, Any]] = [{"type": "text", "text": content}]
                for b64 in imgs:
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        }
                    )
                oai_messages.append({"role": role, "content": parts})
            else:
                oai_messages.append({"role": role, "content": content})

        model_name = (model_override or self.cloud_model or "").strip()
        base_lower = self.cloud_base_url.lower()

        # Автоматическая защита от несовпадения имени модели и провайдера:
        if "openai.com" in base_lower:
            if not model_name or any(s in model_name.lower() for s in ("llama", "deepseek", "qwen", "mistral")):
                model_name = "gpt-4o-mini"
        elif "googleapis.com" in base_lower:
            if not model_name or any(s in model_name.lower() for s in ("llama", "gpt", "deepseek", "qwen")):
                model_name = "gemini-2.0-flash"
        elif "openrouter.ai" in base_lower:
            if not model_name or ("/" not in model_name and not model_name.startswith("gpt")):
                if "llama" in model_name.lower():
                    model_name = "meta-llama/llama-3.3-70b-instruct:free"
                elif "deepseek" in model_name.lower():
                    model_name = "deepseek/deepseek-r1:free"
                else:
                    model_name = "deepseek/deepseek-r1:free"
        elif "groq.com" in base_lower:
            if "/" in model_name:
                model_name = model_name.split("/")[-1].replace(":free", "")
            if not model_name or any(s in model_name.lower() for s in ("gpt", "gemini")):
                model_name = "llama-3.3-70b-versatile"
        elif not model_name:
            model_name = "llama-3.3-70b-versatile"

        do_stream = on_stream is not None and not json_format

        payload: dict[str, Any] = {
            "model": model_name,
            "messages": oai_messages,
            "temperature": float(options.get("temperature", 0.3)),
            "stream": do_stream,
        }
        if json_format:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.cloud_api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter" in self.cloud_base_url.lower():
            headers["HTTP-Referer"] = "https://github.com/sosiskacorp/DIBot"
            headers["X-Title"] = "DIBot Education Assistant"

        started = time.perf_counter()

        if not do_stream:
            resp = self._session.post(
                f"{self.cloud_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            if not resp.ok:
                err_text = resp.text[:400]
                if resp.status_code == 404 or "model_not_found" in err_text:
                    raise LLMError(
                        f"Модель «{model_name}» не найдена на сервере {self.cloud_base_url}. "
                        "В боковом меню выберите актуальную модель или нажмите «Загрузить модели»."
                    )
                raise LLMError(f"Ошибка облачного API ({resp.status_code}): {err_text}")
            data = resp.json()
            elapsed = time.perf_counter() - started
            choice = (data.get("choices") or [{}])[0]
            msg_obj = choice.get("message") or {}
            usage = data.get("usage") or {}
            return ChatResult(
                text=(msg_obj.get("content") or "").strip(),
                model=f"cloud:{data.get('model') or model_name}",
                task=task,
                thinking=(msg_obj.get("reasoning_content") or "").strip(),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                seconds=round(elapsed, 3),
                done_reason=choice.get("finish_reason") or "stop",
            )

        # Режим потоковой передачи токенов в реальном времени (Server-Sent Events)
        try:
            resp = self._session.post(
                f"{self.cloud_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
                stream=True,
            )
            if not resp.ok:
                err_text = resp.text[:400]
                if resp.status_code == 404 or "model_not_found" in err_text:
                    raise LLMError(
                        f"Модель «{model_name}» не найдена на сервере {self.cloud_base_url}. "
                        "В боковом меню выберите актуальную модель или нажмите «Загрузить модели»."
                    )
                raise LLMError(f"Ошибка облачного API ({resp.status_code}): {err_text}")
        except requests.RequestException as exc:
            raise LLMError(f"Ошибка подключения к облачному API: {exc}") from exc

        text_parts: list[str] = []
        thinking_parts: list[str] = []
        display_model = f"cloud:{model_name}"
        done_reason = "stop"
        output_tokens = 0

        for line_raw in resp.iter_lines():
            if not line_raw:
                continue
            line = line_raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except Exception:
                continue

            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            delta = choice.get("delta") or {}
            if choice.get("finish_reason"):
                done_reason = choice["finish_reason"]

            delta_reasoning = delta.get("reasoning_content") or ""
            if delta_reasoning:
                thinking_parts.append(delta_reasoning)
                if on_stream:
                    try:
                        on_stream("thinking", "".join(thinking_parts), display_model)
                    except Exception:
                        pass

            delta_content = delta.get("content") or ""
            if delta_content:
                text_parts.append(delta_content)
                output_tokens += 1
                if on_stream:
                    try:
                        on_stream("content", "".join(text_parts), display_model)
                    except Exception:
                        pass

        elapsed = time.perf_counter() - started
        full_text = "".join(text_parts).strip()
        full_thinking = "".join(thinking_parts).strip()

        if "<think>" in full_text and "</think>" in full_text:
            t_start = full_text.find("<think>") + 7
            t_end = full_text.find("</think>")
            if not full_thinking:
                full_thinking = full_text[t_start:t_end].strip()
            full_text = (full_text[:t_start - 7] + full_text[t_end + 8:]).strip()

        return ChatResult(
            text=full_text,
            model=display_model,
            task=task,
            thinking=full_thinking,
            prompt_tokens=0,
            output_tokens=output_tokens,
            seconds=round(elapsed, 3),
            done_reason=done_reason,
        )

    # ------------------------------------------------------------- проверка

    def health(self) -> HealthReport:
        """Проверить связь, наличие моделей и целостность маршрутов."""
        report = HealthReport(reachable=False, base_url=self.base_url)
        try:
            report.available = self.available_models(refresh=True)
            report.reachable = True
        except OllamaUnavailable as exc:
            report.problems.append(str(exc))
            return report

        for task in self.router.routing:
            route = self.router.route(task)
            if route.model not in report.available:
                report.problems.append(
                    f"задача {task}: основной модели {route.model} нет в хранилище"
                )
            if not any(m in report.available for m in route.fallback):
                report.notes.append(
                    f"задача {task}: ни одна запасная модель не установлена "
                    f"({', '.join(route.fallback) or 'список пуст'})"
                )

        declared = set(self.router.models)
        undeclared = [m for m in report.available if m not in declared]
        if undeclared:
            report.notes.append(
                f"модели вне реестра (не используются для роутинга): {', '.join(undeclared)}"
            )

        if report.reachable and not report.available:
            report.problems.append("сервер отвечает, но моделей нет")
        return report

    def smoke_test(self, timeout: float | None = None) -> list[tuple[str, str, bool, str]]:
        """Короткий прогон по каждой задаче: (задача, модель, успех, текст/ошибка)."""
        results: list[tuple[str, str, bool, str]] = []
        for task in ("chat", "notes", "summary", "math", "code", "latex", "verify"):
            route = self.router.route(task)
            if task == "vision":
                continue
            try:
                res = self.chat(
                    [{"role": "user", "content": "Ответь одним словом: работает"}],
                    task=task,
                    timeout=timeout or 180,
                    max_attempts=1,
                )
                results.append((task, res.model, True, res.text[:60].replace("\n", " ")))
            except LLMError as exc:
                results.append((task, route.model, False, str(exc)[:120]))
        return results

    def close(self) -> None:
        self._session.close()


# ---------------------------------------------------------------- утилиты


def _normalise_messages(messages: Messages) -> list[dict[str, Any]]:
    if not messages:
        raise ValueError("Пустой список сообщений")
    normalised: list[dict[str, Any]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = msg.get("content") or ""
        item: dict[str, Any] = {"role": role, "content": content}
        if msg.get("images"):
            item["images"] = list(msg["images"])
        normalised.append(item)
    return normalised


def _encode_image(source: str | Path | bytes | bytearray) -> str:
    """Ollama принимает base64 без data-префикса."""
    if isinstance(source, (bytes, bytearray)):
        return base64.b64encode(bytes(source)).decode("ascii")
    if isinstance(source, Path):
        return base64.b64encode(source.read_bytes()).decode("ascii")
    if source.startswith("data:"):
        return source.split(",", 1)[1]
    if len(source) < 260 and ("\n" not in source):
        try:
            path = Path(source)
            if path.exists():
                return base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            pass
    return source


def _error_message(resp: requests.Response) -> str:
    try:
        return str(resp.json().get("error") or resp.text)
    except ValueError:
        return resp.text[:200]


def make_client(config_path: str | Path | None = None) -> LLMClient:
    return LLMClient(config_path=config_path)
