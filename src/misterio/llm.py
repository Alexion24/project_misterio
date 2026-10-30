from typing import Protocol

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from misterio.contracts import LlmCallSpec


class LlmError(RuntimeError):
    """Любая ошибка обращения к LLM: сеть, HTTP-статус, неожиданный формат."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MISTERIO_",
        env_file=".env",
        extra="ignore",
    )

    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "local"
    llm_model: str = "qwen3:8b"
    # Локальная модель может долго грузиться при первом вызове
    # и медленно отвечать, поэтому таймаут намного больше, чем у облачного API.
    llm_timeout_s: float = 300.0
    llm_connect_timeout_s: float = 5.0


class Llm(Protocol):
    def complete(self, spec: LlmCallSpec) -> str: ...


class FakeLlm:
    def __init__(self, text: str = "fake-llm-output") -> None:
        self.text = text

    def complete(self, spec: LlmCallSpec) -> str:
        return self.text


class OpenAiCompatibleLlm:
    def __init__(self, settings: Settings, client: httpx.Client | None = None) -> None:
        self._settings = settings
        base_url = settings.llm_base_url.rstrip("/") + "/"
        self._client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            timeout=httpx.Timeout(
                settings.llm_timeout_s,
                connect=settings.llm_connect_timeout_s,
            ),
        )

    def complete(self, spec: LlmCallSpec) -> str:
        try:
            response = self._client.post(
                "chat/completions",
                json={
                    "model": spec.model or self._settings.llm_model,
                    "messages": [{"role": "user", "content": spec.prompt}],
                    "temperature": spec.temperature,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPError as e:
            raise LlmError(f"LLM недоступен или вернул ошибку: {e}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LlmError(f"Неожиданный формат ответа LLM: {e!r}") from e
        return content or ""
