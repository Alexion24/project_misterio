from dataclasses import dataclass
from typing import Literal, Protocol

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

    llm_api_key: str = ""


@dataclass(frozen=True)
class LlmConfig:
    base_url: str
    model: str
    api_style: Literal["openai", "ollama"]
    requires_api_key: bool = False
    timeout_s: float = 300.0
    connect_timeout_s: float = 5.0


class Llm(Protocol):
    def complete(self, spec: LlmCallSpec) -> str: ...


class FakeLlm:
    def __init__(self, text: str = "fake-llm-output") -> None:
        self.text = text

    def complete(self, spec: LlmCallSpec) -> str:
        return self.text


class OpenAiCompatibleLlm:
    def __init__(
        self,
        config: LlmConfig,
        api_key: str = "",
        client: httpx.Client | None = None,
    ) -> None:
        self._config = config
        base_url = config.base_url.rstrip("/") + "/"
        self._ollama_chat_url = None
        if config.api_style == "ollama":
            self._ollama_chat_url = base_url.removesuffix("v1/") + "api/chat"
        headers = (
            {"Authorization": f"Bearer {api_key}"} if config.requires_api_key else None
        )
        self._client = client or httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(
                config.timeout_s,
                connect=config.connect_timeout_s,
            ),
        )

    def complete(self, spec: LlmCallSpec) -> str:
        try:
            if self._ollama_chat_url:
                response = self._client.post(
                    self._ollama_chat_url,
                    json={
                        "model": spec.model or self._config.model,
                        "messages": [{"role": "user", "content": spec.prompt}],
                        "think": False,
                        "stream": False,
                        "options": {
                            "temperature": spec.temperature,
                            "top_p": spec.top_p,
                            "num_predict": spec.max_tokens,
                        },
                    },
                )
            else:
                response = self._client.post(
                    "chat/completions",
                    json={
                        "model": spec.model or self._config.model,
                        "messages": [{"role": "user", "content": spec.prompt}],
                        "temperature": spec.temperature,
                        "top_p": spec.top_p,
                        "max_tokens": spec.max_tokens,
                    },
                )
            response.raise_for_status()
            data = response.json()
            content = (
                data["message"]["content"]
                if self._ollama_chat_url
                else data["choices"][0]["message"]["content"]
            )
        except httpx.HTTPError as e:
            raise LlmError(f"LLM недоступен или вернул ошибку: {e}") from e
        except (KeyError, IndexError, TypeError, ValueError) as e:
            raise LlmError(f"Неожиданный формат ответа LLM: {e!r}") from e
        return content or ""
