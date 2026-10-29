from typing import Protocol

import httpx
from pydantic_settings import BaseSettings, SettingsConfigDict

from misterio.contracts import LlmCallSpec


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MISTERIO_")

    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_api_key: str = "local"
    llm_model: str = "llama3.2"


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
            timeout=30.0,
        )

    def complete(self, spec: LlmCallSpec) -> str:
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
        return content or ""
