import json

from pytest_httpx import HTTPXMock

from misterio.contracts import LlmCallSpec
from misterio.llm import FakeLlm, OpenAiCompatibleLlm, Settings


def test_fake_llm_is_deterministic() -> None:
    llm = FakeLlm(text="fixed")
    spec = LlmCallSpec(model="unused", prompt="про трактор")
    assert llm.complete(spec) == "fixed"
    assert llm.complete(spec) == "fixed"


def test_openai_compatible_complete_uses_httpx_mock(httpx_mock: HTTPXMock) -> None:
    settings = Settings(
        llm_base_url="http://llm.test/v1",
        llm_api_key="sk-test",
        llm_model="test-model",
    )
    httpx_mock.add_response(
        method="POST",
        url="http://llm.test/v1/chat/completions",
        json={"choices": [{"message": {"content": "mocked-reply"}}]},
    )
    llm = OpenAiCompatibleLlm(settings)
    text = llm.complete(LlmCallSpec(model=settings.llm_model, prompt="hello"))
    assert text == "mocked-reply"

    request = httpx_mock.get_request()
    body = json.loads(request.content)
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
