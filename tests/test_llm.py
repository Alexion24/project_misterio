import json

import httpx
import pytest
from pytest_httpx import HTTPXMock

from misterio.contracts import LlmCallSpec
from misterio.llm import FakeLlm, LlmError, OpenAiCompatibleLlm, Settings


def _llm() -> OpenAiCompatibleLlm:
    return OpenAiCompatibleLlm(
        Settings(
            llm_base_url="http://llm.test/v1",
            llm_api_key="sk-test",
            llm_model="test-model",
        )
    )


def _spec() -> LlmCallSpec:
    return LlmCallSpec(model="test-model", prompt="hello")


def test_fake_llm_is_deterministic() -> None:
    llm = FakeLlm(text="fixed")
    spec = LlmCallSpec(model="unused", prompt="про трактор")
    assert llm.complete(spec) == "fixed"
    assert llm.complete(spec) == "fixed"


def test_openai_compatible_complete_uses_httpx_mock(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        method="POST",
        url="http://llm.test/v1/chat/completions",
        json={"choices": [{"message": {"content": "mocked-reply"}}]},
    )
    assert _llm().complete(_spec()) == "mocked-reply"

    request = httpx_mock.get_request()
    body = json.loads(request.content)
    assert request.headers["Authorization"] == "Bearer sk-test"
    assert body["model"] == "test-model"
    assert body["messages"] == [{"role": "user", "content": "hello"}]


def test_http_status_error_becomes_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(status_code=500)
    with pytest.raises(LlmError):
        _llm().complete(_spec())


def test_connection_error_becomes_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_exception(httpx.ConnectError("refused"))
    with pytest.raises(LlmError):
        _llm().complete(_spec())


def test_malformed_response_becomes_llm_error(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(json={"unexpected": "shape"})
    with pytest.raises(LlmError):
        _llm().complete(_spec())
