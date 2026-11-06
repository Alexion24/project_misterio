import socket
from collections.abc import Iterator
from threading import Thread

import pytest
import uvicorn
from playwright.sync_api import Page, expect

from misterio.api import create_app
from misterio.llm import FakeLlm


class FakeModelSwitcher:
    active = "qwen"

    def active_key(self) -> str:
        return self.active

    def switch(self, key: str) -> None:
        self.active = key


@pytest.fixture(scope="module")
def app_url() -> Iterator[str]:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    config = uvicorn.Config(
        create_app(FakeLlm(), model_switcher=FakeModelSwitcher()),
        host="127.0.0.1",
        port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)
    thread = Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        thread.join(0.01)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    thread.join(timeout=5)


def test_model_switch_keeps_page_and_root_url(page: Page, app_url: str) -> None:
    page.goto(app_url)
    page.locator('select[name="model"]').select_option("gigachat")
    page.get_by_role("button", name="Переключить").click()

    expect(page.get_by_text("Модель переключена.", exact=False)).to_be_visible()
    expect(page).to_have_url(f"{app_url}/")
