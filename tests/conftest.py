import pytest

from misterio.models import DEFAULT_MODEL_KEY, MODEL_OPTIONS


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--model-profile",
        action="store",
        default=DEFAULT_MODEL_KEY,
        choices=tuple(MODEL_OPTIONS),
        help="model profile used by explicitly enabled model evaluations",
    )
