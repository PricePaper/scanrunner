"""Shared pytest fixtures for scanrunner tests.

All tests run against real artifacts (real images, real Odoo harness, real
SQLite). Mocks are forbidden by project policy.
"""

import sys
from pathlib import Path

# Make the single-file docscanner.py importable as a module without packaging.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest

INV_GOOD = PROJECT_ROOT / "inv" / "good"
INV_UNREADABLE = PROJECT_ROOT / "inv" / "unreadable"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: tests that take long enough to skip by default; "
        "run with `pytest -m slow` to include.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip @pytest.mark.slow by default; opt in with `-m slow`."""
    if config.getoption("-m"):
        return
    skip_slow = pytest.mark.skip(reason="slow; opt in with `-m slow`")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def good_invoice_paths() -> list[Path]:
    paths = sorted(INV_GOOD.glob("*.jpg"))
    if not paths:
        pytest.skip(f"No sample invoices in {INV_GOOD}")
    return paths


@pytest.fixture(scope="session")
def first_good_invoice(good_invoice_paths: list[Path]) -> Path:
    return good_invoice_paths[0]


@pytest.fixture(scope="session")
def unreadable_paths() -> list[Path]:
    paths = sorted(INV_UNREADABLE.glob("*.png"))
    if not paths:
        pytest.skip(f"No unreadable samples in {INV_UNREADABLE}")
    return paths
