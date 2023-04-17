from pathlib import Path
import pytest


@pytest.fixture
def resource_dir() -> Path:
    return Path(__file__).parent.parent / "resources"
