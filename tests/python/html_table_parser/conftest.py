from pathlib import Path
from typing import Dict
import pytest


@pytest.fixture(scope="session")
def html_files() -> Dict[str, str]:
    folder = Path(__file__).parent.parent / "resources/html_table_parser"
    output = {}
    for html_file in folder.glob("*.html"):
        output[html_file.name] = html_file.read_text()
    return output
