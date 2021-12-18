from pathlib import Path
from typing import Dict
import pytest
import sm.misc as M


@pytest.fixture(scope="session")
def html_files() -> Dict[str, str]:
    folder = Path(__file__).parent.parent / "resources/html_table_parser"
    output = {}
    for html_file in folder.glob("*.html"):
        output[html_file.name] = M.deserialize_text(html_file)
    return output
