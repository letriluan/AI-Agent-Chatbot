import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest
from evals.fixture import create_database


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "fixture.sqlite"
    create_database(path)
    return str(path)
