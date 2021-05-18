import os
import shutil

import pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_output():
    yield

    # Cleanup output directory after all tests execute.
    if os.path.exists("output/"):
        shutil.rmtree("output/")
