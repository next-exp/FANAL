import os
import pytest

from hypothesis import settings
from hypothesis import Verbosity

@pytest.fixture(scope = 'session')
def FANAL_DIR():
    return os.environ['FANALPATH']


@pytest.fixture(scope = 'session')
def FANAL_DATA_DIR(FANAL_DIR):
    return os.path.join(FANAL_DIR, "data/")
