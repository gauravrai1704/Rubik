import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: long-running test (run with -m slow)")
