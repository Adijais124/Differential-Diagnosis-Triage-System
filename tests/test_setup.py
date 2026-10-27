"""Initial smoke test verifying pytest setup and environment."""
import sys


def test_environment_smoke():
    """Verify that the test suite runs and python is available."""
    assert sys.version_info >= (3, 10), "Python version must be 3.10+ (recommended 3.11)"
