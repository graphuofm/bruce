"""Pytest configuration for bruce-py tests."""
import sys
import pytest


def pytest_report_header(config):
    """Show which bruce installation we're testing."""
    try:
        import bruce
        return [
            f"bruce: {bruce.__version__} from {bruce.__file__}",
            f"python: {sys.version.splitlines()[0]}",
        ]
    except ImportError:
        return ["bruce: NOT INSTALLED"]
