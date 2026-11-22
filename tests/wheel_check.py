"""Entry point for the wheel test. Not collected by pytest itself."""

import os
import sys

import pytest

os.environ["PYFYLR_REQUIRE_INSTALLED"] = "1"
here = os.path.dirname(os.path.abspath(__file__))
target = os.path.join(here, "test_release_wheel.py")
sys.exit(pytest.main(["-q", "--tb=line", "-p", "no:cacheprovider", target]))
