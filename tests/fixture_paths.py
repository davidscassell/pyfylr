"""Fixture roots for the monorepo and a standalone pyfylr tree.

In this monorepo, fixtures live at the repository root. An export places the
same tree at tests/fixtures, next to these tests.
"""
from pathlib import Path


def fixtures_dir() -> Path:
    tests = Path(__file__).resolve().parent
    packaged = tests / "fixtures"
    monorepo = tests.parent.parent / "fixtures"
    if packaged.is_dir():
        return packaged
    return monorepo
