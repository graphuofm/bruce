"""Package-surface parity: every native symbol is exported AND typed.

Three surfaces must agree, or a user hits one of them and not the others:

  1. ``bruce._bruce``            — what PyO3 actually registered
  2. ``bruce.__all__``           — what ``import bruce`` gives you
  3. ``bruce/_bruce.pyi``        — what a type checker and an IDE see

The gap this file exists to prevent: ``KvSnapshot`` shipped registered in
(1) but missing from (2), and ``QuerySession`` — the entry point to the
whole query layer — shipped missing from (3). Both were invisible to
every other test, because every other test imports what it needs
directly. Adding a ``#[pyclass]`` or ``#[pyfunction]`` without touching
``__init__.py`` and ``_bruce.pyi`` now fails here instead of shipping.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import bruce
import bruce._bruce as native


def _native_public() -> set[str]:
    """Classes and functions PyO3 registered on the extension module."""
    out = set()
    for name in dir(native):
        if name.startswith("_"):
            continue
        obj = getattr(native, name)
        if inspect.isclass(obj) or inspect.isroutine(obj):
            out.add(name)
    return out


def _stub_path() -> Path:
    """The .pyi that ships beside the extension module."""
    return Path(bruce.__file__).parent / "_bruce.pyi"


def _stub_toplevel() -> set[str]:
    tree = ast.parse(_stub_path().read_text())
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_every_native_symbol_is_re_exported():
    missing = sorted(_native_public() - set(bruce.__all__))
    assert not missing, (
        f"registered in _bruce but absent from bruce.__all__: {missing}. "
        "Add them to the import block AND __all__ in bruce/__init__.py."
    )


def test_every_exported_name_is_importable():
    for name in bruce.__all__:
        assert hasattr(bruce, name), f"bruce.__all__ names {name}, which is not importable"


def test_every_native_symbol_is_typed():
    missing = sorted(_native_public() - _stub_toplevel())
    assert not missing, (
        f"registered in _bruce but absent from _bruce.pyi: {missing}. "
        "A type checker cannot see these; add stubs."
    )


def test_stub_has_no_stale_entries():
    """A stub for something that no longer exists is worse than no stub:
    it type-checks code that fails at import."""
    py_only = {"Any", "np", "npt"}  # imported names, not declarations
    stale = sorted(_stub_toplevel() - _native_public() - py_only)
    assert not stale, f"_bruce.pyi declares symbols the extension does not provide: {stale}"


def test_stub_parses_and_ships_with_the_wheel():
    path = _stub_path()
    assert path.exists(), f"no type stub next to the installed extension ({path})"
    ast.parse(path.read_text())


@pytest.mark.parametrize("name", ["QuerySession", "KvMemory", "KvSnapshot", "Operator"])
def test_headline_classes_reachable_from_the_package_root(name):
    """Spot-pin the classes a user is most likely to reach for first."""
    assert inspect.isclass(getattr(bruce, name))
