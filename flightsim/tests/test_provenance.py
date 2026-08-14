"""The provenance ledger's own rules.

Review asked for an unambiguous separation between numbers taken from cited
tables and numbers that were predicted, with no credit given to a calibrated
value for landing in a plausible range. This file is what makes that
separation enforceable rather than aspirational: a constant added without a
ledger entry fails the build.

The precedent is `validation.Reference`, whose mandatory `source` field is
asserted by test_validation.py. This is the same idea applied to every
constant rather than only to published reference values.
"""

import pytest

from flightsim import provenance
from flightsim.provenance import LEDGER, Entry


def test_every_entry_uses_one_of_the_four_categories():
    """Four categories, mutually exclusive. A fifth would mean the distinction
    review asked for has been blurred."""
    for name, entry in LEDGER.items():
        assert entry.category in provenance.CATEGORIES, (
            f"{name} has category {entry.category!r}, "
            f"which is not one of {provenance.CATEGORIES}"
        )


def test_sourced_and_declared_entries_carry_a_usable_detail():
    """A SOURCED entry without document, table and page is not sourced, it is
    asserted. A DECLARED entry without its sensitivity is not declared, it is
    hidden. Length 20 is the same bar test_validation.py sets on
    `Reference.source`."""
    for name, entry in LEDGER.items():
        if entry.category in ("SOURCED", "DECLARED"):
            assert len(entry.detail) > 20, f"{name} has no usable detail"


def test_derived_and_calibrated_entries_name_inputs_that_exist():
    """A DERIVED value is only as good as what it was derived from, so its
    inputs must themselves be in the ledger and reachable."""
    for name, entry in LEDGER.items():
        if entry.category in ("DERIVED", "CALIBRATED"):
            assert entry.inputs, f"{name} is {entry.category} but names no inputs"
            for dep in entry.inputs:
                assert dep in LEDGER, f"{name} depends on {dep}, which is not in the ledger"


def test_the_dependency_graph_has_no_cycles():
    """A cycle would let two numbers justify each other with nothing underneath.
    Every chain must bottom out in SOURCED or DECLARED entries."""
    for name in LEDGER:
        seen = set()
        stack = [name]
        while stack:
            current = stack.pop()
            assert current not in seen, f"{name} has a cyclic dependency via {current}"
            seen.add(current)
            stack.extend(LEDGER[current].inputs)


def test_an_entry_with_an_unknown_category_is_rejected():
    """The rules must be able to fail. A test that can only pass demonstrates
    nothing -- PROJECT.md's falsification rule applied to this file."""
    bad = Entry(category="PROBABLY_FINE", detail="x" * 30)
    assert bad.category not in provenance.CATEGORIES
