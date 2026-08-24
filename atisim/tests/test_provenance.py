"""The provenance ledger's own rules.

Review asked for an unambiguous separation between numbers taken from cited
tables and numbers that were predicted, with no credit given to a calibrated
value for landing in a plausible range. This file makes the separation itself
enforceable: categories are exclusive, DERIVED chains name inputs that exist,
the graph is acyclic, and every chain bottoms out.

WHAT IT DOES NOT DO is check COVERAGE -- every test here iterates `LEDGER`
against itself, so a constant that never got an entry is invisible to all of
them. This docstring used to end "a constant added without a ledger entry fails
the build", which described a test that did not exist. The one that does is
`test_audit_regression.py::test_the_provenance_ledger_does_not_cover_the_source_modules`,
and its scope is stated in `provenance.py`'s own docstring.

The precedent is `validation.Reference`, whose mandatory `source` field is
asserted by test_validation.py. This is the same idea applied to every
constant rather than only to published reference values.
"""

import pytest

from atisim import provenance
from atisim.provenance import LEDGER, Entry


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
    Every chain must bottom out in SOURCED or DECLARED entries.

    A node reached twice by two DIFFERENT paths is a diamond, not a cycle, and
    is legal: one SOURCED number can feed several DERIVED ones, which happens
    as soon as a reference length appears under more than one coefficient. What
    is illegal is a node appearing twice on the SAME path, because that is the
    chain closing on itself. So the walk carries the path it took rather than a
    set of everything it has ever seen -- the latter cannot tell the two apart
    and rejects the diamond.
    """
    for name in LEDGER:
        stack = [(name, (name,))]
        while stack:
            current, path = stack.pop()
            for dep in LEDGER[current].inputs:
                assert dep not in path, f"{name} has a cyclic dependency via {dep}"
                stack.append((dep, path + (dep,)))


def test_an_entry_with_an_unknown_category_is_rejected():
    """The rules must be able to fail. A test that can only pass demonstrates
    nothing -- PROJECT.md's falsification rule applied to this file."""
    bad = Entry(category="PROBABLY_FINE", detail="x" * 30)
    assert bad.category not in provenance.CATEGORIES


def test_the_747_reference_geometry_is_sourced_from_cr2144():
    """These three are the foundation everything else in this work rests on.
    If they are ever reclassified, the chain above them is no longer sourced."""
    for name in ("b747.S", "b747.b", "b747.c"):
        assert LEDGER[name].category == "SOURCED"
        assert "IX-3" in LEDGER[name].detail, f"{name} must cite its table"


def test_the_effective_tail_arm_is_derived_and_never_sourced():
    """It is a ratio of two tabulated derivatives, not a measured dimension.
    Quoting it as 747 geometry would be a category error -- design section 7d."""
    entry = LEDGER["b747.l_eff"]
    assert entry.category == "DERIVED"
    assert set(entry.inputs) == {"b747.Cmq", "b747.CLq", "b747.c"}


def test_the_loading_shape_is_declared_and_carries_its_sensitivity():
    """Taper ratio is not in CR-2144 and is not recoverable from S, b and cbar
    (design section 3f), so the shape is a choice and must be reported as one."""
    entry = LEDGER["strip.loading_shape"]
    assert entry.category == "DECLARED"
    assert "sensitivit" in entry.detail.lower()


def test_the_calibrated_lift_slope_names_the_number_it_is_pinned_to():
    """A calibrated value with no stated target is just a number."""
    entry = LEDGER["strip.lift_slope"]
    assert entry.category == "CALIBRATED"
    assert "b747.Clp" in entry.inputs
