"""Where every constant in this model came from.

Review's requirement, stated directly: it must be possible to say which numbers
are bulletproof -- read from a cited table -- and which were predicted, with no
credit given to a predicted number for landing in a plausible range.

Four categories, mutually exclusive:

  SOURCED     read directly from a cited table. `detail` carries document,
              table and page. Nothing else counts as sourced.
  DERIVED     computed from SOURCED values by a stated exact relation. The
              relation is citable; the number is not independently checkable.
  CALIBRATED  fitted so the model reproduces a SOURCED number to a stated
              tolerance. The fit target is an input.
  DECLARED    chosen. Not derivable from any source this project holds.
              `detail` must carry the sensitivity range.

`inputs` names other ledger entries. It is what makes a DERIVED number's chain
walkable back to something SOURCED, and test_provenance.py asserts the chain
exists, is acyclic, and bottoms out.

This module holds no aircraft data. It holds statements ABOUT data, so that
adding a constant without saying where it came from fails the build.
"""

from typing import NamedTuple

CATEGORIES = ("SOURCED", "DERIVED", "CALIBRATED", "DECLARED")


class Entry(NamedTuple):
    """One constant's provenance.

    `detail` is free text because the four categories need different things
    from it -- a citation, a relation, a fit target, a sensitivity range -- and
    a schema rigid enough to hold all four would be harder to read than the
    prose it replaced. What is NOT free text is `category` and `inputs`, which
    are what the tests actually enforce.
    """

    category: str
    detail: str
    inputs: tuple[str, ...] = ()


LEDGER: dict[str, Entry] = {}
