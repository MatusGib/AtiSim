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


# NASA CR-2144, Heffley & Jewell, "Aircraft Handling Qualities Data", December
# 1972, Section IX. Table IX-3 is "B-747 DIMENSIONAL, MASS AND FLIGHT CONDITION
# PARAMETERS", printed page 229; its header carries the reference geometry and
# its columns carry one flight condition each. Flight condition 9 is the cruise
# case this project uses. Verified against the document, not against a summary.
_CR2144_IX3 = "NASA CR-2144 Table IX-3 header, printed p.229, verified against the document"
_CR2144_IX4 = "NASA CR-2144 Table IX-4, printed p.230, flight condition 9"
_CR2144_IX8 = "NASA CR-2144 Table IX-8, printed p.234, flight condition 9, primed"

LEDGER: dict[str, Entry] = {
    # -- 747 reference geometry, straight off the table --------------------
    "b747.S": Entry("SOURCED", f"5500 ft^2 wing area. {_CR2144_IX3}"),
    "b747.b": Entry("SOURCED", f"195.68 ft wing span. {_CR2144_IX3}"),
    "b747.c": Entry("SOURCED", f"27.31 ft mean aerodynamic chord. {_CR2144_IX3}"),

    # -- the derivatives the tail arm is built from -------------------------
    # Dimensional in the source; the non-dimensionalisation is CR-2144
    # Appendix A's own relation, applied in aircraft.py.
    "b747.Zq": Entry("SOURCED", f"-5.16, dimensional. {_CR2144_IX4}"),
    "b747.Mq": Entry("SOURCED", f"-0.339, dimensional. {_CR2144_IX4}"),
    "b747.CLq": Entry(
        "DERIVED",
        "CLq = -Zq * 2 * m * U0 / (qS * c), CR-2144 Appendix A. Gives 5.9450.",
        inputs=("b747.Zq", "b747.c"),
    ),
    "b747.Cmq": Entry(
        "DERIVED",
        "Cmq = Mq * 2 * Iy * U0 / (qS * c^2), CR-2144 Appendix A. Gives -23.9232.",
        inputs=("b747.Mq", "b747.c"),
    ),
    "b747.Clp": Entry("SOURCED", f"Lp' = -0.465, primed dimensional. {_CR2144_IX8}"),

    # -- the one new relation this work introduces --------------------------
    "b747.l_eff": Entry(
        "DERIVED",
        "l_eff/c = -Cmq/CLq. Stengel Flight Dynamics 2nd ed eqs. 3.4-7 and "
        "3.4-12; the tail lift slope cancels in the ratio. Gives 4.0241 chords "
        "= 109.90 ft. NOT 747 geometry: it attributes both derivatives to the "
        "tail, and the wing/fuselage share is not separated. Justified "
        "empirically -- the value falls inside the real aircraft's 100-110 ft "
        "-- not by a computed error bar. See design section 3d and 7d.",
        inputs=("b747.Cmq", "b747.CLq", "b747.c"),
    ),

    # -- strip model, phase 3 ------------------------------------------------
    "strip.loading_shape": Entry(
        "DECLARED",
        "Elliptic spanwise loading. Taper ratio is not tabulated in CR-2144 and "
        "is NOT recoverable from S, b and cbar -- the required shape factor "
        "0.72873 lies below the trapezoidal minimum of 0.75, because the 747 "
        "planform is cranked (design section 3f). Sensitivity is mandatory: "
        "every result re-run against uniform and taper-based shapes with the "
        "spread reported.",
    ),
    "strip.lift_slope": Entry(
        "CALIBRATED",
        "Effective section lift slope, scaled so integrating a rigid roll rate "
        "reproduces the tabulated Clp exactly. For elliptic loading the strip "
        "integral gives Clp_hat = -a0/8, hence a0 = -8*Clp. This is an "
        "EFFECTIVE value absorbing sweep, the tail's share of Clp, and the "
        "difference between elliptic strip theory and the real wing. It is not "
        "an airfoil property and must not be quoted as one.",
        inputs=("b747.Clp", "strip.loading_shape"),
    ),
    "strip.n_stations": Entry(
        "DECLARED",
        "Number of spanwise and longitudinal sample stations. Chosen by "
        "convergence, not by taste: Task 6 runs the refinement study and the "
        "sensitivity is the movement between the chosen count and double it.",
    ),
    "airframe.tail_arm_band": Entry(
        "DECLARED",
        "Plausibility band [2.0, 6.0] chords on the derived l_eff. Brackets "
        "conventional tail-aft configurations. Its only job is to reject "
        "derivative sets whose CLq and Cmq disagree about what aircraft they "
        "describe, not to police physics. Measured outcome: boeing747 4.0241 "
        "and boeing747_approach 3.8519 pass; cessna172 0.8558 and cherokee "
        "1.2802 fail and are excluded from the strip path. Sensitivity: no "
        "result depends on the band's edges, only on which aircraft pass, and "
        "the two groups are separated by a factor of three.",
    ),
}
