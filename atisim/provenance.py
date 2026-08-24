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

This module holds no aircraft data. It holds statements ABOUT data.

WHAT IS ACTUALLY ENFORCED, stated precisely because this docstring used to
overstate it. It said "adding a constant without saying where it came from fails
the build", and nothing checked that: every test in `test_provenance.py` iterates
`LEDGER` against itself, so the entries are checked for internal consistency --
categories, inputs, acyclicity, the chain bottoming out -- and NOTHING checked
COVERAGE. The audit measured what the claim was worth: about 346 non-trivial
numeric literals across nine physics modules against 13 entries, roughly 2%.

The direction that was missing now exists, as
`test_audit_regression.py::test_the_provenance_ledger_does_not_cover_the_source_modules`.
It parses the MODULE-LEVEL numeric constants of five physics modules -- `aero`,
`airframe`, `atmosphere`, `trim`, `wind` -- with `ast`, and fails if one appears
that is neither in this ledger nor in that test's recorded baseline. So the true
statement is narrower than the old one and worth having:

    a NEW module-level constant in one of those five modules, added without a
    ledger entry, fails the build.

Constants inside functions, in the other four physics modules, and the aircraft
data in `aircraft.py` are NOT covered. The baseline set may only ever shrink;
widening it to admit a new constant is the one move that would make the check
meaningless.
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
        "spread reported. MEASURED on a cubic spanwise profile, the case an "
        "equivalent rate cannot represent at all and therefore an upper bound "
        "rather than a typical value: elliptic 1.967e-06, tapered 2.018e-06, "
        "uniform 3.148e-06, a full spread of 49.7% of the mean. Read that as "
        "two numbers, not one: the two shapes that actually taper toward the "
        "tips agree to 2.6%, and the 49.7% is driven entirely by the uniform "
        "shape, which loads the tips where a cubic gust is largest and is not "
        "a defensible transport planform. It is in the sweep as a bracket, not "
        "as a candidate. Quote 2.6% as the shape cost and 49.7% as the bound.",
    ),
    "strip.lift_slope": Entry(
        "CALIBRATED",
        "Effective section lift slope, scaled so integrating a rigid roll rate "
        "reproduces the tabulated Clp -- IN THE CONTINUUM LIMIT, which is where "
        "the identity holds and which is not the station count the code ships "
        "with; see strip.n_stations for the 82.6% that costs and for why it was "
        "left alone. For elliptic loading the strip "
        "integral gives Clp_hat = -a0/8, hence a0 = -8*Clp. This is an "
        "EFFECTIVE value absorbing sweep, the tail's share of Clp, and the "
        "difference between elliptic strip theory and the real wing. It is not "
        "an airfoil property and must not be quoted as one.",
        inputs=("b747.Clp", "strip.loading_shape"),
    ),
    "strip.n_stations": Entry(
        "DECLARED",
        "Number of spanwise and longitudinal sample stations, N_SPAN = N_LON = "
        "9. Odd, so a station sits on the centreline and the symmetric pair "
        "cancels exactly. "
        "SENSITIVITY, MEASURED AND NOT FLATTERING: at 9 stations the strip "
        "integral returns 82.6% of the Clp it is calibrated against -- a -17.4% "
        "understatement -- because the elliptic chord is sqrt-singular at the "
        "tips and the trapezoidal rule cannot resolve it. Observed convergence "
        "order 1.50, stable across every refinement from 21->41 to 1281->2561: "
        "0.922 at 15 stations, 0.954 at 21, 0.983 at 41, 0.994 at 81, 0.9985 at "
        "201. About 19 stations for 5% and 56 for 1%. "
        "DECISION, taken deliberately in the remediation pass and recorded so it "
        "is not re-taken silently: the count STAYS AT 9 and "
        "strip.lift_slope stays calibrated in the continuum. The two repairs "
        "mean different things -- raising N keeps a0 meaning what its docstring "
        "says and costs field evaluations per step; recalibrating a0 at the "
        "shipped N makes the identity exact in the code as run but makes a0 "
        "depend on N, changing what the constant is. Neither is settled by any "
        "source this project holds, and switching calibration basis silently is "
        "the one move that would be wrong. What it costs today is nothing "
        "quoted: the Parks core has no spanwise variation, so point and strip "
        "paths agree to 0.000000 m there. See docs/ASSUMPTIONS.md F5.",
    ),
    # -- microburst, from the paper held in refs/ -----------------------------
    # Oseguera & Bowles, NASA TM-100632, July 1988, held at
    # refs/NASA-TM-100632-Oseguera-Bowles-1988.pdf. Printed pp. 4-5 and the
    # appendix's "From TASS" block, re-read during the remediation pass.
    #
    # THE FOUR ARE NOT FOUR INDEPENDENT NUMBERS, and reading them as such was a
    # documented error in wind.py until that pass. The paper's own order is
    # "Analysis of TASS data indicated ... z_m/z* = 0.22", then "Recalling that
    # z_m/z* = 0.22, the values 1.1212 and 12.5 were obtained from iteration",
    # then u_max = 0.2357*lambda*R from those. One empirical input and three
    # consequences -- of which 1.1212 happens also to be independently
    # obtainable, since it solves exp(-x^2)(2x^2+1) = 1 with no z in it.
    "wind.MICROBURST_ZM_OVER_ZSTAR": Entry(
        "SOURCED",
        "0.22, the altitude of maximum outflow over z*. THE EMPIRICAL INPUT of "
        "the four, taken from TASS numerical-model data. Oseguera & Bowles, "
        "NASA TM-100632, printed p.4 and the appendix's 'From TASS' block.",
    ),
    "wind.MICROBURST_PEAK_RADIUS_RATIO": Entry(
        "SOURCED",
        "1.1212, r/R at maximum outflow. Oseguera & Bowles, NASA TM-100632, "
        "printed p.5, obtained there by iteration. Also the root of "
        "exp(-x^2)(2x^2+1) = 1, which makes it the one member of the set that "
        "does not rest on the 0.22 input.",
    ),
    "wind.MICROBURST_ZSTAR_OVER_EPS": Entry(
        "SOURCED",
        "12.5, out-of-boundary-layer scale over in-boundary-layer scale. "
        "Oseguera & Bowles, NASA TM-100632, printed p.5, obtained by iteration "
        "GIVEN z_m/z* = 0.22 -- so it is a consequence of that input, not a "
        "check on it.",
        inputs=("wind.MICROBURST_ZM_OVER_ZSTAR",),
    ),
    "wind.MICROBURST_UMAX_COEFF": Entry(
        "SOURCED",
        "0.2357 in u_max = 0.2357 * lambda * R. Oseguera & Bowles, NASA "
        "TM-100632, printed p.5, stated there as following from 1.1212 and "
        "12.5.",
        inputs=("wind.MICROBURST_PEAK_RADIUS_RATIO", "wind.MICROBURST_ZSTAR_OVER_EPS"),
    ),

    # -- declared numbers whose sensitivity is already written at the constant --
    "wind.LEE_WAVE_WAVELENGTH": Entry(
        "DECLARED",
        "25 km. Doyle et al. 2011 gives 20-35 km for the TROPOSPHERE and says "
        "stratospheric wavelengths are shorter without quantifying them, so no "
        "held source supplies the number this project needs. 25 km is the "
        "middle of the band the paper does give. Sensitivity: it does not move "
        "the F-factor peak at all -- with no horizontal perturbation the index "
        "is -w/V, independent of wavelength -- but it sets the encounter "
        "duration and the pitching gust rate, so any result depending on those "
        "must state the value used.",
    ),
    "aero.V_MIN": Entry(
        "DECLARED",
        "1.0 m/s airspeed floor. A NaN guard, not a physical scale: alpha, beta "
        "and the three non-dimensional rates divide by V. Sensitivity: "
        "`jnp.maximum(x, 1.0)` returns x exactly for x >= 1, so it cannot move "
        "anything at or above 1 m/s, bit for bit -- asserted by "
        "test_the_airspeed_floor_no_longer_reaches_dynamic_pressure. Below "
        "1 m/s it changes beta and the rates and NOT dynamic pressure, which is "
        "built from the true airspeed; applying it to qbar as well used to make "
        "a stationary airframe produce 1.4 to 171 N out of still air.",
    ),
    "trim.ALPHA_LIMIT": Entry(
        "DECLARED",
        "15 deg, the sense gate in `trim.is_physical`. PROJECT.md section 7 "
        "puts the linear-aero ceiling at 10-12 deg, so this sits a little "
        "beyond the amber band and never rejects a legitimate trim. "
        "Sensitivity: every aircraft in the registry trims at 3-6 deg at its "
        "own cruise condition, so the gate is nowhere near binding on real "
        "data -- which is the property that lets it be applied unconditionally, "
        "and is asserted as the positive control in test_trim.py.",
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
