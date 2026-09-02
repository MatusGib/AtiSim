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
_MEHTA = (
    "R. S. Mehta, 'Modeling Clear-Air Turbulence with Vortices Using "
    "Parameter-Identification Techniques', J. Guidance, Control & Dynamics "
    "10(1), Jan-Feb 1987, 27-31 (AIAA 84-2083), converged five-vortex "
    "solution on p. 30"
)
_LESTER = (
    "P. F. Lester, O. Sen, R. E. Bach Jr., 'The Use of DFDR Information in the "
    "Analysis of a Turbulence Incident over Greenland', Mon. Wea. Rev. 117, "
    "May 1989, 1103-1107"
)

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
    # -- the superseded Hannibal core radius --------------------------------
    # Not used by any computation. It is here so the value the project flew
    # before session 22 is recorded in the code rather than only in git, and so
    # the test that pins the Scorer ratio has something to compare against.
    "wind.HANNIBAL_R0_SUPERSEDED": Entry(
        "SOURCED",
        "600 ft, Parks et al. 1985 J. Aircraft 22(2) pp.127-128 AS TRANSCRIBED. "
        "SUPERSEDED session 22 by Wingrove & Bach 1994 Fig. 4, which gives the "
        "same vortex a 1000 ft DIAMETER -- 500 ft of radius -- and whose Morton "
        "row reproduces this project's Morton radius exactly. Parks has never "
        "been retrieved, so the conflict is decided on which document is held, "
        "not on which is primary. Kept because Parks' own Scorer check "
        "(spacing/diameter 'about 2.9 to 3.5') reproduces at 600 ft and does "
        "NOT at 500 ft, which is real evidence the other way; see "
        "test_wind.test_the_spacing_to_core_diameter_ratio_and_what_session_22_cost_it.",
    ),

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

    # -----------------------------------------------------------------------
    # Session 23, the CAT source pass. Eleven module-level constants entered
    # `wind.py` and the coverage check failed the build for every one of them,
    # which is exactly what this module's docstring promises. Every entry below
    # is SOURCED: this session added no declared numbers to `wind.py`, and that
    # is the point of the Mehta case in particular.
    # -----------------------------------------------------------------------
    "wind.MEHTA_HANNIBAL_X_FT": Entry(
        "SOURCED",
        "(-12384, -6669, -343, 3761, 12272) ft, the along-flightpath positions "
        f"of the five converged vortices. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_Z_FT": Entry(
        "SOURCED",
        "(-3516, -1836, -94, -254, 1738) ft, aircraft height ABOVE each core. "
        "The sign convention is fixed on Mehta p. 29, which derives z < 0 from "
        f"a negative horizontal perturbation. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_R0": Entry(
        "SOURCED",
        "500.5 ft converged core radius. Independently corroborated by NASA "
        "TM-102186 p. 3-4, 'a diameter of 1,000 ft'. Note Mehta's PRE-FIT "
        "manual estimate was 600 ft, which is the most likely origin of the "
        f"600 ft once attributed to Parks 1985. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_V0": Entry(
        "SOURCED",
        "86.8 ft/s converged tangential velocity. TM-102186 rounds the same "
        "quantity to 87 ft/s. Against Parks' 85 the spread is 2.1%, and the "
        f"field is exactly linear in V0 so that is the spread on any wind. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_PSI_DEG": Entry(
        "SOURCED",
        "31 deg between the wind vector and the flightpath, Mehta p. 29. This "
        "is the only oblique traverse the project holds, and it is what "
        f"licenses `VortexArray.cos_dpsi` existing at all. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_ALTITUDE": Entry(
        "SOURCED",
        "37,000 ft. Mehta p. 29: the flight path was 'nearly straight and level "
        f"at an altitude of 37,000 ft'. {_MEHTA}",
    ),
    "wind.MEHTA_HANNIBAL_CORE_PAIR": Entry(
        "DERIVED",
        "(2, 3), the indices of the two vortices whose cores the aircraft "
        "penetrated. Derived by |z| < r0 over MEHTA_HANNIBAL_Z_FT, which "
        "selects exactly -94 and -254 ft against a 500.5 ft radius. Matches "
        "TM-102186 Fig. 7's 'two significant vortices'.",
        inputs=("wind.MEHTA_HANNIBAL_Z_FT", "wind.MEHTA_HANNIBAL_R0"),
    ),
    "wind.TM102186_HANNIBAL_NZ": Entry(
        "SOURCED",
        "(-1.0, +1.7) g, the normal acceleration the DC-10 recorded in this "
        "encounter. NASA TM-102186 p. 3-4: 'the wide fluctuations in the normal "
        "acceleration from +1.7 to -1.0 g'. A BAND TO REPORT AGAINST, never a "
        "target: PROJECT.md section 5 rules absolute load agreement out on "
        "aircraft type.",
    ),
    "wind.TM102186_HANNIBAL_GUST_PERIOD": Entry(
        "SOURCED",
        "5 s. NASA TM-102186 p. 3-4: 'sharp up-and-down gusts about 5 sec "
        "apart'. Nothing in the model consumes it -- it is a REFERENCE, and "
        "session 23c made it the one this project checks itself against that "
        "the identification did not set. V0 and r0 were fitted to these winds, "
        "so an amplitude comparison partly re-derives the fit; the spacing "
        "comes from the core positions and the aircraft's speed instead. The "
        "flown run gives 5.29-5.43 s depending on which reading of 'apart' is "
        "taken, and the whole residual is accounted for by this 747 flying "
        "M 0.80 where 5.0 s over the 4,104 ft separation needs M 0.85.",
    ),
    "wind.MEHTA_COST_STARTUP": Entry(
        "SOURCED",
        f"482. {_MEHTA} p. 29, the cost of the MANUAL startup estimate for "
        "n = 2 -- 'the cost ... is 482 for the initial estimates'. Held apart "
        "from MEHTA_COST because it is a guess and not a fit. Quoting it "
        "beside a converged cost as one 'history' is the same error this "
        "project caught TM-102186 making with Schultz's Table 1 initial "
        "estimates (PROJECT.md section 5), and PROJECT.md section 3 made it "
        "once before session 23c separated them.",
    ),
    "wind.MEHTA_COST": Entry(
        "SOURCED",
        f"{{2: 355, 3: 303, 4: 226, 5: 214}}. {_MEHTA} pp. 29-30, the CONVERGED "
        "cost at each array size, from his Eq. (A3) with B the identity. Eq. "
        "(A3) carries a 1/N, so these are MEAN squares and convert to an RMS "
        "wind residual without N -- which the paper never states and no source "
        "held here supplies. Units are assumed (ft/s)^2: e is a difference of "
        "winds from Eq. (4), which is homogeneous in V0, and V0 is quoted in "
        "ft/s. Fig. 5 nevertheless plots the horizontal wind in knots, so "
        "mehta_residual_ceiling is written to be immune to that and "
        "mehta_unmodelled_wind is not.",
    ),
    "wind.MEHTA_COST_SATURATES_AT": Entry(
        "SOURCED",
        f"5. {_MEHTA} p. 30: 'Further increases in the number of vortices "
        "(n = 6,7, etc.) do not result in decreases in the cost. In fact, the "
        "algorithm pushes the extra vortices away from the flight path'. This "
        "is what makes MEHTA_COST[5] a FLOOR for a Rankine array against this "
        "record rather than the point one author stopped, and therefore what "
        "lets the residual bound the field FORM.",
    ),
    "wind.LESTER_LEE_WAVE_WAVELENGTH": Entry(
        "SOURCED",
        f"22,000 m. {_LESTER} p. 1106, 'wavelength about 22 km', derived from "
        "DFDR data at 10 km and read there as a mountain lee wave. Held BESIDE "
        "the declared LEE_WAVE_WAVELENGTH rather than replacing it -- different "
        "mountain range, different altitude -- so its job is to bound that "
        "declaration, which it does at 12%.",
    ),
    "wind.LESTER_GREENLAND_NZ": Entry(
        "SOURCED",
        f"(-1.0, +2.7) g. {_LESTER} p. 1105: 'Vertical accelerations reached "
        "+2.7g, -1.0g'. A B-747, which is the type this project models. Same "
        "standing as TM102186_HANNIBAL_NZ: a band to report against.",
    ),
    "wind.LESTER_GREENLAND_ALTITUDE_GAIN": Entry(
        "SOURCED",
        f"300 m. {_LESTER} p. 1105: 'a sudden altitude gain of 1000 feet "
        "(300 m)'.",
    ),
    "wind.LESTER_GREENLAND_ALTITUDE": Entry(
        "SOURCED",
        f"33,000 ft MSL. {_LESTER} p. 1105, Pan American Flight 125 over "
        "southern Greenland at 62 N 48 W, 1654 UTC 22 January 1985.",
    ),
    "wind.DFDR_WIND_RMS_ERROR": Entry(
        "DERIVED",
        "{horizontal 2.449, vertical 2.236} m/s, the RSS of the contributions "
        f"{_LESTER} Table 1 p. 1105 tabulates separately: horizontal dV_xy 1.0, "
        "dV 1.0, V d(psi+beta) 2.0; vertical dh_dot 1.0, V d(Theta-alpha) 2.0. "
        "The relation is sqrt of the sum of squares, which the paper's own "
        "Eqs. (8) and (9) state. THIS IS THE BOUND ON EVERY IDENTIFIED VORTEX "
        "PARAMETER and replaces the reasoned +/-25% docs/ASSUMPTIONS.md E2 "
        "carried: against Mehta's V0 it is 8.45%.",
        inputs=("wind.DFDR_WIND_RMS_ERROR_SPEED",),
    ),
    "wind.DFDR_WIND_RMS_ERROR_SPEED": Entry(
        "SOURCED",
        f"250 m/s. {_LESTER} Table 1 is headed 'Level flight, V = 250 m/s', so "
        "the errors above are that speed's and the implied flow-angle error "
        "0.458 deg only follows at it.",
    ),
    "wind.DRYDEN_LW": Entry(
        "SOURCED",
        "1750 ft = 533.4 m, the MIL-F-8785C vertical scale length above 2000 ft. "
        "It is the half of that model this project CAN transcribe: the spec "
        "states L_w as a constant in that regime, while sigma_w is a chart "
        "against altitude and exceedance probability that PROJECT.md section 3 "
        "records as un-digitised. Nothing here supplies a sigma_w -- "
        "`dryden_vertical_field` takes it as an argument and "
        "scripts/cat_bounds.py sweeps it, so the reported 4-5 m/s is what a "
        "result IMPLIES and never an intensity this project asserts.",
    ),
    "wind.DRYDEN_ALTITUDE_FLOOR": Entry(
        "SOURCED",
        "2000 ft = 609.6 m. Below it MIL-F-8785C's low-altitude model applies "
        "and L_w is a function of height rather than the constant above, so "
        "DRYDEN_LW is wrong there. Carried as a named floor rather than a "
        "comment because every CAT case this project holds is far above it and "
        "a future low-altitude use would otherwise inherit the wrong length "
        "silently -- the same failure mode as the recovery band.",
    ),
}
