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
    "b747.Mwd": Entry("SOURCED", f"-0.000116, dimensional. {_CR2144_IX4}"),
    "b747.Cmadot": Entry(
        "DERIVED",
        "Cmadot = Mwd * 2 * Iy * U0^2 / (qS * c^2), the Cmq relation with one "
        "more U0 because Mwd is per unit wdot and alphadot = wdot/U0. Gives "
        "-6.3360, and round-trips to Table IX-4's -0.000116. Table IX-4's "
        "Zwd is NOT taken: it converts to a negative CL_alphadot, which "
        "downwash lag says is the wrong sign. PROJECT.md section 5.",
        inputs=("b747.Mwd", "b747.c"),
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
    # -- the Hannibal core radius, and why there is no "superseded" entry ---
    # There was one, for `wind.HANNIBAL_R0_SUPERSEDED` = 600 ft, recording the
    # value the project flew before session 22. Session 26 obtained Parks et al.
    # 1985 and put 600 ft BACK into `PARKS_CASES`, so the constant it recorded
    # no longer names a superseded value and has been removed rather than
    # inverted -- an entry that flips its meaning is worse than no entry.
    #
    # Worth recording that this ledger had the argument right before the paper
    # arrived: its old text noted that Parks' Scorer check "reproduces at 600 ft
    # and does NOT at 500 ft, which is real evidence the other way". It did, and
    # it was. The full discrepancy across the four lineages is `ASSUMPTIONS.md`
    # E12; the coherent triples are in `wind.PARKS_CASES` with their citation.

    # -- the Lamb-Oseen matching constants ----------------------------------
    "wind.LAMB_OSEEN_RC_OVER_R0": Entry(
        "SOURCED",
        "0.892135132495 = 1/s where s = 1.1209064228 solves exp(s^2) = 1 + 2 s^2, "
        "the peak of the Lamb-Oseen profile (1/s)(1 - exp(-s^2)). Choosing the "
        "core parameter rc this way puts the profile's peak AT r0, so it agrees "
        "with Parks' Rankine core on the radius Parks actually identified. "
        "SOURCED to the Lamb-Oseen profile itself -- a classical closed-form solution of the Navier-Stokes equations for a decaying line vortex -- rather than to any particular vortex: the number is a property of the PROFILE and is the same for every (r0, V0). Solved, not transcribed; test_wind re-derives it with brentq and fails on a typo.",
    ),
    "wind.LAMB_OSEEN_CIRCULATION": Entry(
        "SOURCED",
        "1.397952547316 = (1/s)/f(s) at the same s, so that the profile's PEAK "
        "tangential velocity is exactly V0. Together with RC_OVER_R0 this makes "
        "Lamb-Oseen and Rankine agree on both numbers Parks identified -- core "
        "radius and peak tangential velocity -- and differ only in shape, which "
        "is what makes substituting one for the other a controlled experiment. Like RC_OVER_R0 it is a property of the profile, identical for every (r0, V0), and re-derived in test_wind rather than trusted.",
    ),

    # -- the Prandtl-Glauert guard rails ------------------------------------
    # DECLARED, not sourced, and deliberately placed where they cannot bind.
    "aero.PG_MACH_MAX": Entry(
        "DECLARED",
        "0.90. Prandtl-Glauert's 1/sqrt(1-M^2) diverges at M 1 and the linearised "
        "subsonic theory it comes from has stopped applying well before that. "
        "0.90 is chosen to sit ABOVE every condition this project flies -- the "
        "fastest is the 747's M 0.80 -- so it is a guard that never binds rather "
        "than a transonic model, which this project does not have. A run that "
        "reaches it is already outside the band checks.recovery_band gates on. "
        "Sensitivity is therefore identically zero at every recovery point, "
        "asserted in test_aero.",
    ),
    "aero.PG_FLOOR": Entry(
        "DERIVED",
        "1 - PG_MACH_MAX^2 = 0.19. The smallest value of 1 - M^2 the "
        "Prandtl-Glauert denominator may see, so jacfwd gets a finite derivative "
        "even where a root-find probes past the cap. Derived from PG_MACH_MAX and "
        "nothing else.",
        inputs=("aero.PG_MACH_MAX",),
    ),

    # -- the ISA's own earth radius -----------------------------------------
    "atmosphere.R_EARTH_ISA": Entry(
        "SOURCED",
        "6,356,766 m, ICAO Doc 7488 / ISO 2533's nominal earth radius -- the "
        "value the standard atmosphere's own geopotential conversion "
        "H = R z / (R + z) is DEFINED with. It is deliberately not a geodetic "
        "radius: 6,378,137 equatorial and 6,356,752 polar both disagree with "
        "the ISA tables this module reproduces, because 6,356,766 is the radius "
        "that makes those tables self-consistent at 45 deg latitude under a "
        "constant g0. Added session 23, when the geometric/geopotential "
        "conversion the module had documented as unnecessary was measured to "
        "cost 0.159% of density at 30,000 ft and 0.368% at 40,000 ft.",
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
    "wind.DRYDEN_LU": Entry(
        "SOURCED",
        "Equal to DRYDEN_LW. MIL-F-8785C makes the turbulence ISOTROPIC above "
        "2000 ft: L_u = L_v = L_w = 1750 ft and sigma_u = sigma_v = sigma_w. "
        "Ledgered despite being an alias because the EQUALITY is the sourced "
        "claim, not the number -- below the floor the three diverge and none of "
        "these constants applies. The claim is also checked without the "
        "document: `test_lateral.py` verifies that the longitudinal and "
        "transverse spectra satisfy the isotropic relation "
        "Phi_t = (Phi_u - Omega dPhi_u/dOmega)/2 to machine precision, which a "
        "mismatched pair of forms would fail.",
        inputs=("wind.DRYDEN_LW",),
    ),
    "wind.DRYDEN_LV": Entry(
        "SOURCED",
        "Equal to DRYDEN_LW, for the same isotropy statement as DRYDEN_LU. The "
        "lateral and vertical components share BOTH the scale length and the "
        "spectral form -- `dryden_spectrum` serves them both -- which is why "
        "there is no separate transverse spectrum function.",
        inputs=("wind.DRYDEN_LW",),
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

    # -- TPAWS Table 1, ingested whole (design phase S0) ---------------------
    # `atisim/data/tpaws_tm2012_217337_table1.csv`, written by
    # `scripts/tpaws_table1_ingest.py` from the PDF's TEXT LAYER: no
    # digitisation, no reading uncertainty, no calibration step. Every column
    # below is therefore SOURCED in the strict sense the docstring requires --
    # read directly from a cited table -- which is not true of this project's
    # other figure-derived data. The aircraft is NASA Langley's B-757 ARIES.
    "tpaws.table1": Entry(
        "SOURCED",
        "53 significant turbulence events. NASA/TM-2012-217337, Hamilton, "
        "Proctor & Ahmad, 'Flight Tests of the Turbulence Prediction and "
        "Warning System (TPAWS)', February 2012, NTRS 20120003172, Table 1 "
        "'Summary of Significant Turbulence Events', printed p. 8. The row "
        "count is the document's own: printed p. 7, 'a summary of the key in "
        "situ parameters from 53 significant turbulence events', and printed "
        "p. 5, 'ARIES tallied 53 encounters with CIT'. Two rows print a RANGED "
        "altitude -- 232-05 '31 to 35' and 235-05* '22 to 19' -- and are "
        "carried as first/last exactly as printed, never collapsed to a "
        "midpoint. A previous parse dropped those two and reported 51.",
    ),
    "tpaws.sigma_dn": Entry(
        "SOURCED",
        "0.20-0.49 g across the 53 rows. THE WINDOW IS NOT THE WHOLE RECORD, "
        "and the distinction decides what the column may be compared against: "
        "printed p. 5 defines sigma_dn as a function of time over a 'shifting "
        "/ sliding window' of tau = 5 s, subtracting the mean over that same "
        "window, and printed p. 125 names the tabulated scalar 'the peak "
        "sigma_dn'. So each value is the MAXIMUM OVER THE ENCOUNTER OF A 5 s "
        "RUNNING STANDARD DEVIATION. A running maximum is >= the whole-record "
        "value, so a peak factor built on this denominator is SMALLER than one "
        "built on a whole-record sigma.",
        inputs=("tpaws.table1",),
    ),
    "tpaws.dn_extremes": Entry(
        "SOURCED",
        "Delta_n_max +0.37 to +1.30 g, Delta_n_min -0.34 to -1.40 g. The "
        "ENCOUNTER's extremes, printed p. 7, 'the peak normal load "
        "acceleration'; Table 1's column group header, 'Peak In Situ "
        "Turbulence (g's)', governs these and sigma_dn alike. Figure 2, "
        "printed p. 6, plots the pair over 606 encounters 'assuming a 5 second "
        "window', from Bowles and Buck (2009).",
        inputs=("tpaws.table1",),
    ),
    "tpaws.weight": Entry(
        "SOURCED",
        "167.7-192.0 klb, tabulated FOR EVERY ROW -- which is rare enough in "
        "this project's sources to be worth naming. IT DOES NOT CLOSE "
        "PROJECT.md 5.19, and must not be read as doing so: 5.19 wants the "
        "weight of a DC-10-10 (N1809U) on the day of the Hannibal encounter, "
        "and this is a B-757 on 53 other encounters. What it supplies is a "
        "population in which the quantity 5.19 lacks is present, not the "
        "missing value.",
        inputs=("tpaws.table1",),
    ),
    "tpaws.tas": Entry(
        "SOURCED", "185-238 m/s true airspeed.", inputs=("tpaws.table1",),
    ),
    "tpaws.vertical_wind": Entry(
        "SOURCED",
        "Peak vertical wind, +2.77 to +23.42 and -2.09 to -21.42 m/s. Printed "
        "p. 128's Figure 133 computes the companion sigma_u/sigma_w peaks from "
        "20 Hz in situ wind data 'assuming a 5 second window'.",
        inputs=("tpaws.table1",),
    ),
    # -- TPAWS Figure 133, read as VECTORS not pixels ------------------------
    # `atisim/data/tpaws_fig133_sigma_uw.csv`, written by
    # `scripts/digitise_tpaws_fig133.py`. Unlike every other digitised file in
    # this project it carries NO pixel error budget: the chart is vector art,
    # each marker is a filled path whose coordinates the PDF states exactly,
    # and the axes are calibrated on the printed tick labels' own text boxes.
    # The calibration residual is 1.3e-3 data units, which is the author's
    # plotting precision rather than a reading error.
    "tpaws.fig133": Entry(
        "SOURCED",
        "78 events. NASA/TM-2012-217337 Figure 133, printed p. 128: "
        "'Corresponding peak values of sigma_u and sigma_w peaks for all 2002 "
        "turbulence events. Computed from in situ 20 Hz wind data assuming a 5 "
        "second window.' THE POPULATION IS NOT TABLE 1'S: this is all 2002 "
        "events, where Table 1 carries the 49 SIGNIFICANT 2002 events "
        "(sigma_dn >= 0.2 g), so Figure 133's set contains Table 1's and adds "
        "~29 weaker ones. The scatter is unlabelled, so no marker can be paired "
        "to a row and any ratio of the two must be bracketed.",
    ),
    "tpaws.sigma_w": Entry(
        "SOURCED",
        "2.14-8.62 m/s, mean 4.233. The PEAK of a 5 s running standard "
        "deviation of vertical wind, matching sigma_dn's own reduction. 79% of "
        "the points lie above the 1:1 line against sigma_u, which is the "
        "paper's own anisotropy finding (printed pp. 127-128) and is the check "
        "that the axes were not read swapped. NOT CO-LOCATED WITH THE LOAD "
        "PEAK: Figure 134 puts the sigma_u and sigma_w peaks up to 2 km apart, "
        "85% within 900 m, so a ratio of separately located maxima estimates a "
        "transfer gain rather than being one.",
        inputs=("tpaws.fig133",),
    ),

    # -- Stewart's B-757 normal-force set (design phase S1) -----------------
    # `atisim/data/stewart_tm2003_212666_table1.csv`, written by
    # `scripts/stewart_table1_ingest.py` from the text layer. This is the
    # TABLE, not the paper's flight illustration: printed p. 7 says that
    # illustration's measurements "were combined asynchronously ... not
    # representative of an operational data stream" and are "only for
    # illustrative purposes", so nothing is taken from its figures.
    "stewart.table1": Entry(
        "SOURCED",
        "72 rows. NASA/TM-2003-212666, Stewart, 'Description of a Normal-Force "
        "In-Situ Turbulence Algorithm for Airplanes', December 2003, NTRS "
        "20040021314, Table 1 'Aerodynamic parameters', printed pp. 12-13. "
        "Three coefficients against dynamic pressure at 0, 10, 20, 30 and 40 "
        "kft, clean configuration at a nominal 180,000 lb (printed p. 4). The "
        "dynamic-pressure grid is printed three times, once per coefficient, "
        "and the ingest asserts all three agree.",
    ),
    "stewart.CNalpha": Entry(
        "SOURCED",
        "4.847-7.437 per radian across 24 columns. THE AIRCRAFT IS A B-757, "
        "NOT ANY AIRCRAFT THIS PROJECT FLIES, so it is a comparison target and "
        "never an input. Its value is that each column carries a Mach number "
        "through the paper's own Eq. (8) (printed p. 8), giving M 0.242-0.851 "
        "-- a FIRST-HAND published lift-curve slope against Mach, which is the "
        "external check ASSUMPTIONS C3's Prandtl-Glauert axis has never had. "
        "TWO CONFOUNDS, stated because the comparison is worthless without "
        "them: at fixed altitude and weight a higher q is also a LOWER TRIM "
        "ANGLE OF ATTACK, so a column mixes Mach with trim-alpha nonlinearity; "
        "and C_N is the total NORMAL force, not lift.",
        inputs=("stewart.table1",),
    ),
    "stewart.CNq": Entry(
        "SOURCED",
        "3.63-8.54 per radian. Same table and same two confounds.",
        inputs=("stewart.table1",),
    ),
    "stewart.CNdelta": Entry(
        "SOURCED",
        "0.380-0.543 per radian, elevator. Same table. Printed p. 9 reports "
        "the elevator and pitch-rate terms were 'relatively insignificant' for "
        "the illustrated run.",
        inputs=("stewart.table1",),
    ),
    "tpaws.reflectivity": Entry(
        "SOURCED",
        "0-40 dBz peak radar reflectivity factor per event, the band printed "
        "p. 7 states for this table. Five rows carry the printed asterisk, "
        "'events where the radar was in long pulse mode, and which are not "
        "included in radar scoring'.",
        inputs=("tpaws.table1",),
    ),
}
