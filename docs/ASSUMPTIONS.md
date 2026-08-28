# Model assumptions — what is assumed, why, and how big it is

Companion to `PROJECT.md`. That document records **what has been measured**; this one
records **what has been assumed**, with a bound on each. Written session 11, after external
review asked that the underlying assumptions be well founded and explained.

**The rule here is the same as §3's:** an assumption without a number is not documented, it
is asserted. Every row below either carries a measured bound or says plainly that it has
none and why.

---

## What this model is scoped to

**Turbulence-encounter analysis of a rigid fixed-wing aircraft over seconds to minutes.**
Concretely: the Wingrove & Bach Fig. 8 discriminator, the Bowles F-factor against thrust
authority, and a Dryden layer to come.

That scope is what makes most of the assumptions below acceptable. Several of them would
be indefensible for a different use — navigation over hundreds of kilometres, a full
flight-envelope simulator, or anything asserting agreement below about 0.5%.

**Three questions decide whether an assumption is safe here:**

1. Does it bias a quantity the project actually reports?
2. Is the bias smaller than the agreement being claimed?
3. Does it get worse over the encounter window, or stay bounded?

An assumption that fails (2) is the dangerous kind, because it is invisible: the model
agrees with a source to 0.4% while carrying a 0.4% systematic error, and the agreement is
partly luck. Session 11 flagged **two** entries as failing (2). Session 12 measured both,
and **A2 turned out to pass** — the 0.383% gravity error reaches the phugoid 1:1 but the
lateral modes only at 0.06–0.08%, so it does not threaten the agreements it appeared to.
**E2, the vortex span ratio, still fails (2)** and is the one to respect.

**Session 23 retired A1 and A2 by modelling them, and found a third entry failing (2)
while doing it.** A3 — a geometric height fed to a geopotential atmosphere — had its bound
quoted on the *altitude* difference rather than on the density difference the aero forces
actually see. Re-measured, that is **0.37% at 747 cruise**, the same size as the gravity
error A2 was worried about and independent of it. It is a one-line fix that would move §4
rows, so it is recorded rather than taken in a documentation pass.

---

## A. Frames and Earth

### A1. RETIRED session 23 — flat, non-rotating Earth; NED is an inertial frame

**This is no longer assumed. It is modelled.** `atisim/earth.py` carries the WGS-84
ellipsoid and `dynamics.earth_acceleration_terms` carries the rotating-Earth equations, in
the formulation JSBSim 1.3.1 uses, established from the running binary rather than
transcribed. The entry is kept below because it records what the assumption cost, and
because §4's older rows were measured under it.

**What is modelled now:**

| | |
|---|---|
| Geometry | WGS-84 ellipsoid. `a` and `f` are the defining pair; `b` and `e²` are derived, never quoted |
| Position | ECEF, as an offset from a run anchor. Geodetic ↔ ECEF by Bowring at three iterations, which reaches the float64 floor |
| Local frame | NED at the **aircraft's own** geodetic latitude, a derived view rather than the propagation frame |
| Gravitation | `GM/r²` with the J2 zonal harmonic, `J2 = 1.08262982e-3` |
| Rotation | `Ω = 7.292115e-5` rad/s — Coriolis `2Ω×v`, centrifugal `Ω×(Ω×r)`, and the body-rate frame transfer, all three |
| Verification | Term-by-term against a frozen JSBSim reference over a latitude × altitude × attitude probe grid; geodesy to 2e-11 rad, gravity to 4.6e-13 relative |

**What is still not modelled** — each has its own entry below, because none of them is
covered by the JSBSim comparison: gravity truncated at J2 (A5), no polar motion, nutation
or time-varying gravity (A6). Position is stored anchor-relative rather than absolute,
which is a deliberate deviation from JSBSim's `FGPropagate` (A7), and trim is now a
function of latitude and heading (A8).

**What the retired assumption cost, measured session 11** — kept as the record. It lived in `dynamics.derivatives`, whose Newton-Euler equations carried no Earth-rate or transport terms, and in `state.py`, which treated NED as inertial; it was justified by the encounter lasting 1.5 s (a vortex core) to 95 s (a microburst to ground) and covering a few kilometres:

| | 747 at 235.9 m/s, 12,192 m | Cherokee at 50 m/s, 1,500 m |
|---|---|---|
| Coriolis `2Ω×V` | 0.03441 m/s² = **0.00351 g** | 0.00729 m/s² = 0.00074 g |
| Transport `V²/(R+h)` | 0.00872 m/s² = 0.00089 g | 0.00039 m/s² = 0.00004 g |
| Position error over a 20 s window | **6.88 m** | 1.46 m |

**Verdict at the time: sound for this use case.** Load-factor excursions the project
reports are of order 1 g (−1.23 g for the vortex, −1.90 g for the manoeuvre), so 0.0035 g
is 0.35% of the signal and does not accumulate within a window. The limit it named — "not
sound for any claim about ground track over hundreds of km, or a run longer than about ten
minutes", 6.88 m per 20 s growing as t² — is the limit this change removes.

### A2. RETIRED session 23 — constant gravity, g = 9.80665 m/s²

**Session 12 decided `g(h)` was not worth modelling, and session 23 modelled it anyway —
not because that decision was wrong, but because it was overtaken.** `earth.gravitation`
is now `GM/r²` with J2, and the centrifugal term is in the equations of motion, so the
gravity the aircraft feels falls out of the Earth model rather than being chosen. The
session-12 measurement below is what says the change was *safe*, and it is kept in full.

**What the change was actually worth, measured session 23.** The same 747 cruise
comparison, re-run as `earth.FLAT` (constant `G0` along the local vertical) against the
shipped `earth.WGS84_J2` at 47N, using `validation.longitudinal_modes` and
`lateral_modes` — the same functions §4's rows are asserted through:

| Mode at 747 cruise | FLAT, g = 9.80665 | WGS84_J2 at 47N | Movement | Session 12 predicted |
|---|---|---|---|---|
| phugoid ωn | 0.055319 | 0.055117 | **−0.3664%** | −0.3798% |
| phugoid ζ | 0.055818 | 0.055525 | −0.5265% | −0.5385% |
| short period ωn | 0.950773 | 0.950776 | **+0.0002%** | +0.0002% |
| short period ζ | 0.342523 | 0.342508 | −0.0044% | −0.0046% |
| Dutch roll ωn | 0.943052 | 0.942335 | −0.0761% | −0.0788% |
| Dutch roll ζ | 0.035998 | 0.035849 | −0.4146% | −0.4288% |
| roll τ | 1.794822 | 1.793781 | −0.0580% | −0.0602% |
| spiral τ | 137.9664 | 138.0403 | +0.0536% | +0.0552% |
| *trim α* | *4.629171°* | *4.599953°* | *−0.6312%* | *−0.6535%* |

**Session 12's prediction has the right sign in every row and is slightly high in magnitude
wherever the movement is large enough to tell — short-period ωn at +0.0002% has no digits
left to differ in — and there is one reason for it.** It used `g(h) = g₀(R/(R+h))² = 9.76922`, a spherical inverse-square value. The
apparent gravity the rotating WGS-84 model actually produces at 47N and 12,192 m is
**9.770541** — J2 gravitation 9.786389 less the 0.015848 m/s² centrifugal term along the
local vertical — which is −0.3682% against `G0` rather than −0.3816%. That is a ratio of 0.965, and
dividing each measured row by its predicted one gives **0.963 to 0.978** — the same
number to within the residual differences of baseline, since the FLAT column is itself
the six-unknown trim on an ECEF state and not the plant session 12 measured.

**Lanchester's 1:1 is now measured against a gravity the model computes rather than one
imposed for the experiment: phugoid ωn −0.3664% against an apparent-gravity change of
−0.3682%.** That is the strongest form the claim has had.

**The FLAT column reproduces session 12's own `g = 9.80665` column exactly on both
frequencies** — 0.055319 and 0.950773, to every digit it published — which is what says
the comparison is like-for-like. It did NOT until session 23 fixed
`validation.longitudinal_matrix`, which was taking its Jacobian at `q = 0` while the
trim it was handed is an equilibrium only at the transport rate; that gave 0.055292
here. §4 carries the repair.

**Apparent gravity now varies with latitude, which is what a constant g could never carry:**
9.780282 m/s² at the equator, 9.808052 at 47N, 9.832067 at the pole — a **0.53% spread**,
an order above the worst movement in the table. Session 12's closing caveat ("revisit if
the project ever compares one aircraft across two altitudes") extends to latitude, and is
the case this change serves.

**Where:** `atmosphere.G0`, used by `dynamics.derivatives`, `trim`, `specific_force`.

**Why:** simplicity, and it is exact at sea level.

**Bound, measured** — using `g(h) = g₀·(R/(R+h))²`, R = 6,371,008.8 m:

| Aircraft | Altitude | True g | Model is |
|---|---|---|---|
| 747 cruise | 12,192 m | 9.76922 | **+0.383% high** |
| 747 approach | 0 m | 9.80665 | exactly right |
| Cherokee | 1,500 m | 9.80204 | +0.047% high |

Lanchester gives `ωn_phugoid = √2·g/u₀`, so a gravity error should map **1:1** into phugoid
frequency. Session 11 reasoned from that to a warning covering every mode. **Session 12
measured it instead, and the warning was too strong.**

**Bound, measured session 12** — `atisim.dynamics.G0` replaced by `g(h)`, re-trimmed,
all five modes recomputed. §4's tolerance is the one each mode is actually asserted to:

| Mode at 747 cruise | g = 9.80665 | g(h) = 9.76922 | Movement | §4 tolerance | Consumed |
|---|---|---|---|---|---|
| phugoid ωn | 0.055319 | 0.055109 | **−0.3798%** | 5% | 7.6% |
| phugoid ζ | 0.055956 | 0.055654 | −0.5385% | 10% | 5.4% |
| short period ωn | 0.950773 | 0.950775 | **+0.0002%** | 3% | 0.01% |
| short period ζ | 0.342526 | 0.342510 | −0.0046% | 5% | 0.1% |
| Dutch roll ωn | 0.943202 | 0.942458 | −0.0788% | 2% | 3.9% |
| Dutch roll ζ | 0.036085 | 0.035931 | −0.4288% | 10% | 4.3% |
| roll τ | 1.795366 | 1.794285 | −0.0602% | 5% | 1.2% |
| spiral τ | 138.0424 | 138.1187 | +0.0552% | 2% | 2.8% |
| *trim α* | *4.6362°* | *4.6059°* | *−0.6535%* | — | — |

**Control: the same sweep on the sea-level approach 747 moves every quantity by exactly
0.0000%**, since `g(0) = g₀` identically. The effect is altitude and nothing else.

**Three things the measurement says that the reasoning did not:**

1. **Lanchester holds, to three figures.** Phugoid ωn moves −0.3798% against a gravity
   change of −0.3816%. The 1:1 claim is now measured rather than asserted.
2. **The 1:1 mapping is the phugoid's alone.** Short period is *immune* (+0.0002%), and the
   lateral modes move 0.055–0.079% — **five to fifteen times smaller** than the agreements
   session 11 feared for them (Dutch roll ωn 0.4%, spiral τ 0.8%, roll τ 0.9%). So the
   blanket "no sub-0.5% claim at altitude is safe" was wrong: it applied the phugoid's
   sensitivity to modes that do not have it.
3. **The lateral modes' sensitivity is indirect.** There is no gravity term dominating the
   lateral equations. g moves the **trim point** — α falls 0.65%, because less weight needs
   less lift — and the derivatives are then read at a different α. That is why the lateral
   movement is an order of magnitude below the phugoid's.

**Decision, session 12 — SUPERSEDED session 23, and the reasoning is kept because it was
sound at the time.** Two of its three grounds have since gone: `G0` did not have to move
after all (`earth.gravitation` computes gravity, and `atmosphere.G0` stays what it is, so
no §4 baseline was moved to get here), and the project now does compare one aircraft
across latitudes, which is the case it named as the one a constant g cannot serve.

**Decision, session 12: `g(h)` is NOT modelled, and this bound closes the entry.** Every
movement is comfortably inside the tolerance of the check it would affect — the worst
consumes 7.6% of its band. Against that, `G0` is imported by `dynamics`, `trim`, `aircraft`
and `vortex_viz`, and changing it would move §4 baselines that are off-limits to feature
work. And every result the project quotes is at **one altitude per aircraft**, so a constant
g is *exactly* right per run; the bias only exists for comparisons across altitudes, which
the project does not make.

**What replaces the old warning:** a sub-0.5% claim at altitude is unsafe **for the
phugoid**, which carries the full 0.38%. It is safe for the lateral modes (0.06–0.08%) and
for the short period (~0). Revisit if the project ever compares one aircraft across two
altitudes, which is the case a constant g genuinely cannot serve.

### A3. A GEOMETRIC altitude is fed to a GEOPOTENTIAL atmosphere

**Rewritten session 23, and the mismatch is now nameable in a way it was not.** Before the
Earth model there was one height and its kind was ambiguous. Now there are two and the
code picks: `dynamics.derivatives` takes `h` from `earth.ecef_to_geodetic` — a **geodetic,
that is geometric, height above the ellipsoid** — and hands it straight to
`atmosphere.density`, whose docstring says it treats its argument as **geopotential**.

**Where:** `dynamics.py:155` produces it, `dynamics.py:161-162` consumes it. `sensors.py`
and `state.altitude` return the same geodetic height.

**Bound, measured session 23 — and the old bound was on the wrong quantity.** A3 used to
quote "0.17% at 11 km, 0.31% at 20 km" from `atmosphere.py`. Those are the *altitude*
differences `h − H`, where `H = Rh/(R+h)`. What the aero forces see is the **density**
error, which is two to three times larger because density falls exponentially:

| h (m) | H geopotential (m) | h − H (m) | ρ(h) | ρ(H) | error in ρ |
|---|---|---|---|---|---|
| 0 | 0.0 | 0.00 | 1.225000 | 1.225000 | 0.0000% |
| 300 (747 approach) | 300.0 | 0.01 | 1.190106 | 1.190107 | −0.0001% |
| 1524 (737 approach) | 1523.6 | 0.36 | 1.055546 | 1.055585 | −0.0036% |
| 2500 | 2499.0 | 0.98 | 0.956859 | 0.956954 | −0.0100% |
| 11278 (737 cruise) | 11258.1 | 19.93 | 0.348309 | 0.349405 | **−0.3138%** |
| 12192 (747 cruise) | 12168.7 | 23.29 | 0.301558 | 0.302668 | **−0.3665%** |
| 20000 | 19937.4 | 62.59 | 0.088035 | 0.088908 | −0.9821% |

**Verdict: sound at approach, and it is NOT the smaller sibling of A2 at cruise — it is the
same size.** 0.37% of density at 747 cruise sits directly on lift, drag and thrust, against
the 0.37% of gravity that A2 measured and that reaches the phugoid 1:1. The two are
independent and the same order, so a sub-0.5% claim at cruise altitude carries both.

**It is not fixed here, deliberately.** The fix is one line — convert `h` to `H` before
calling the atmosphere — and it would move every §4 row taken at cruise, which is exactly
the kind of change §4's standing rule keeps out of a documentation pass. It is recorded as
a known, bounded, one-line defect with its cost measured, which is what this file is for.
Below 3 km it is under 0.01% and nothing needs to wait for it.

### A4. There is no ground

**Where:** `integrate.step` and `rollout`. Nothing in the engine stops a trajectory at
h = 0, and `atmosphere.py` keeps extrapolating below it — see D2.

**Bound, measured — and it is not a bound so much as an absence.** A 747-approach
released at 300 m in a 20° nose-down attitude crosses h = 0 at **t = 8.65 s** at a
**25.1 m/s** sink rate and integrates on to **−698 m**, every sample finite throughout.
Pinned by `test_the_integrator_has_no_ground_plane`.

**What keeps it from being a live defect is a guard at one call site, not in the engine.**
`scripts/microburst.py` — the one shipped analysis that flies at terrain — cuts its run at
one wingspan of clearance, with a stated rationale: there is no terrain, no landing gear
and no ground effect in this model, so an aeroplane within its own span of the surface is
not flying any more and the integration past that point is arithmetic rather than physics.
**That guard protects exactly one caller.** Any new script that flies low inherits nothing.

**Verdict: undeclared until the remediation pass, now declared.** The right response is a
guard where the reader will look for one, not a ground plane in the engine — JAX cannot
raise inside `jit`, so a clamp inside `step` would silently bounce a trajectory instead of
stopping it, which is worse than integrating through. Anything flying below ~1 km AGL
should truncate at its own clearance and say so.

### A5. Gravity is truncated at J2, because JSBSim's is

**Where:** `earth.gravitation`, the `"j2"` branch. `GM/r²` with the second zonal harmonic
and nothing beyond it. J3 onwards, the tesseral terms and the geoid are all absent.

**Why:** the whole Earth model exists to reproduce JSBSim's, term by term, and JSBSim
1.3.1's `FGInertial` stops at J2. Adding a term JSBSim does not have would break the
comparison this work is verified by, and would be a change nothing in this repository
could check.

**Bound, measured session 23 — and it needs its own entry precisely because no JSBSim
comparison can see it.** Both engines are truncated the same way, so the cross-code check
is blind to it. The reference has to be WGS-84's own closed-form normal gravity, whose
defining values `γ_e = 9.7803267715` and `γ_p = 9.8321863685` m/s² are recorded in
`docs/superpowers/plans/2026-08-26-wgs84-earth-rotation.md`:

| | model apparent g | WGS-84 normal gravity | error | relative |
|---|---|---|---|---|
| equator, h = 0 | 9.780282 | 9.7803267715 | −4.48e-05 | **4.6e-06** |
| pole, h = 0 | 9.832067 | 9.8321863685 | −1.19e-04 | **1.2e-05** |

**Verdict: sound, with a factor of 288 of margin over the term this change was about.**
1.2e-04 m/s² at worst, against a Coriolis acceleration of 0.03441 m/s² at 747 cruise —
a factor of 288. In relative terms it is 1.2e-05 against A3's 3.7e-03 density error at 747 cruise, so it is 300x below the other systematic error in this section.

**Not sound for:** anything that needs absolute gravity to better than about 1e-5 relative
— gravimetry, precise geoid work, or a comparison against a source that used full normal
gravity. It is worth knowing that the J2-truncated value is **not** the WGS-84 normal
gravity constant, because it looks as though it should be: an early version of the tier-0
invariant test asserted `9.7803267715` at `rel=1e-6` and would have failed at 4.6e-6.

### A6. No polar motion, no nutation, no time-varying gravity

**Where:** by omission. `earth.OMEGA_WGS84` is a scalar constant, the ECEF z-axis is fixed
through the pole, and `gravitation` is a function of position alone.

**UNMEASURED, AND STATED AS SUCH.** None of the three has been given a number here, and
this entry exists so that absence is on the record rather than invisible. What can be said
without measuring: all three are below A5's J2 truncation over a flight of minutes — polar
motion moves the pole by metres over months, nutation is an arcsecond-scale effect on a
timescale of days to years, and solid-Earth and ocean tides perturb g in the 1e-7 range —
and **JSBSim does not model any of them either**, so they are invisible to the comparison
in exactly the way A5 is.

**Verdict: acceptable for this project's scope**, which §"What this model is scoped to"
puts at seconds to minutes. It fails the standard this file sets — an assumption without a
number is asserted, not documented — and it fails it deliberately, because bounding any of
the three properly needs a source this repository does not have and inventing one is worse
than the gap.

### A7. Position is stored anchor-relative, not as an absolute ECEF coordinate

**Where:** `State.pos_ecef`, and `state.py`'s header calls it out as the one deliberate
deviation from JSBSim's `FGPropagate`.

**Why: numerical, not physical.** The equations are identical either way — every term in
`earth_acceleration_terms` takes the absolute position, and `state.absolute_ecef`
reconstitutes it — but what gets *differenced* by a convergence study or a trajectory
comparison is the stored quantity, and float64 resolves a small number far better than a
large one.

**Bound, measured:** float64's ulp doubles at each power of two, so the ratio depends on
the offset a run reaches and is exactly a power of two:

| stored quantity | magnitude | ulp |
|---|---|---|
| absolute ECEF coordinate | 6,390,329 m | 9.313e-10 m |
| offset, F4's original 12 km case | 12,184 m | 1.819e-12 m — **512× finer** |
| offset, the 20 s cruise run of F4 below | 4,718 m | 9.095e-13 m — **1024× finer** |

**Verdict: sound, and it is the reason F4 has a floor worth having.** The cost is that
`pos_ecef` is meaningless without its `Anchor`, which is why `earth.Anchor` has no default
and every caller states where it is flying.

### A8. Trim is a function of latitude and heading

**Where:** `trim.trim`, six unknowns `[alpha, elevator, throttle, phi, aileron, rudder]`
against six residuals, with `beta = 0` closing the system. It was three unknowns against
three residuals before session 23.

**This is not an assumption so much as a change in what "trimmed" means, and it is here
because it invalidates the reflex that a trim is a property of the aircraft and the flight
condition.** On a rotating Earth the lateral rows of the residual are not identically
zero, so steady flight requires a small bank and a little aileron and rudder to hold it.

**Measured, 747 at cruise on `WGS84_J2`:**

| latitude | heading | trim α | bank φ | aileron (rad) | rudder (rad) | throttle |
|---|---|---|---|---|---|---|
| 0° | 000 | 4.577395° | −0.000000° | −7.0e-09 | 7.4e-09 | 0.730260 |
| 0° | 090 | 4.549582° | 0.000000° | −3.9e-20 | −3.7e-22 | 0.726795 |
| 47°N | 000 | 4.599953° | **−0.148170°** | 2.2e-07 | 3.2e-08 | 0.733221 |
| 47°N | 090 | 4.581012° | **−0.204368°** | 1.06e-04 | 1.11e-05 | 0.730712 |
| 47°S | 000 | 4.599977° | **+0.148170°** | −2.3e-07 | −2.2e-08 | 0.732949 |
| 47°S | 090 | 4.581012° | **+0.204368°** | −1.06e-04 | −1.11e-05 | 0.730712 |
| 70°N | 000 | 4.614575° | −0.190031° | 2.9e-07 | 3.7e-08 | 0.735061 |
| 70°N | 090 | 4.605152° | −0.331661° | 2.7e-04 | 2.8e-05 | 0.733736 |

**The bank is antisymmetric in latitude to every digit shown, and vanishes on the equator**, which is
what says it is the Earth's rotation and not a solver artefact — a sign error would be
symmetric. Heading matters as much as latitude at 70°N, where 000 → 090 nearly doubles it.

**Verdict: sound, and the trap it replaces is the reason `trimmed_controls` changed
signature.** Callers used to build controls from `(elevator, throttle)` and silently drop
the aileron and rudder the trim had solved for; taking the whole solution makes the old
call a `TypeError`. Measured cost of dropping them, 747 at 47N flying the trim it was just
handed: `|vdot|` 1.887e-05 at heading 090 against 1.092e-12 with all four kept.

**Not sound for:** reading a trim as a property of (aircraft, airspeed, altitude) alone.
Every stored or quoted trim now carries a latitude and a heading, and comparing two of them
across sites compares two different equilibria.

---

## B. Mass and structure

### B1. The airframe is rigid

**Where:** everywhere. `Aircraft.inertia` is a constant tensor; there are no structural
degrees of freedom.

**The problem:** CR-2144 §IX's derivative plots are labelled **"Flexible"**. The
derivatives carry aeroelastic corrections for a structure this model does not have. The
data and the model disagree about what kind of aircraft this is.

**Bound: NONE, and it cannot be produced from sources the project holds.** Quantifying it
needs a rigid-airframe derivative set for the same aircraft and condition, which CR-2144
does not tabulate.

**Verdict: the most significant open assumption in the project.** It is not a reason to
distrust the mode comparisons — those are closed-loop against the source's own arithmetic
from the source's own derivatives, so they test the solver regardless (see `PROJECT.md`
§3, "Source qualification"). It *is* a reason not to claim the model represents a real
747's structural response.

### B2. Constant mass and inertia; no fuel burn

**Where:** `Aircraft` is a constant pytree through every rollout.

**Bound, measured:** 747 trim thrust is 186.0 kN; at a typical high-bypass cruise TSFC of
0.55 lb/(lbf·h) that is 2.898 kg/s.

| Run length | Fuel burned | Fraction of mass |
|---|---|---|
| 20 s (an updraft column) | 58 kg | **0.020%** |
| 60 s (a fixed-control hold) | 174 kg | 0.060% |
| 600 s (ten minutes) | 1,739 kg | 0.602% |

**Verdict: sound**, comfortably. Even the ten-minute figure is below the ±25% band §5
places on the vortex parameters. Revisit only if a run ever exceeds an hour.

### B3. The aircraft is symmetric: Ixy = Iyz = 0

**Where:** `aircraft.inertia_tensor` accepts only `Ixz`.

**Verdict: sound.** Every source the project uses tabulates only Ixz, because a
port/starboard-symmetric airframe has no other product of inertia. Ixz itself is **not**
neglected and `PROJECT.md` §2 notes it "is not negligible for the 747" — session 11
measured its consequence: it is the +0.302 intercept in the roll-mode law (§4).

### B4. The accelerometer is at the CG, and the CG is at the inertia reference

**Where:** `dynamics.specific_force`.

**Why:** it is the quantity `load_factor` needs, and Fig. 8 is built from DFDR normal
acceleration.

**Bound: none, and this is a caveat on the comparison rather than on the model.** A real
DFDR accelerometer sits metres from the CG and therefore reads an additional
`ω̇ × r + ω × (ω × r)`. In a pitch manoeuvre with the sensor well forward that term is not
negligible, and **the Fig. 8 reference data carries it while this model does not.** The
project's Fig. 8 claims are orderings rather than values (§5), which is what keeps this
safe — but it is another reason those claims must stay orderings.

### B5. The 747's slug mass is built through a truncated g₀

**Where:** `aircraft._B747_G = 32.174`, used once, to turn Table IX-3's tabulated weight
into slugs for the non-dimensionalisation.

**Why it is an assumption at all:** `units.py` states the rule — *"Never inline a
conversion factor anywhere else"* — and this breaks it. The value truncates
g₀ = **32.17404855643044** ft/s², which is `LB2KG/SLUG2KG` and `G0 · M2FT` alike, both
exact from the 1959 international agreement.

**Bound, measured: 1.509e-6 relative**, and it reaches only this aircraft. The same
function ships its kilogram mass as `W · LB2KG`, which is exact, so the two masses inside
one constructor disagree by that amount. Making the change — `m = W · LB2KG / SLUG2KG`,
pure `units.py` constants — moves **19 quantities on the 747 and nothing on any other
aircraft**, all at ≤ 4.1e-6 relative: `CLa`, `CLq`, `CLde`, `CYb`, `CYdr` at 1.5e-6,
`CL0` 2.4e-6, `e` 2.5e-6, `CD0` 4.1e-6, and the modes and trim below 1.2e-6. **Every
number `PROJECT.md` §4 quotes is unchanged at its quoted precision**, as are the ledger's
5.9450, −23.9232 and 4.0241.

**Verdict: the change was made, measured, and deliberately reverted.** It breaks two
pre-existing tests — `test_extracting_rk4_step_did_not_move_a_single_bit` and
`test_logging_the_run_did_not_move_the_headline_numbers` — which assert **bit equality**
against values captured before earlier refactors. Their whole value is that they admit no
tolerance; re-pinning them would spend that guarantee to buy a cosmetic rule fix that
moves no result. **The flaw is smaller than the fix.** Recorded here rather than closed,
and the reasoning is repeated at the constant itself.

---

## C. Aerodynamics

### C1. Lift is linear in α unless the aircraft carries a table

**Where:** `aero.coefficients`. `CL = CL0 + CLa·α` when `CL_table_alpha` is empty, which is
every entry except the two 737s.

**Changed in session 20.** `aero.coefficients` now interpolates a CL(α) table when one is
present, and the 737 entries carry 737.xml's own — four points, peaking at 1.20 near 13.18°
and falling. Two of the three consequences `PROJECT.md` §5 and §7 record are therefore no
longer unconditional:

- **The ±g asymmetry is reachable for an aircraft with a table.** It was a property of the
  linear form, not of the airframe: the 737's table has slope 4.400 below zero incidence
  against 4.3478 above, so an up-gust and an equal down-gust no longer give equal and
  opposite increments. It remains exactly odd-symmetric for every entry without a table.
- **The Cessna's stall tables are still unused**, but now for want of a caller rather than
  for want of a mechanism.
- **|α| past ~10–12° still reports lift the sources deny** — for the linear entries. The 737
  matches its source to better than 1e-9 across the full ±26° table.

### C2. Aerodynamics are quasi-steady: no α̇ or unsteady lag

**Where:** `aero.coefficients` takes instantaneous `vel_rel` and `omega_rel` only.

**Bound, measured session 11 — and this one is now exact rather than attributed.**
Restoring Caughey's own tabulated CLα̇ = 6.7 recovers his published A[1,1] to **0.03%** and
A[1,2] to 0.003%; the model's A[2,2] **is** his raw Eq. (5.51) Mq. So the entire effect of
the omission is a known multiplicative factor `1/(1 − Zẇ)` on the Z row plus one term in
the M row. `PROJECT.md` §4 carries the table.

**Verdict: sound and bounded.** This is the best-characterised assumption in the project.

### C3. Stability derivatives are constant across the whole flight envelope

**Where:** `Aircraft` is a constant pytree. `aero.coefficients` uses the same numbers at
any speed, altitude and angle of attack the sim reaches.

**Why:** CR-2144 tabulates derivatives at discrete flight conditions. The project
transcribed one per aircraft, two for the 747.

**This is the largest unbounded assumption in the model**, and it is the one that most
deserves attention next. The derivatives are a small-perturbation linearisation about a
single point, and the sim is nonlinear and flies away from that point:

| Run | Excursion from the linearisation point |
|---|---|
| Trim (747 cruise) | M 0.7995 — the reference |
| Lee wave, south leg | minimum airspeed 226.9 m/s → **M 0.7690, ΔM = −0.031** |
| Manoeuvring Fig. 8 point | peak \|α\| **10.31°** against a 5.7° linearisation |

CR-2144's own §IX plots show CLα, CDα, Cmα, Cmq, CLδe and Cmδe all varying with Mach
through the transonic region, so ΔM = 0.03 near M 0.80 is not a small excursion — it is
where those curves bend.

**Bound: none.** Producing one means digitising the derivative-vs-Mach plots, which §7
already declines for the same reason it declines further flight conditions: they are chart
reads off a poor scan, and a chart read is weaker evidence than the tabulated set already
in use. **The honest statement is that every result away from the trim point inherits an
unquantified derivative error, and that this is why §5 caps analysis windows at the linear
range.**

### C4. Parabolic drag polar plus a Korn wave-drag rise

Documented in `PROJECT.md` §5 with measured residuals: within 0.004 near the fit, up to
0.014 below M 0.75 and 0.006 above M 0.88. Not repeated here.

### C5. Thrust acts along body x, through the CG, with no engine dynamics

**Where:** `aero.thrust_force` — `throttle × max_thrust × (ρ/ρ₀)^lapse`.

**Three separate assumptions bundled together, and they are not equally safe:**

- **No spool dynamics.** A high-bypass turbofan takes several seconds from idle to full
  thrust. Every result in the project is either fixed-throttle or trimmed, so nothing
  currently depends on a throttle transient — but the F-factor work compares against
  *thrust authority*, which is a statement about what the engines could do. If a recovery
  manoeuvre is ever flown, spool time is the first thing that must be added.
- **No thrust moment.** The 747's engines hang below the CG, so real thrust produces a
  nose-up pitching moment that changes with throttle. Unmodelled, and unquantified —
  CR-2144 does not tabulate a thrust-line offset.
- **No Mach dependence of thrust.** Only a density lapse.

**Verdict: sound for the fixed-throttle encounters flown so far; the first thing to fix
before any powered-recovery result.**

### C6. Control surfaces move instantly, with no rate limit or actuator lag

**Where:** `Controls` is applied directly; `manual.py` ramps the *stick*, not the surface.

**Bound, measured:** a 747 elevator moves at roughly 37°/s, so the manoeuvring case's
8.926° deflection would take **0.241 s** in reality — **3.65% of the 6.609 s short
period** — and the model applies it in one 0.02 s sample.

**Verdict: sound for mode and frequency work; a real caveat on the manoeuvring Fig. 8
point.** §8 already records that Δθ moves from 25° to 30° across defensible pulse lengths,
so a 0.24 s ramp sits inside that existing sensitivity rather than beside it.

### C7. Drag responds to pitch rate and elevator, giving an `X_q` the source does not model

**Where:** `aero.coefficients` builds `CD` from the **total** `CL`, which carries
`CLq·q̂` and `CLδe·δe` as well as `CL0 + CLa·α`. So a pitch rate changes induced drag.

**This is the engine being *more* complete than its reference, not less.** CR-2144
tabulates no `C_Dq` at all, and its linear model has no row for this term.

**Bound, measured:** the engine's stability-axis `A[0,2] = X_q = −0.8807 s⁻¹` at the
approach condition, against the parabolic polar's own prediction of **−0.8887**, agreeing
to **0.9%** — which identifies the mechanism rather than merely noting the element is
non-zero. Pinned by `test_the_engine_carries_an_Xq_that_cr2144_does_not_model`.

**Verdict: correct physics, and it matters for interpretation rather than accuracy.** It
is one of the two terms that decide phugoid damping, and it is why a "clean-room" rebuild
from CR-2144's tabulated derivatives is not an independent instrument for that mode — it
is a *different* model, missing terms this one has. An audit pass that did not know this
concluded from such a rebuild that CR-2144 contradicted itself by a factor of 4.4 in
`Xu`. It does not. See `AUDIT.md` §2.3.

### C8. Wave drag acts on the total `CL`, including the rate and control contributions

**Where:** `aero.wave_drag` and the Korn `CL/(10cos³Λ)` term in `drag_divergence_mach`
both receive the same total `CL` as C7 describes. Related to C7 but a separate channel —
compressibility rather than induced drag.

**Bound, measured** on the 747 at its cruise condition (M = 0.80, α = 4.64°), as the
change in total `CD` against a `CD` whose wave term saw only the α part of `CL`:

| condition | `CL` | wave drag | vs α-only | ΔCD/CD |
|---|---|---|---|---|
| at trim (δe = −0.025°) | 0.657 | 0.001053 | 0.001055 | **−0.004%** |
| q = 10 °/s | 0.675 | 0.001246 | 0.001055 | **+0.43%** |
| half elevator (12.5°) | 0.736 | 0.002090 | 0.001055 | +2.03% |
| full elevator (25°) | 0.816 | 0.003748 | 0.001055 | +4.48% |

**Verdict: defensible, and now declared with its size.** At trim it is nothing; the fourth
power in Lock's law means it only bites when a large control input is held at cruise Mach,
which is not a condition this project reports. Quote it beside any result that pulls hard
and fast at high Mach.

### C9. Every lift increment acts at the CG's relative wind

**Where:** `aero.aero_forces_moments` builds ONE lift vector, perpendicular to the
relative wind **at the CG**, and one moment. A surface at arm `l` actually meets a local
wind tilted by `ε = q·l/V`, and the streamwise component of its tilted lift is what makes
that channel dissipative. The model omits that component, so a control moment does work on
the airframe with nothing opposing it.

**Bound, measured — and this is a genuine energy-conservation violation, not a rounding
artefact.** The Cherokee delivers `P_aero = +56,927.7 W` in motionless air with the
throttle shut, at |q| = 8.256 rad/s and full elevator. The omitted term cancels the
elevator's moment power **exactly** — algebraically, to a relative residual of **1.9e-16**
— when its lift increment is placed at `l_δe = −(Cmδe/CLδe)·c`, and restoring it takes the
violating region from **1190 grid points to 0**.

**Why it is nevertheless bounded, and wholly outside the declared envelope:**

| Check | Result |
|---|---|
| `P_aero > 0` among 80,000 randomised states inside the declared abs(α) ≤ 12° envelope | **0** |
| threshold pitch rate at which any aircraft enters the region | **84–201 °/s** |
| `max(E − E₀)` over every still-air run tried, including feedback laws designed to pump it | **+0 exactly** |
| a fixed-control pull from trim | never reaches the region |

**Verdict: documented, and deliberately NOT repaired.** Three reasons, each sufficient.
*(1)* The tilt applies to **every** lift contribution at its own arm — `CL0`, `CLa`, `CLq`
as well as `CLδe` — and the model has arms for none of the others, so correcting one
channel would make the force build-up less coherent than leaving all four alone.
*(2)* The arm is only valid under an attribution this project has already rejected:
`l_δe = −Cmδe/CLδe·c` holds if *all* of both derivatives come from the tail, and
`airframe.effective_tail_arm`'s docstring records that separating the wing's share was
tried, gave the wing 83% of `CLq`, implied a 23-chord arm, and was rejected with an
explicit do-not-re-attempt note. No held source settles it. *(3)* It would move trim, all
five modes and every headline number.

**What would settle it:** a source giving the wing/tail split of `CLδe`, or replacing the
derivative build-up with a strip or panel force model that places every load where it
acts. Pinned by five tests around
`test_the_elevator_moment_power_is_cancelled_by_its_own_lift_tilt`.

### C10. `δa` is a compound control treated as a single angle

**Where:** `Controls.aileron`, and `aileron_limit = 20°` on every aircraft.

CR-2144's own footnote to Table IX-1 defines `δa` as *"total deflection of right inboard
aileron plus left inboard aileron with the effect of outboard ailerons included"*. The
engine treats it as one deflection angle with one limit, which is what the source's
derivatives are referenced to and therefore correct — but it means `aileron_limit` is a
declared limit on a **compound** quantity and is not the travel of any single surface.

**Verdict: document, do not reinterpret.** Rescaling the limit to a per-surface figure
would break the correspondence with the derivative it multiplies. Anyone comparing this
number against a 747 flight manual is comparing two different quantities.

### C11. The two routes to the tail arm disagree by 2–2.9× on the light aircraft

**Where:** `airframe.effective_tail_arm` takes `l/c = −Cmq/CLq`. The control pair gives the
same geometry independently as `l/c = −Cmδe/CLδe`. Nothing in the engine compares them.

**Bound, measured:**

| aircraft | rate pair `−Cmq/CLq` | control pair `−Cmδe/CLδe` | ratio |
|---|---|---|---|
| 747 cruise | 4.0241 | 3.9694 | 1.4% |
| 747 approach | 3.8519 | 3.9645 | 2.9% |
| **Cherokee** | **1.2802** | **2.5621** | **2.00×** |
| **Cessna 172** | **0.8558** | **2.4681** | **2.88×** |

**The split is exactly the sourcing split**, and that is the finding. The two CR-2144
aircraft agree to a few per cent, which is an independent corroboration of that
transcription the project did not previously claim. The two that disagree are precisely
the two whose cited source file is not in this repository — see `aircraft.py`'s status
notes on the Cherokee and the Cessna.

**Verdict: unresolvable here, and it must stay that way.** **Which** of the two estimates
is wrong, or whether both are, cannot be determined without the source. Adjusting either
derivative to make them agree would be fabrication.

*Identified improvement, not taken in this pass:* `tail_arm_is_plausible` reads only the
rate estimate and checks it against a band. It rejects both light aircraft, but by luck
rather than by design — requiring the two routes to **agree** as well as to fall in band
would reject them for the actual reason. That changes which aircraft may enter the strip
path, so it is a scoped modelling change with its own re-measurement, not a repair.

---

## D. Atmosphere

### D1. ISA exactly, with no weather deviation; two layers to 20 km; dry air

**Where:** `atmosphere.py`.

**Verdict: sound, and deliberately so.** The wind fields *are* the weather in this project,
and each is taken from a source that specifies its own conditions. Adding a temperature
offset would mean inventing one, which §3's rule forbids. Humidity changes density by well
under a percent at these altitudes.

### D2. The atmosphere has no ceiling and no floor

**Where:** `atmosphere.temperature`, `pressure`, `density`. The module docstring says
"0 to 20 km"; nothing enforces either end.

**Bound, measured:** above 11 km it holds T = 216.65 K **forever** — which is right to
20 km and wrong above it, since the real stratosphere warms again. Below sea level the
troposphere lapse continues without clamp or warning:

| altitude | temperature | density |
|---|---|---|
| −5,000 m | 320.65 K | 1.9305 kg/m³ |
| −50,000 m | 613.15 K | **30.468 kg/m³** |

Pinned by `test_the_atmosphere_extrapolates_below_sea_level_without_limit`.

**This is reachable, not hypothetical**, because A4 says nothing stops a trajectory at
h = 0. The two compound: a run that flies into the ground keeps integrating, and the air
it flies through gets denser without limit.

**Verdict: a declared design decision, left as it is.** Clamping changes behaviour for
existing callers, and JAX cannot raise inside `jit`, so a clamp would silently return a
plausible number for an altitude the model does not cover — the same objection as A4's.
The honest instrument is a caller-side check. Anything integrating near either end must
bound its own altitude and say so.

---

## E. Wind and turbulence

### E1. One-way coupling — the aircraft does not disturb the air

**Where:** the wind model contract; `wind_ned` is a function of position and state, never
of the aircraft's own circulation.

**Verdict: sound.** Every field in use is an atmospheric phenomenon orders of magnitude
larger than the aircraft. It would **not** hold for a wake-vortex encounter behind a
preceding aircraft, which §7 lists as a possible future field.

### E2. The aircraft is a point for translational gusts

**Where:** the wind field is evaluated at `state.pos_ned` — one point — and `field_model`
derives `omega_gust` from the analytic gradient, which is a **first-order** correction for
variation across the airframe.

**Bound, measured session 11 — and this has never been stated:**

| Field | Characteristic scale | In 747 spans (59.64 m) |
|---|---|---|
| **Parks Hannibal vortex, core radius** | 182.9 m | **3.07** |
| **Parks Morton vortex, core radius** | 137.2 m | **2.30** |
| Wingrove updraft, column radius | 2,359 m | 39.6 |
| Doyle lee wave, quarter wavelength | 6,250 m | 104.8 |
| Oseguera microburst, radius | 1,000 m | 109.8 (Cherokee spans) |

**The vortex is the marginal case, and it is the headline result.** At 2.3–3.1 spans the
linear-gradient `omega_gust` correction is doing real work rather than being a small
tidy-up, and second-order variation across the span is not represented at all. Everything
else in the project is 40 spans or more and is comfortably a point.

**Verdict: sound for the updraft, lee wave and microburst; the weakest link in the vortex
result.** It compounds with §5's existing ±25% band on the identified vortex parameters
and with §2's note that the rotational gust from a Wingrove-scale vortex already exceeds
the 747's full aileron authority by ~1.5×. Vortex conclusions should stay orderings, which
is what §5 already requires for a different reason.

**Bound, measured session 13 — the consequence, not just the scale ratio.** The point model
takes the gust gradient as the tangent at the CG. `wind.sampled_rates` fits the slope across
the airframe instead, over the span and the derived tail arm. The difference between them is
the cost of treating the aircraft as a point, and it is now measured along a traverse of the
Parks Hannibal core, **normalised by `V₀/r₀`, the core's own characteristic pitch-rate input**:

| Station | Correction (units of `V₀/r₀`) |
|---|---|
| 0.50 r₀ | **0.0000** |
| 0.99 r₀ | **0.0000** |
| **1.00 r₀** | **2.0000** |
| 1.10 r₀ | 0.7472 |
| 1.25 r₀ | 0.1086 |
| 2.00 r₀ | 0.0250 |
| 3.00 r₀ | 0.0072 |

**Two things this says that the scale ratio alone did not.**

1. **Inside the core the correction is exactly zero**, not merely small. Parks' Rankine profile
   is *linear* in radius, so a point sample plus an analytic gradient is not an approximation at
   all while the airframe is inside the core. **This is why the existing vortex results survived
   this assumption**, and it is specific to this field — it would not hold for a Dryden field or
   a wake vortex.
2. **At the core boundary the gradient is discontinuous.** The velocity is continuous there —
   both branches agree, which `test_the_two_forms_agree_at_the_core_edge` already asserted — but
   the derivative is not: inside `∂w/∂x = +V₀/r₀`, outside at `r = r₀` it is `−V₀/r₀`. The two
   one-sided derivatives differ by `2·V₀/r₀` and have **opposite signs**. So at the boundary the
   tangent is not merely inaccurate, it is *ambiguous*, and `vortex_wind`'s strict `<` resolves
   the tie toward the outside branch — the wrong side, since the airframe is still almost
   entirely inside the core. The fit has no such ambiguity.

**Verdict: bounded, and the bound is a profile rather than a number.** The correction is zero
where the aircraft spends most of the encounter and reaches a full sign reversal at one
crossing. Asserted by `test_the_curvature_correction_across_the_parks_core_is_measured` and
`test_the_rankine_gradient_is_discontinuous_at_the_core_edge`.

**What is still not bounded:** everything above concerns the *linear* fit. Genuine curvature
across the span — a profile that bends rather than ramps — is carried only by
`wind.strip_roll_moment`, and its accuracy rests on a DECLARED spanwise loading shape. The two
physically defensible shapes agree to **2.6%**; including a uniform distribution as a bracket
widens that to **49.7%**. See `atisim/provenance.py`, `strip.loading_shape`.

**Session 14: the strip path is now flyable.** The correction measured above is no longer
only a diagnostic — `loads.strip_model` feeds strip-integrated rolling moments into the
equations of motion through `integrate.step`. Any strip result **must be quoted with the
loading-shape sensitivity beside it** (2.6% across defensible shapes, 49.7% including a
uniform bracket), because that sensitivity is the dominant remaining uncertainty in it.

**And it moved nothing.** Measured, not assumed: flying the Parks Hannibal vortex both ways
gives a position difference of **0.000000 m**, and the Fig. 8 point is unchanged at
d(θ) 2.240°, d(n) −1.235 g. The updraft column is likewise unchanged. Both zeros have one
cause, and it is a property of the fields rather than a defect in the seam — the vortex has
no east variation, and the updraft is axisymmetric about an axis the aircraft flies straight
through, so in both cases every strip sees the same vertical gust and the antisymmetric roll
integral cancels. On a field that does vary across the span (a cubic in east) the same code
moves the aircraft 0.187 m, which is the only evidence that the seam works at all.

**Only the rolling moment is strip-integrated.** Pitch and yaw still come from the
point-plus-gradient treatment. The vortex's dominant input is *pitch*, so the headline Fig. 8
number is still produced entirely by the old path. **Do not read the strip path as having
fixed the vortex result** — it did not touch it. What it improves is the lateral response to
fields with genuine spanwise structure, which is what flying small-scale fields requires and
which none of this project's four source fields happens to have.

**What the rigid-rotation diagnostic does and does not say.** `gust_rates` reads three
entries of the velocity-gradient tensor and calls them `(p_g, q_g, r_g)`. That is lossless
exactly when the tensor is skew-symmetric, because a rigid rotation has three free parameters
and so does the triple. Inside the Parks core the diagnostic is **+1.0000** — solid-body
rotation, tensor skew — and since a rigid rotation is linear in position by definition, the
point treatment is exact there. That is the same fact the zero curvature correction above
records, reached from the other side. Outside the core it is **−1.0000**: irrotational flow,
symmetric tensor, pure **strain**, which the model has no channel for.

**A strip pitch integral would not fix that**, and an earlier draft of this section said it
would. A pitch integral gives each longitudinal station the gust at its own `x` rather than
fitting one slope, so it addresses *curvature in `w(x)`*; it never reads `∂u/∂z`. The strain
component and the curvature component are different failures, and that they coincide outside
the Parks core is a property of the Rankine profile rather than a general result. The
implication that does hold is one-directional: diagnostic `= +1` ⇒ rigid rotation ⇒ linear
field ⇒ point model exact.

**What a pitch integral would cost, since it is the obvious next step.** Not the integral —
the loading distribution. The roll integral calibrates cleanly because `Clp` is wing-dominated
and the elliptic chord gives the closed form `∫y²c dy = c₀b³π/64`, so `a₀ = −8·Clp` falls out.
Longitudinally there is no chord distribution: the load is wing plus tail, and
`airframe.effective_tail_arm` already attributes **both** `CLq` and `Cmq` entirely to the
tail, with an explicit do-not-re-attempt note recording that separating the wing's share via
Etkin's two-dimensional results made the wing 83% of `CLq` and implied a 23-chord arm. So a
pitch integral needs a DECLARED wing/tail load split — the number this project has already
derived, rejected and documented as unrecoverable — plus its own sensitivity sweep and its own
calibration against `Cmq`. It is not a smaller job than the roll integral was.

**Closed: the load-factor path now sees the increment.** `dynamics.specific_force` and
`dynamics.load_factor` take and forward it, and `vortex_viz._measure` re-invokes the load
model per sample exactly as it already re-invokes the wind model — necessary because it
receives a `State` trajectory rather than a `SimState` one, so the increment cached on
`SimState` is not in what it is handed. Because `specific_force` INVERTS `derivatives`' force
sum rather than recomputing it, only `CL` can ever reach it; `Cl`, `Cm` and `Cn` enter the
moment and cannot. That is asserted, not assumed, by
`test_a_lift_increment_reaches_the_load_factor`. It is also why this was exact rather than
merely small before the fix, while `strip_increment` populated `Cl` alone — and why every
measured number above is unchanged by closing it.

### E3. The field is frozen — wind depends on position, not time

**Where:** `field_model(field)` wraps `pos_ned -> wind_ned`.

**Why:** Taylor's frozen-turbulence hypothesis, and every source field is quoted as a
spatial structure.

**Verdict: sound for these fields**, whose evolution timescales (minutes for a microburst,
tens of minutes for a lee wave) are long against a traverse of seconds. **It stops being
true for Dryden**, which is a stochastic process in time — see §7's extensibility table.

### E4. Wind is sampled once per step and held across all four RK4 stages

**Where:** `integrate.step`. Its module docstring used to justify this as "the standard
treatment for Dryden and von Kármán turbulence"; **no source for that claim exists in this
repository and none was found**, and since it was the sole stated justification for a
choice that costs three orders of accuracy, the remediation pass replaced it with the
reasoning itself — a stochastic field is drawn from a key, so re-sampling it per stage
would make the realisation depend on dt and a convergence study would be measuring the
noise process rather than the integrator.

**Why it was the one seam session 11's verification did NOT cover.** The order-of-accuracy
test flies at fixed controls in still air, so it cannot see a wind term evaluated at the
wrong stage. The Galilean test uses a *steady* wind, whose material derivative is zero, so
it could not see a spurious `−m·dW/dt` term either — the other error `PROJECT.md` §2
explicitly warns about.

**Bound, measured session 12: no spurious body force, to 1e-9 m against an exact solution.**
Two tests, in `test_verification.py`:

| Test | Instrument | Result |
|---|---|---|
| `..._time_varying_uniform_wind_adds_no_body_force` | every aerodynamic coefficient and the thrust zeroed, so free fall is the **closed form** and the wind has no legitimate route into the equations at all; flown through a uniform wind swinging at 3 rad/s to 29.46 m/s, peak \|dW/dt\| = **88.39 m/s² (9.01 g)** | position matches `p₀ + v₀t + ½gt²` to **3.98e-12 m** over 300 steps, against a 1e-9 m bound |
| `..._step_ignores_the_wind_the_previous_step_applied` | full 747 aerodynamics; one step taken twice, varying **only** `SimState.wind_ned` — the cached previous wind, which is the ingredient such a term would be differenced from | **bit-identical** |

**An invariance assertion is the wrong instrument here, and that is worth recording.**
Writing `ṽ_b = v_b − Cᵀ W(t)` for the air-relative body velocity and differentiating gives
`ṽ̇_b = F(ṽ_b,ω)/m + g_b − ω×ṽ_b − Cᵀ Ẇ`: the air-relative state obeys the still-air
equation **plus** a `−Cᵀ Ẇ` term. So a time-varying wind is *not* a change of inertial
frame, and two runs offset by `W(0)` genuinely must diverge. The seam needs a closed form,
not an invariance.

**The falsification was run, since a test that can only pass demonstrates nothing.** With
the bug injected into `step` — differencing the cached previous wind against the current
sample, the one line anyone would write — the free-fall figure becomes **13.33 m**, ten
orders of magnitude above the 1e-9 m bound. Both tests fail on it, and the
Galilean test **passes with the bug still in** (2.7e-15 on quaternion, 6.9e-16 on `ω`,
against its own 1e-11 tolerances) once the wind cache is seeded consistently. Its blindness
is therefore measured rather than argued.

**Verdict on the body-force half: closed.** The seam Dryden will load is the one
now covered.

**Session 15 — the OTHER half of this entry, and it was not closed.** Everything
above concerns whether the hold introduces a spurious **force**. It does not. But
this entry's own second paragraph says the order-of-accuracy test "flies at fixed
controls in still air, so it cannot see a wind term evaluated at the wrong
stage", and nothing ever measured the **order** with a field switched on.
`verification.fixed_control_refinement` now takes a `wind_model`. Measured:

| Case | Observed order |
|---|---|
| still air (the control, same window, same aircraft) | **3.9891** |
| C∞ field — lee wave, 1.2 km, 25 m/s | **1.0537** |
| …with the hold removed (wind re-sampled per RK4 stage) | **4.0542** |
| across a Rankine core traverse | **non-monotone; no order exists** |

**The scheme is first order in a spatially varying wind field**, and the
falsification attributes it: restoring per-stage sampling restores fourth order.
This is a **property of the deliberate design choice**, not a defect — holding
the wind is right for a stochastic field, which is what the choice was made for,
and it is wrong only in the sense that a deterministic spatial field could do
better.

**Bound — CORRECTED in the remediation pass, and it was wrong by ~80×.** This entry
used to bound the seam by an h-vs-h/2 **discretisation** refinement inside the first
Parks core: 0.0169 m over a 366 m traverse, worth 0.0024 m/s of gust against a ~26 m/s
peak, "~1e-4 relative", and it concluded "no result the project quotes is affected".

**That is the wrong instrument.** The flaw is a **scheme** error, and h-vs-h/2 measures
the discretisation error with the hold still in place — refining dt refines a different
scheme, not this one. The right instrument is hold-vs-per-stage at the **same** dt.
Measured on the number `scripts/vortex.py` prints, the in-core Fig-8 Δθ:

| dt | wind held | wind re-sampled per stage | cost |
|---|---|---|---|
| 0.02 | 2.2596° | 2.2230° | **−1.62%** |
| **0.01 (published)** | **2.2400°** | **2.2216°** | **−0.82%** |
| 0.005 | 2.2271° | 2.2179° | −0.41% |

It halves with dt, which is what an O(h) error must do and is what identifies it as this
mechanism rather than another.

**No conclusion changes** — `PROJECT.md` §5 caps the vortex claims at orderings and puts
±25% bands on the identified parameters. **But the quoted 2.240° is not good to four
figures**, and this entry said it was. Its last two digits are scheme-dependent; quote
2.24°. Pinned by `test_the_wind_hold_costs_the_headline_figure_more_than_E4_bounds_it`.

**Verdict: bounded, and the trade-off is now stated rather than latent.** What
changes is expectation: refining dt through a wind field buys `O(h)`, not
`O(h⁴)`, so a convergence study that assumes fourth order will misread its own
output. Revisit if a deterministic field ever needs an accuracy the step size
cannot cheaply buy — per-stage sampling costs four field evaluations per step
instead of one, and a 64³ grid samples in 0.6–2.8 ms, so the cost is small; the
reason not to do it unconditionally is that it is wrong for Dryden.

### E5. Superposition is exact for the fields but not for the microburst's ground boundary condition

**Where:** `wind.superpose`.

**The sum itself is exact.** `wind_ned` is bit-identical to a hand-written sum and
`omega_gust` agrees to 6.5e-16, which is what makes "a vortex array sitting in background
turbulence" cost nothing beyond the two components.

**What does not survive addition is a boundary condition.** The microburst's defining
property — both components vanish at z = 0, which Oseguera & Bowles' introduction singles
out as *the* thing earlier analytic models got wrong — is a property of that field alone.
Superposing anything with a non-zero ground value destroys it. Measured: adding a uniform
3 m/s downdraft puts **exactly 3 m/s through the ground at every radius**, on the axis and
at the peak-outflow radius alike.

**Verdict: latent, and declared before it bites.** No run currently superposes onto the
microburst. The module comment saying summation "is exact within the model's own
linearisation" is true of the velocities and says nothing about boundary conditions, which
is the gap this entry fills. A composed field that includes the microburst must be checked
at z = 0 by its caller.

### E6. The microburst has no ceiling — the source's fourth parameter is not modelled

**Where:** `wind.microburst(u_max, radius, z_m)` carries **three** parameters. Oseguera &
Bowles specify **four**: their summary says "a microburst can be modeled by specifying four
characteristic parameters", and their symbol list defines the missing one as
`z_h` — *depth of outflow*.

**Bound, measured.** Without `z_h` the field has no upper bound. The on-axis downdraft
grows monotonically with altitude to a closed-form asymptote `λ(z* − ε)`:

| altitude | on-axis downdraft |
|---|---|
| 300 m (the project's penetration altitude) | 15.21 m/s |
| 1,000 m | 37.95 m/s |
| 3,000 m | 49.97 m/s |
| ≥ 10 km, forever | **50.64 m/s (98.5 kt)** |

**The transcription itself is exact** — eqs. (5)–(6) verified over 12,000 points to 4e-15
(`w`) and 1.6e-11 (`u`) — so this is a missing parameter, not a transcription error.

**Verdict: declared; implementing `z_h` is a scoped change, not a repair.** The audit puts
the paper's own `z_h` for this profile at 294.4 m, essentially the 300 m the project's
penetration starts at — so the runs on record sit inside the model's valid band, **but by
coincidence of the chosen altitude, not by anything the code enforces**. Adding `z_h`
changes the field and moves every microburst number the project reports, which makes it a
modelling decision with its own re-measurement.

### E7. The along-track shear index needs the caller's turn rate

**Where:** `wind.along_track_shear(pos_ned, vel_ned, accel_ned, field)`.

The index is `dU_x/dt` where `U_x` is the wind resolved along the ground track. Both the
wind **and the track direction** depend on time, so the derivative has two groups: Proctor
et al. Eq. (4), and the rotation of the track itself. Until the remediation pass the
heading was held fixed and the second group was missing entirely; the derivation and its
reduction to Eq. (4) at ψ̇ = 0 are now in that function's docstring.

**Bound, measured.** The term is `ψ̇ · (W_h · n̂)`, the cross-track wind times the track's
turn rate. At a **standard-rate turn one core radius above a Parks core**, where the
Rankine tangential velocity is fully horizontal and equal to `v₀`, it is worth
**ΔF = 0.1423** — the *entire* FAA 1 km alerting threshold of 0.1. The single-core closed
form `v₀·ψ̇/g` gives 0.1383 of that.

**It is exactly zero on every run the project reports**, and that is measured rather than
assumed: ψ̇ was **identically 0.0** at all 77,036 samples of the two lee-wave legs and the
microburst penetration, and both scripts' printed output was byte-identical across the
change. Every field in `wind.py` has zero east wind on the north axis and every run is
flown due north, so neither factor is ever non-zero.

**The residual assumption is on the caller.** A caller that passes `accel_ned = 0` gets
Eq. (4) and nothing else. The two shipped scripts rebuild the acceleration from the same
dynamics the rollout flew, so their straight track is a measurement; a new caller that
does not will under-report the hazard in a turn by up to the whole threshold.

### E8. The longitudinal station set is entirely aft of the CG

**Where:** `airframe.stations` builds its longitudinal set from `−arm` to `0`.

**Bound, measured.** The set's centroid is **−16.75 m** — every station behind the CG, none
in front — so `sampled_rates`' pitch channel is a **backward secant** rather than a centred
one, carrying an `O(arm/2 · f'')` bias. E2's headline claim that the correction is "exactly
zero inside the core, not merely small" therefore holds on the **downstream** half of a
core traverse and not the upstream half:

| position | correction, in `V₀/r₀` |
|---|---|
| +0.50 r₀, +0.99 r₀ | **< 1e-9** (E2's "exactly zero") |
| **−0.99 r₀** | **1.80** |
| +1.00 r₀ | 2.00 |
| −1.00 r₀ | 0.156 |

At −0.99 r₀ the CG is inside the core while the tail, 33.5 m behind it, is outside, so the
fit straddles the gradient discontinuity.

**Verdict: uncertain, and not repairable from held sources.** Centring the stations needs a
forward extent and **no source tabulates one** for any aircraft here; the aft arm is itself
derived from `−Cmq/CLq` rather than measured, and inventing a nose station would breach the
project's rule against inventing constants. Nothing the project publishes moves — the
default wind path uses `gust_rates`, the analytic tangent, not `sampled_rates` — so what
changes is the size of the error E2 attributes to the point model on the upstream half.

### E9. The vortex core branch resolves the tie at exactly `r = r₀` to the outside

**Where:** `wind.vortex_wind` selects the solid-body branch with a strict `r² < r₀²`, so a
sample landing exactly on the core edge takes the irrotational branch.

**Bound, measured: the choice is unobservable in value.** The two branches agree at the
boundary to **~1e-13** — the field is C⁰ there — so what differs is only which side's
*derivative* a differentiating sampler sees. The set of positions hitting the tie exactly
has measure zero, and round-off in the position decides it anyway.

**Verdict: leave it.** A one-character change picks the inside branch instead, and there is
no basis for calling that an improvement rather than a different arbitrary choice. What is
worth knowing is the related fact E8 and E2 both turn on: the *derivatives* differ by
`2·V₀/r₀` with opposite signs across `r = r₀`, so the right-hand side is C⁰ but not C¹ and
RK4 across it has an error depending on where the step grid lands.

---

## F. Numerics

### F1. Fixed-step RK4 at 50 Hz

**Bound, measured session 11:** observed order **3.99982** on a closed-form problem and
**3.98913** through the real 6-DOF. Asserted nowhere before that session.

### F2. The quaternion is re-normalised every step

**Where:** `integrate.step`, after `rk4_step`.

**Bound, measured:** `max |1 − ‖q‖| = 1.1e-16` after 3,000 steps — machine precision. The
projection is removing essentially nothing, which is the desired result: a normalisation
doing real work every step would mean the integrator was drifting off the unit sphere and
the projection was hiding it.

### F3. float64 throughout

Documented in `PROJECT.md` §1, with the reason: a Newton trim solve to 1e-10 and quaternion
norm stability over 1e5 steps are both marginal in float32.

### F4. There is a round-off floor on trajectory differences

**Bound, measured session 11:** at 40,000 ft `pos_ned` carries a 12,184 m altitude that
float64 resolves to 2.7e-12 m. The discretisation error reaches that floor near **7e-11 m**,
and refining past dt = 1/128 makes the answer *worse* — measured pairwise order **−0.685**
at dt = 1/256.

**Verdict: not an assumption so much as a ceiling, and it binds anything that measures a
difference of two trajectories at altitude.** It is why the order-of-accuracy window stops
at dt = 1/32. Any future convergence study must check it is above the floor before
believing its own slope.

**Re-measured session 23 on the ECEF state, and the floor MOVED DOWN because the
differenced quantity got smaller.** Position is now an offset from the anchor (A7), so a
trajectory difference no longer carries the aircraft's altitude at all — what gets
differenced is the ground track. It is isolated here two independent ways, and they agree.
The first is an exact symmetry: under `earth.FLAT` the Earth does not rotate and gravity's magnitude
depends on neither latitude nor longitude, so the whole problem is invariant under a
rotation about the Earth's axis. Flying the frozen 747 case from anchors at longitude 0 and
137° therefore runs **identical physics through different ECEF numbers**, and all of the
difference is round-off. Pinned by
`test_the_round_off_floor_on_an_anchor_relative_ecef_trajectory`:

| | |
|---|---|
| floor over the 20 s run | **1.153e-11 m** |
| ground-track offset reached | 4,717.87 m |
| ulp of that offset | 9.095e-13 m |
| floor, in ulp | **12.7** |

**The second route is session 11's own: a dt refinement sweep, which lands on the same
number and is what makes it believable.** Re-run
on the ECEF state (747 cruise, `WGS84_J2` at 47N, `dt_ref = 1/16384`):

| dt | error (m) | pairwise order |
|---|---|---|
| 1/32 | 1.4629e-08 | — |
| 1/64 | 9.2853e-10 | 3.978 |
| 1/128 | 7.1654e-11 | 3.696 |
| 1/256 | 1.6152e-11 | 2.149 |
| 1/512 | **1.2171e-11** | 0.408 |
| 1/1024 | 1.3799e-11 | **−0.181** ← refining now makes it worse |

**1.2e-11 m, against 1.153e-11 m from the longitude symmetry — a 5% agreement between a
discretisation experiment and an algebraic one.** Session 11's floor was 7e-11 m and was
reached at dt = 1/128; the same 7e-11 m is now merely a point still converging at order
3.7, and the sequence runs two more octaves before it stalls. **The gain is A7's**: the
differenced quantity no longer carries the aircraft's 12,184 m altitude.

**The shipped order-of-accuracy window is unchanged at 1/4 … 1/32 and did not need to
move** — the fitted order there is 3.9872, against the 3.98913 §4 records — but it now sits
even further above the floor than it was written to.

**And the FLAT-vs-pre-Earth comparison is NOT a measurement of this floor, though the
design doc expected it to be.** `earth.FLAT` was meant to reproduce the pre-Earth plant to
round-off; measured against the frozen pre-Earth trajectory it diverges by **4.892e-01 m**
over the same 20 s — ten orders above the floor above, and physics rather than arithmetic.
`FLAT` is a *curved* Earth: its constant gravity points along the local geodetic vertical
at the aircraft's own position, so gravity's **direction** rotates by `Vt/R` along the
flight path, where the pre-Earth plant had one fixed NED frame. The divergence is that term
and only that term — it matches `g V t³/(6R)` to better than 1.4% across the whole window
and grows cubically (1009.9× from 2 s to 20 s). Pinned by
`test_flat_is_a_curved_earth_and_its_cost_is_the_local_verticals_rotation`.

**Verdict: the ceiling is 6× lower than it was, and `FLAT` is a configuration of the
rotating plant rather than a way back to the old one.** Anything that needs bit-comparable
flat-Earth behaviour should compare against the frozen trajectory and expect 0.49 m at
20 s, not zero.

### F5. The strip integral at the shipped station count returns 82.6% of its own calibration

**Where:** `airframe.N_SPAN = 9`, which is what `loads.strip_model` builds.

`airframe.calibrated_lift_slope` sets `a₀ = −8·Clp` from the elliptic-loading identity
`Clp_hat = −a₀/8`. **That identity is exact only in the continuum limit**, and the
docstring said "exactly" without saying so. At the shipped station count the trapezoidal
quadrature returns:

| stations | strip `Clp` / tabulated `Clp` | error |
|---|---|---|
| **9 (shipped)** | **0.826** | **−17.4%** |
| 15 | 0.922 | −7.8% |
| 21 | 0.954 | −4.6% |
| 41 | 0.983 | −1.7% |
| 81 | 0.994 | −0.59% |
| 201 | 0.9985 | −0.15% |
| 2001 | 0.99995 | −4.7e-5 |

**Convergence measured, because "slow" needed a number: the observed order is 1.50**,
stable to three digits across every refinement from 21→41 up to 1281→2561. That is the
signature of the sqrt singularity in the elliptic chord at the tips, which the trapezoidal
rule cannot resolve. Extrapolating the fitted order: **~19 stations for 5% and ~56 for
1%.** Every existing test overrides the count (201, 2001, 21); none exercised 9 until the
audit added `test_the_production_station_count_is_the_one_that_is_wrong`.

**Verdict: `N_SPAN` stays at 9 and `a₀` stays calibrated in the continuum, deliberately.**
The two defensible repairs mean different things and neither is settled by any source
here:

- **Raise `N_SPAN` until converged.** Keeps `a₀` meaning what its docstring says. Costs
  field evaluations per step, and at order 1.5 the count needed for a genuinely converged
  integral is large.
- **Calibrate `a₀` against the discrete quadrature at the shipped `N`.** Makes the identity
  exact *in the code as run*, which is arguably what a calibration is for. But it makes
  `a₀` depend on `N_SPAN`, changing what the constant means — and `a₀` is already an
  *effective* value absorbing sweep and the tail's share of `Clp`.

Switching calibration basis silently is the one move that would be wrong, so neither was
taken in a remediation pass. **What this costs today is nothing the project quotes**: the
Parks core traverse gives point-vs-strip 0.000000 m because the field has no spanwise
variation there, so the headline number is the point model's either way.

**State the loading-shape sensitivity beside any strip result**: **2.6%** across the two
shapes that actually taper toward the tips, **49.7%** including the uniform bracket, which
is in the sweep as a bound and not as a candidate transport planform.

### F6. `airframe._active_shape` is process-global mutable state

**Where:** a module-level name set by a context manager, read by
`chord_distribution` — so a **physics parameter** is carried in process-global state.

**Bound: none, and that is the point.** Correctness under `pytest-xdist` or threads is
assumed and untested; the shipped suite runs single-process, where it is fine.

**Verdict: a refactor, not a bug fix.** Threading the shape through the call chain would
touch every strip call site. Anything that parallelises the suite or calls the strip path
from more than one thread must check this first.

### F7. A control channel with zero authority makes a Newton solve return NaN in silence

**Where:** the Cessna's `CYdr = Cldr = Cndr = 0`, a declared modelling choice in
`aircraft.py` — the source omits two of the three and the one it gives has the wrong sign
for this package's convention, and a side force with no matching yawing moment is worse
than no rudder.

**The undeclared consequence:** the rudder column of any control Jacobian is identically
zero, so `det = 0` and `jnp.linalg.solve` returns `[nan, nan, inf]` **without raising**.
Pinned by `test_a_control_channel_with_zero_authority_returns_nan_in_silence`.

**Bound: NO LONGER LATENT — it went live in session 23, exactly as predicted.** This entry
used to read "latent for the shipped solvers: `trim` does not carry rudder as an unknown, so
nothing in the package hits it. A steady-turn solve is the obvious next one that would." The
rotating-Earth trim carries **rudder as an unknown**, because Coriolis has to be trimmed
laterally — so the Cessna's zero rudder column became a real singular Jacobian and `trim`
returned six NaNs.

**Resolved, and the resolution is the one this entry asked for.** The Newton step is now
`jnp.linalg.lstsq` rather than `jnp.linalg.solve`, which takes the minimum-norm step and
leaves the null direction alone. Measured across the registry at 47°N: every aircraft
converges, the Cessna's rudder comes out at **exactly 0.0**, and all six of its residuals
reach 1e-13. It is trimmable without a rudder because the **aileron carries yaw through
`Cnda`** — that is why six residuals vanish and not five.

**The thing that had to be checked before adopting it**, since a solver that always returns
something could hide an unsolvable request — which would be worse than a NaN, not better:
stripping the 747 of pitch authority as well leaves `qdot` at **9.3e-5**, eight orders above
a converged solve, in the residual `trim` already returns. A loud failure stays loud. Pinned
by `test_a_rudderless_aircraft_trims_instead_of_returning_nan` and
`test_lstsq_does_not_hide_a_genuinely_unsolvable_trim`.

`rcond` is left at the default: measured 1.090e-12 on the 747 at 1e-6, 1e-10, 1e-14 and
None alike, so the residual is set by Newton convergence, not by singular-value truncation.

**Partial mitigation, and it is narrower than it looks:** `conftest.py` sets
`jax_debug_nans`, so the suite catches it. That is a **test-time setting only** — nothing
enables it for `scripts/` or for a library caller.

**Verdict: CLOSED.** This entry previously read "documented, not guarded", and reasoned that
since JAX cannot raise inside a jitted function the only available guard was an explicit
non-jitted precondition at the solver's entry. That reasoning was sound but it framed the
problem as *detection*, and the better answer turned out to be *not producing the NaN in the
first place*. `lstsq` needs no precondition, stays jittable, and keeps the `vmap` in
`minimum_drag_speed` that a non-jitted guard would have cost.

---

## Summary: which assumptions need action

| | Assumption | Status | Action |
|---|---|---|---|
| 1 | **E4** wind held across RK4 stages | **body force CLOSED session 12; ORDER measured session 15; BOUND CORRECTED in the remediation pass** | no spurious body force, to 1e-9 m against a closed form. But the scheme is **first order** in a spatially varying field (1.05 against 3.99 in still air; 4.05 with the hold removed). The old bound, 0.0024 m/s of gust error in the Parks core, measured the wrong quantity by ~80×: hold-vs-per-stage at the published dt costs **0.82%** of the headline in-core Δθ. No conclusion moves, but **2.240° is not good to four figures** |
| 2 | **B1** rigid airframe vs flexible data | **unquantifiable** | cap claims; do not assert structural fidelity |
| 3 | **C3** derivatives frozen across the envelope | **unbounded** | state the excursion with every result away from trim |
| 4 | **E2** point-aircraft gusts, vortex at 2.3–3.1 spans | **CLOSED for the linear fit, session 13; strip path flyable and measured, session 14** | correction is exactly 0 inside the core and 2.0·`V₀/r₀` at the boundary, where the gradient is discontinuous. Curvature beyond the linear fit rests on a DECLARED loading shape: 2.6% across defensible shapes, 49.7% including a uniform bracket. Flying the strip path moves the vortex result by **0.000000 m** — the field has no spanwise variation — so the headline number is still the point model's. **Roll only**; a pitch integral is the open work |
| 5 | **A2** constant g, +0.383% at cruise | **CLOSED, session 12** | not modelled: worst mode movement is 7.6% of its tolerance. Phugoid only carries the full 0.38% |
| 6 | **C5** no thrust moment, no spool | sound for now | required before any powered-recovery result |
| 7 | **B4** accelerometer at CG vs DFDR | caveat | keep Fig. 8 claims as orderings |
| 8 | **C9** every lift increment acts at the CG's relative wind | **bounded, wholly outside the envelope** | a real energy violation — +56.9 kW on the Cherokee at 8.26 rad/s — and **0 of 80,000** in-envelope states show it, threshold pitch rate 84–201 °/s, `max(E − E₀) = +0` exactly in every still-air run. Deliberately not repaired: the correction needs an arm for every lift channel and the only available one rests on an attribution this project has rejected |
| 9 | **F5** strip quadrature at 9 stations | **measured, decision recorded** | returns **82.6%** of its own calibration target, order 1.50, ~56 stations for 1%. `N_SPAN` and `a₀` both left as they are, because the two repairs mean different things and neither is settled here. Costs nothing quoted — the vortex field has no spanwise variation |
| 10 | **C11** the two tail-arm routes | **unresolvable** | 1.4% and 2.9% on the CR-2144 pair; **2.0× and 2.9×** on the two aircraft whose source file is missing. Do not adjust either derivative to make them agree |
| 11 | **E6** microburst `z_h` unmodelled | **bounded by run geometry, and only just** | on-axis downdraft grows to a **50.6 m/s** asymptote; the runs sit inside the valid band by choice of altitude, not by anything enforced |
| 12 | **A4 / D2** no ground, no atmosphere floor | **declared, guarded at one call site** | a 747-approach integrates to −698 m through air of increasing density. Truncate at your own clearance |
| 13 | **E7** along-track shear needs the caller's turn rate | **repaired, residual on the caller** | the term was missing entirely; worth **ΔF = 0.1423** at a standard-rate turn, and identically zero on all 77,036 samples of the runs on record |
| 14 | **F7** singular control Jacobian returns NaN silently | **latent** | no shipped solver carries rudder as an unknown; a steady-turn solve would be the first |

Items 1 and 5 — the two session 11 flagged as new and actionable — are both closed by
measurement, and in both cases the measurement changed the answer the reasoning had given.
Items 2 and 3 are honest limits rather than bugs, and the correct response to both is to
stop short of claims they cannot support.

**Items 8–14 were added by the remediation pass**, from the audit's list of assumptions
the code makes and the register never declared. Two things are worth saying about them as
a group. First, **most are not repairs and must not become repairs**: C9, C11, E6, E8, F5
and F7 are all cases where the flaw is measured and bounded and the correct form is *not*
established by any source this project holds, so changing the code would trade a documented
limitation for an invented certainty. Second, the one that *was* repaired — E7 — moved
nothing the project reports, and the reason is recorded as a measurement rather than an
argument, which is the standard the rest of this register is held to.

---

## How the notebook demonstrates validity

`notebooks/solver-validation.ipynb` exists because a table of numbers in a document cannot
be interrogated, and review specifically asked for an interface where problems with known
answers could be run by hand.

**What makes it evidence rather than decoration is the division of labour.** The notebook
contains **no arithmetic**. Every number it displays is computed by `atisim.verification`
or `atisim.validation`, and every one of those numbers is also asserted by a test in
`atisim/tests/`. So the notebook cannot drift from the code, and it cannot quietly
disagree with the suite. `pytest --nbval-lax notebooks/` executes it as a required gate, so
a notebook that stops working fails the build rather than rotting.

**Since session 23 it also states which Earth each register is asking for, because the same
function under the other one measures something else.** Tier 0 runs on `WGS84_J2` at 47N,
the Earth the package ships. The Caughey and CR-2144 comparisons run on `earth.FLAT` at sea
level, because every published matrix element and root they are checked against was derived
on a flat, non-rotating Earth, and because every tier-1 law here — Lanchester, the neutral
point, the affine mode relations — is a claim about the aerodynamic derivative chain and
not about the Earth. The notebook mirrors `test_verification.py` and `test_validation.py`
in this exactly, which is what keeps "the notebook runs the code the suite asserts on" true.

It demonstrates validity in three distinct registers, and the distinction is the point:

**1. Verification — run a problem whose exact answer is known.** Refine the timestep on a
harmonic oscillator and watch the error fall as dt⁴. Nothing about aircraft enters. If this
fails, the integrator is broken and no aerodynamic data can rescue it. The notebook plots
the refinement against a slope-4 reference line, so the claim is visual rather than a
single number to take on trust — and the cell that follows shows how close the finest step
is to the round-off floor (F4), which is the part a bare slope would hide.

**2. Sensitivity — change one coefficient, get a predicted change.** This is review's
request read literally. The notebook sweeps CD0 and Cmα and overlays the analytic law on
the model's output. What makes it convincing is that the sweeps are *falsifiable*: three of
the four relations originally expected were the wrong functional form, and the notebook
shows the affine fits that actually hold, with the intercept labelled as the term the
textbook approximation drops. A sweep that could only ever agree would demonstrate nothing.

**3. Reproduction — match an independent implementation element by element.** The notebook
prints the model's plant matrix beside Caughey's published one with a per-element
difference column, then shows the two omitted elements being reconstructed from his own
tabulated α̇ derivatives. This is the strongest form available: same input data, different
code, published intermediates. A reader can check any single element by hand.

**What the notebook deliberately does not do** is claim the model behaves like a real
Boeing 747. Every tier-2 comparison is closed-loop against a document's own arithmetic —
which is exactly why the age of that document is not a threat, and equally why the result
says nothing about the real aeroplane. The closing cell states that limit explicitly, and
this file is where the reasons live.

**To extend it:** add the computation to `verification.py` or `validation.py`, assert it in
the corresponding test, then add a notebook cell that calls it. Never the other way round.
