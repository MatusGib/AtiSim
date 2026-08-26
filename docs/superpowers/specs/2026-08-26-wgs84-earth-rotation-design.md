# WGS-84 and Earth rotation — design

**Session 23.** Replaces the flat, non-rotating Earth with the ellipsoid and rotation rate
JSBSim uses, in the formulation JSBSim uses.

Retires `ASSUMPTIONS.md` A1 (flat, non-rotating Earth) and A2 (constant gravity), rewrites
A3, and re-measures F4. Turns `audit/INVENTORY.md` R2 from "implicit, by omission" into an
explicit, cited model.

---

## 1. Why

`test_jsbsim_737_layers.py` bounds its transverse residuals against an **Earth-rotation
floor** — 0.409 m/s in `u` over 20 s — and says in its own docstring that this is what
atisim "cannot reproduce even in principle, being flat-Earth and non-rotating". That
allowance is the largest single un-modelled term in the JSBSim comparison. This work
removes the excuse and turns the allowance into an assertion.

`ASSUMPTIONS.md` A1 records the flat Earth as **sound for the current scope** and it is:
0.0035 g of Coriolis against ~1 g excursions, 6.88 m of position error per 20 s. Nothing
here contradicts that verdict. The reason to do the work anyway is that A1 also names what
it is *not* sound for — "any claim about ground track over hundreds of km, or a run longer
than about ten minutes" — and every one of those limits is a limit on where the project
can go next.

## 2. The JSBSim formulation, established from the binary

**This section contains no transcribed equations.** JSBSim ships as a compiled wheel with
no C++ source, so the formulation was recovered by driving the running engine
(v1.3.1, GitHub build 1837, commit 3b25f25e) and solving for the relations that close.
Residuals are in brackets and are machine zero unless stated.

Notation: `v` is velocity relative to ECEF in body axes (JSBSim `vUVW`); `w_be` is the body
rate relative to ECEF (`vPQR`); `w_bi = w_be + Om_b` is the body rate relative to ECI
(`vPQRi`); `Om_b = T_e2b Om` with `Om = [0,0,Omega]` in ECEF; `r_e` is absolute ECEF
position; `F_b` is aero + propulsion + gear, **excluding weight**; `g_b` is gravitation in
body axes.

```
rdot_e   = T_b2e . v
vdot     = F_b/m + g_b - (w_be + 2 Om_b) x v - [Om x (Om x r_e)]_b     [0.0, exact]
qdot     = 0.5 q (x) [0, w_be]                        (q: body -> ECEF)
wdot_be  = I^-1 (M - w_bi x I.w_bi) + w_be x Om_b               [2.4e-14, 6.9e-18]
```

### Three conventions that are pinned, not assumed

| Convention | Evidence | The wrong choice costs |
|---|---|---|
| The local NED frame uses **geodetic** latitude | `T_e2l(geodetic) @ v_ecef` reproduces JSBSim's `v-north/east/down` to machine precision; geocentric does not | 65.2 m/s in `v_north` at 47°N |
| Inertia off-diagonal enters as **+Ixz** | `I^-1(M - w x Iw)` closes to 2.4e-14 with `+Ixz`, 2.3e-3 with `-Ixz` | 4.4% in `pdot` |
| `g` is J2 **gravitation only**; centrifugal is a separate term | JSBSim reports 9.8142 m/s² at the equator, not the 9.7803 of apparent gravity | 0.0339 m/s², and it would be double-counted |

The `+Ixz` result independently corroborates the sign correction made in session 19.

### The frame-transfer sign

`wdot_be = wdot_bi + w_be x Om_b`. The first probe of this was **inconclusive** — with the
aircraft near wings-level the term is 1e-5 and the residual of the wrong sign was the same
order, so both signs "fit". Re-probed with aileron and rudder deflected to raise |w_be| to
0.15 rad/s, the two separate cleanly: **+** gives 6.9e-18, **−** gives 1.9e-5. This is
recorded because the first measurement would have shipped the wrong sign.

### Constants

Recovered from the binary, not quoted from a table. All are exact WGS-84.

| Symbol | Value | How recovered | Agreement |
|---|---|---|---|
| `a` | 6378137.0 m | `inertial/sea-level-radius_ft` at the equator | exact |
| `1/f` | 298.257223563 | from `a` and the polar sea-level radius | 8e-10 rel |
| `GM` | 3.986004418e14 m³/s² | solved from equatorial and polar gravity | 4.1e-13 rel |
| `J2` | 1.08262982e-3 | solved from the same pair | 1.3e-11 rel |
| `Omega` | 7.292115e-5 rad/s | `d(position/epa-rad)/dt` over 10 s | exact |

**`b` is DERIVED from `a` and `f`, not taken from JSBSim.** JSBSim's reported semi-minor
axis is 6356752.314186 m; `a(1-f)` is 6356752.314245 m. The 5.87e-5 m difference is
JSBSim's internal storage in feet round-tripping, not a different ellipsoid. The defining
constants are `a` and `f`, so those are what this model carries, and the discrepancy is
recorded here rather than reconciled.

## 3. `atisim/earth.py`

New module. Pure functions, no state.

- `geodetic_to_ecef(lat, lon, h)` and `ecef_to_geodetic(r)`.
  `ecef_to_geodetic` uses **Bowring's method at a fixed 3 iterations** — fixed rather than
  converged so it stays jittable and differentiable. Measured worst round-trip ALTITUDE
  error over lat in [-90°, 90°] x h in {0, 5, 12.192, 20} km: **1 iteration 3.5e-6 m,
  2 iterations 1.4e-8 m, 3 iterations 3.7e-9 m.** Three is chosen because it reaches the
  float64 floor — **9.3e-10 m** is the ulp of an ECEF coordinate — so further iterations
  buy nothing.

  **Corrected after implementation.** This paragraph first read "3 iterations 1.4e-9 m"
  against "a 1.4e-9 m ulp", which conflated two different quantities: 1.4e-9 m is the
  worst LATITUDE error expressed as a distance, not the altitude error, and the ECEF ulp
  is 9.3e-10 m rather than 1.4e-9 m. The range was also stated as [-89.9°, 89.9°]; the
  non-singular altitude form actually shipped is valid at the poles, so the measurement
  covers [-90°, 90°] inclusive. As implemented and measured: worst altitude error
  **2.94e-9 m**, worst latitude error **4.58e-11 arcsec**.
- `ecef_to_ned_matrix(lat, lon)` — geodetic latitude, per §2.
- `gravitation(r_ecef, model)` — J2 or inverse-square.
- `EarthModel`, a NamedTuple carried as a **static** argument so its branches resolve at
  trace time and cost nothing under `jit`. Three instances: `WGS84_J2` (the default, and
  JSBSim's), `WGS84_INVERSE_SQUARE` (JSBSim's `gravity-model = 0`), and `FLAT`.
- `Anchor(lat0, lon0, h0, r_ecef0, T_e2l0)` — the run's geodetic origin. Static, not state.

Every constant gets a `provenance.py` entry: **SOURCED**, WGS-84, with the verification
against JSBSim v1.3.1 build 1837 named as the check rather than as the source.

## 4. The state — still thirteen elements

```python
class State(NamedTuple):
    pos_ecef: Array   # (3,) m, OFFSET from the run anchor's ECEF position
    vel_body: Array   # (3,) m/s, ECEF-relative, body axes   (JSBSim vUVW)
    quat: Array       # (4,) w x y z, body -> ECEF
    omega: Array      # (3,) rad/s, omega_b/e, body axes     (JSBSim vPQR)
```

**Position is stored as an offset from the anchor, which is a deliberate deviation from
FGPropagate and the only one.** JSBSim carries absolute ECEF. The physics is identical —
it is an origin shift — but the numerics are not: an absolute ECEF coordinate at 6.39e6 m
has an ulp of 9.3e-10 m against 1.8e-12 m for `pos_ned` at 12,184 m, a factor of **512**.
F4 already records that the discretisation-error floor at altitude is 7e-11 m and that
refining past dt = 1/128 makes convergence studies *worse*. Absolute ECEF would push that
floor to roughly 3.6e-8 m and bind `verification.py`'s order-of-accuracy window. The offset
keeps trajectory differences at their present resolution; absolute position is formed only
where gravity and geodesy need it, where the precision does not matter.

The shape is unchanged, so `rk4_step`, `_axpy`, `batch_sim`, and every `lax.scan` carry
work untouched.

### The NED view

`pos_ned(state, anchor) = anchor.T_e2l0 @ state.pos_ecef` — a **pure rotation** of the
offset. No differencing of large numbers, which is what preserves F4.

`altitude(state, anchor)` becomes true geodetic height via `ecef_to_geodetic`, not
`-pos_ned[2]`. These differ: the tangent plane falls away from the ellipsoid as
`d²/2R`, which is 7.8 m at 10 km of ground track and 785 m at 100 km. Anything reading
`-pos_ned[2]` as altitude must move to `altitude()`; this is the single most likely place
for a silent regression and every call site is audited in the plan.

## 5. Dynamics

`derivatives` implements §2 directly. Wind still enters here and nowhere else. `aero.py` is
untouched — it continues to see only `vel_rel` and `omega_rel`, and the standing rule that
it never sees inertial velocity is unaffected.

**The Coriolis term keeps the inertial velocity**, exactly as the existing `-w x v` does.
PROJECT.md §2 names substituting `vel_rel` there as a classic gust-modelling error that
breaks Galilean invariance; adding an Earth rate does not change that argument, and
`test_verification.py`'s Galilean test still guards it.

## 6. Wind fields are unchanged

Every field — vortex, updraft, lee wave, microburst — takes a NED position and keeps
taking one. `derivatives` hands it `pos_ned(state, anchor)`.

This is **correct rather than a compromise**: a wake vortex or a microburst is a local
phenomenon anchored to a place on the ground, and the tangent plane at the anchor is the
frame it is defined in. The fields' own scales (182.9 m to 6250 m) are far below the range
at which tangent-plane curvature matters.

## 7. Trim gains three unknowns

On a rotating Earth, wings-level level flight is **not an equilibrium**: Coriolis puts
~0.0035 g of lateral acceleration into `vdot`, which the present three-unknown solver
neither sees nor can cancel.

```
unknowns  [alpha, elevator, throttle]  ->  [alpha, elevator, throttle, phi, aileron, rudder]
residual  [udot, wdot, qdot]           ->  [udot, vdot, wdot, pdot, qdot, rdot]
```

**The count matters and is easy to get wrong.** Six residuals against the available
freedoms — `alpha`, `beta`, `phi`, `elevator`, `aileron`, `rudder`, `throttle` — is
**seven unknowns for six equations**, a one-parameter family rather than a solution. The
system is closed by imposing **`beta = 0`**, which is the coordinated condition and the
one every existing trim in this project already assumed implicitly. That leaves six and
six, square, and Newton with a forward-mode Jacobian applies unchanged.

`aileron` is in the set and it is not optional: with `beta = 0` and a deflected rudder,
`Cl` is non-zero through `Cl_dr`, so the roll-moment residual cannot vanish without it.
Trimming `phi` and `rudder` alone would leave `pdot != 0`.

The level-flight constraint is no longer `theta = alpha`. With `beta = 0` and `gamma = 0`
it becomes `tan(theta) = cos(phi) tan(alpha)`, which reduces to the old form at
`phi = 0`. At the ~0.2° of bank this trim actually produces the two differ by well under a
microradian, but the constraint is written in its exact form rather than approximated,
because nothing downstream would reveal it if it were wrong.

The trimmed bank angle is a function of latitude and heading. **The 0.2° figure is an
estimate from `atan(0.0035)` and is not yet measured** — the plan's step 3 measures it, and
if it disagrees materially that is a finding, not a tolerance to adjust.

`is_physical` extends to the three new unknowns: `phi` bounded, `aileron` and `rudder`
against `ac.aileron_limit` and `ac.rudder_limit`.

## 8. What FLAT means now, and what it does not

`FLAT` is `Omega = 0`, spherical Earth, constant `G0` — **one plant, configured**, not a
second implementation retained alongside.

**It does not reproduce today's trajectories bit-for-bit, and the claim is weakened rather
than dropped.** Bit-identity is unreachable through an ECEF state: `pos_ned` is formed by
rotating an offset that was itself accumulated in ECEF, so the arithmetic differs even when
the physics does not. The test asserts agreement **to a measured round-off floor**, and
that floor is measured and recorded in F4 rather than assumed.

This is a real weakening of the guarantee the `increment=None` and `load_model=None`
constructions give elsewhere in the codebase, and it is called out here so that no later
reader mistakes the FLAT test for a bit-identity test.

## 9. Verification

| # | Check | Where |
|---|---|---|
| 1 | Term-by-term against JSBSim over a lat x alt x attitude grid: geodetic/ECEF round-trip, `T_e2l`, gravity vector, and each EOM term of §2 | new `scripts/gen_jsbsim_earth_reference.py` -> frozen `jsbsim_earth_reference.xml` -> `test_earth.py` |
| 2 | Closed-form invariants: a mass at rest on the equator stays at rest; free-fall along the local vertical; great-circle transport rate; `Omega -> 0` recovers today's equations | `verification.py` (tier 0 — no aircraft data) |
| 3 | The 737's 0.409 m/s Earth-rotation floor, reproduced rather than allowed for | `test_jsbsim_737_layers.py` |
| 4 | FLAT against the pre-change model, to a measured round-off floor | `test_earth.py` |

The reference is **frozen and checked in**, following `jsbsim_ref.py`'s standing rule: the
suite must run on a machine with no JSBSim installed, and reference drift must show up in
`git diff` rather than as a mysterious change in test results. `test_earth.py` may not
import `jsbsim`.

Check 2 exists because check 1 alone cannot catch an error the two engines **share** —
a sign convention both get wrong is invisible to a comparison and visible to a closed form.

## 10. What this costs

**The whole of §4's evidence ledger is re-measured on the rotating WGS-84 Earth.** This is
not tolerance-widening, which §4 forbids: it is a stated physical change to the model, with
the cause named and the old rows superseded rather than deleted, per the standing rule.

Also re-measured or rewritten: A1 and A2 retired; A3 (geopotential altitude) rewritten now
that geodetic altitude is available; F4 re-measured at the new position magnitudes;
`audit/INVENTORY.md` R2 and `NOTATION.md`'s frame table.

## 11. Decisions taken, and what was rejected

| Decision | Rejected alternative | Why |
|---|---|---|
| Full ECEF/ECI state | Earth-rate terms on the existing NED state | Asked for explicitly. The NED variant is numerically the same over seconds-to-minutes but keeps the flat position bookkeeping A1 names as the limit |
| Anchor-relative position | Absolute ECEF, as FGPropagate | 512x ulp, and F4 already binds convergence studies at the present magnitudes |
| On by default | Off by default, bit-identical | Asked for explicitly, with the ledger re-measurement accepted as the cost |
| Both gravity models, J2 default | J2 only | Mirrors JSBSim, and lets a test isolate what J2 alone is worth |
| FLAT to a measured floor | Two plants for true bit-identity | Two 6-DOF implementations to keep in step forever is the duplication this codebase otherwise refuses |
| Trim gains bank, aileron and rudder, with `beta = 0` closing the system | Keep 3 unknowns, bound the residual | Every run starts from trim; a residual `vdot` would put a slow lateral drift under the entire ledger |

## 12. Staging

```
1. earth.py: constants, geodesy, gravity, EarthModel   -> verify: check 1 (geodesy and
   No state change. Fully testable alone.                          gravity rows), Bowring
                                                                   round-trip at 1.4e-9 m
2. State + derivatives + NED accessors                 -> verify: check 1 (EOM rows),
                                                                   check 2, check 4
3. Trim to six unknowns, beta = 0                      -> verify: all six residuals at
                                                                   machine zero; the bank
                                                                   angle MEASURED at 47N
4. Consumers: wind, sensors, checks, viz, trim,        -> verify: full suite green;
   autopilot, panel, analysis, scripts                           every -pos_ned[2] site
                                                                 audited
5. Re-measure the ledger; rewrite A1-A3, F4, R2        -> verify: check 3 asserts the
                                                                   0.409 m/s floor
```

Steps 1–3 are self-contained. Step 4 is the wide one — 43 files read `pos_ned` — and step 5
cannot start until 4 is green.
