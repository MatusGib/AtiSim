# Agent C — Units and Conventions

Evidence-only. No causal attribution, no remediation.

**Scope:** `aero.py`, `dynamics.py`, `aircraft.py`, `atmosphere.py`, `wind.py`,
`airframe.py`, `loads.py`, `trim.py`, plus `units.py` and `state.py` (the
quaternion and control-sign conventions live there and are consumed by all of
the above).

**Method rule applied throughout:** nothing is accepted from a docstring. Every
convention is driven through the running code, and every reference value is
either a defining standard, an independently written formula in the test script,
or a number read from CR-2144 and restated in the script rather than imported.
Where a sign could cancel, the case is made deliberately asymmetric.

**Interpreter:** `C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe`

**Scripts:** `audit_evidence/scripts/c1..c10*.py`
**Captured output:** `audit_evidence/logs/*.log`

**Nothing under `flightsim/` or `docs/` was modified.** Repo baseline confirmed
green before and after: `pytest` over `test_units, test_aircraft, test_aero,
test_state, test_wind, test_airframe, test_loads, test_atmosphere, test_trim,
test_dynamics` → **176 passed, 1 skipped**.

**Result: 230 checks pass, 2 fail.** Both failures are the same finding (C-F1).

| script | pass | fail |
|---|---|---|
| `c1_units` | 18 | 0 |
| `c2_quaternion` | 14 | 0 |
| `c3_signs_flown` | 66 | 0 |
| `c4_ixz` | 22 | 0 |
| `c5_wind_axis` | 9 | 0 |
| `c6_gust_rates` | 21 | **2** (C-F1) |
| `c7_nondim` | 43 | 0 |
| `c8_dimensional` | 37 | 0 |
| `c9_strip_stations`, `c10_cn_vs_cl` | measurement only | — |

---

## Headline

| # | Finding | Kind | Size |
|---|---|---|---|
| **C-F1** | `wind.strip_clp_from_rate` does **not** reproduce `Clp` at the shipped `airframe.N_SPAN = 9`; it returns **82.64%** of it. Two docstrings claim the identity holds *exactly*. The repo's own calibration test passes only because its helper overrides the count to `n_span=2001`. | magnitude / quadrature | **−17.36%** on the strip rolling coefficient, reaching a flown trajectory |
| **C-F2** | `aircraft.from_dimensional_longitudinal` (and the inline copy in `_boeing_747`) inverts CR-2144's `Z`/`X` derivatives as if Appendix A defined them via `C_L`/`C_D`; Appendix A defines them via `C_N` (up-positive) and `C_X` (aft-positive). Exact only at α₀ = 0. | axis-definition | **−0.66%** on `CLa`, −0.60% on `CDa`, −0.12% on `CD0`; **747 cruise set only** |
| **C-F3** | `aircraft._B747_G = 32.174` is an inlined lb→slug conversion, contrary to `units.py`'s stated rule "Never inline a conversion factor anywhere else", and is a 5-s.f. truncation of `g₀ = 32.17404855643044 ft/s²`. | policy / precision | **+1.51e-6** relative on the mass used inside the conversion chain vs `ac.mass` |

**No sign error was found anywhere.** Every sign convention listed in the brief
was tested on a deliberately asymmetric case and every one is correct — see the
confirmations table below.

C-F2 was independently recorded by an earlier audit phase in `NOTATION.md` §3/§7.2;
I re-derived and re-measured it rather than transcribing it, and my number
(−0.66% on `CLa`) is consistent with that document's "~1% level". C-F1 and C-F3
I did not find recorded anywhere (C-F3's *existence* is noted in `NOTATION.md`
§8 with a slip of "2.4e-6"; my measured value is 1.51e-6).

---

## Main evidence table

Status key: **OK** = confirmed correct; **FINDING** = defect or discrepancy;
**NOTE** = correct but worth recording; **UNTESTABLE** = cannot be settled
without ambiguity.

### 1. Units and dimensional consistency

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| `FT2M` | 0.3048 exact (1959 intl foot) | `==` against a literal from the yard-and-pound agreement | `0.3048` exact | OK |
| `LBF2N` | 4.4482216152605 exact | built as the exact rational `Fraction(45359237,10⁸)·Fraction(980665,10⁵)`, compared bit-exact | matches exactly; = NIST SP811 B.9 | OK |
| `LB2KG` | 0.45359237 exact | `==` literal | exact | OK |
| `SLUG2KG` | `LBF2N/FT2M` | vs 14.593902937206362; cross-check `SLUG2KG/LB2KG == g₀[ft/s²]` | `14.593902937206362`, ratio `32.17404855643044` exact | OK |
| `SLUG_FT2_TO_KG_M2` | `SLUG2KG·FT2M²` | vs 1.3558179483314003, **and** via the independent identity slug·ft² ≡ lbf·ft·s² | both agree bit-exact | OK |
| `KT2MS` | 1852/3600 exact | `==` (1929 intl nautical mile) | `0.5144444444444445` | OK |
| `HP2W` | 550 ft·lbf/s exact | `== 550·LBF2N·FT2M` | `745.6998715822702`; NIST 745.6999 | OK |
| `SLUG_FT3_TO_KG_M3` | `SLUG2KG/FT2M³` | vs 515.3788183931961 | exact | OK |
| `DEG2RAD`, `RAD2DEG`, `PER_DEG_TO_PER_RAD` | π/180, 180/π, 180/π | `==` definitions; round trip | exact | OK |
| Factors never inlined | `units.py` docstring rule | regex sweep of `flightsim/` + `scripts/` for all 13 factor literals | **one violation**: `aircraft.py:222 _B747_G = 32.174` | **C-F3** |
| `trim.ALPHA_LIMIT` | — | uses `math.radians(15.0)` not `DEG2RAD` | numerically identical; policy deviation only | NOTE |
| ISA atmosphere | ICAO Doc 7488 | `T, p, ρ, a` at 0 / 5000 / 11000 / 12192 / 20000 m vs published ISA table | worst error **3e-6 relative** (ρ at 12192 m); tropopause seam continuous to 1e-9 | OK |
| Aero force scaling | `F = q̄·S·C` | measured ratios on doubling ρ, tripling V, ×5 on S | exactly 2, 9, 5 to <1e-11 | OK |
| Moment reference lengths | `l,n` ← span; `m` ← chord | doubled `b` and `c` separately, measured the (l,m,n) response | span → (2,1,2); chord → (1,2,1) | OK |
| `thrust_force` | N, lapses as (ρ/ρ₀)^0.8 | full throttle at ρ₀ vs `max_thrust`; ratio at ρ₀/2 | 773990.6 N = 174.00 klbf = 4×43.5 klbf exactly; ratio = 0.5^0.8 | OK |
| Gravity | `(0,0,G0)` NED | drag-free, thrust-free, ω=0, at φ,θ,ψ = 0.4, −0.3, 1.1 | NED accel = `(4.8e-16, −8.4e-16, 9.80665)` | OK |
| Coriolis `−ω×v` | does no work | rate of change of speed from the transport term alone | exactly 0.0 | OK |
| `f_factor` | dimensionless; +ve hazardous | descending air mass (w=−5) and accelerating tailwind | +0.025 and +0.102; `f = 1/G0` for shear 1 m/s² | OK |
| `from_dimensional_*` helpers | dimensionally self-consistent with `aero.coefficients` | 18 dimensional derivatives pushed through the helpers onto a scratch aircraft, then read back out of the plant by finite difference | **all 18 round-trip to ratio 1.000000** | OK |
| `from_dimensional_longitudinal` axis basis | Appendix A `Z`/`X` ← `C_N`/`C_X` | exact `C_N = C_L cos α + C_D sin α`, `C_X = C_D cos α − C_L sin α` solved simultaneously at α₀=4.60° and compared to the code's α→0 form | `CLa` 4.944127 (code) vs 4.977143 (exact) | **C-F2** |

**C-F2 blast radius, measured:** 747 cruise only. The approach set is
non-dimensional in Table IX-2 (`CLa = 5.70` verbatim). The Cherokee is
linearised at α₀ = 0, where `C_N ≡ C_L` and `C_X ≡ C_D` identically, so the
helper is exact for it. The Cessna sends only `CLde`/`Cmde` through the helper,
and Appendix A's `Z_δe`/`M_δe` carry no α term, so both are exact.

### 2. Sign conventions, traced end to end on asymmetric cases

Every row was driven through `dynamics.derivatives` **and** flown for 3 s
through `integrate.rollout` from a converged trim (trim residuals 1e-20 to
1e-15). All four registry aircraft unless stated.

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| α | positive nose-up | aircraft pitched +8°, velocity vector held due north (level flight path) so α must equal θ exactly; plus ±w probes | `α = +8.000000°` to 1e-9; `w>0 ⇒ α>0`, `w<0 ⇒ α<0` | OK |
| β | positive = relative wind from the **right** | heading north, wind **from the east** at 15 m/s through `dynamics.relative_velocity` | `v_rel = (100, +15, 0)`, `β = +8.531°`; mirrored case gives β<0 | OK |
| β physical consequence | restoring | `CYb, Cnb, Clb` driven at β>0 on all four aircraft | force **left** (Fy<0), weathercock **nose right** (N>0), roll **away** (L<0) — all 4 aircraft | OK |
| δe | +ve = TE down ⇒ `CLde>0`, `Cmde<0`, nose **down** | +5° from trim: coefficient signs, `q̇`, flown Δθ over 3 s, and ΔFz at frozen α | `CLde` +0.364/+0.338/+0.933/+0.468; `Cmde` −1.444/−1.340/−2.390/−1.154; `q̇<0` and Δθ<0 in all 4; ΔFz<0 (lift up) in all 4 | OK |
| δa | +ve ⇒ right-wing-down ⇒ `Clda>0` | +5° from trim: `Clda`, `ṗ`, flown φ over 3 s | `Clda` +0.0136/+0.0461/+0.0531/+0.4173; `ṗ>0` and `φ>0` in all 4 | OK |
| δr | +ve = TE left ⇒ `CYdr>0`, `Cndr<0`, nose **left** | +5° from trim: coefficient signs, `ṙ`, flown Δψ, resulting β, and ΔFy at frozen attitude | `CYdr` +0.115/+0.175/+0.116; `Cndr` −0.125/−0.109/−0.051; `ṙ<0`, Δψ<0, β>0, ΔFy>0 (force to the **right**) — all consistent | OK |
| δr, Cessna 172 | — | `CYdr = Cldr = Cndr = 0` by deliberate design decision in the source comment | no convention exists to test | **UNTESTABLE** |
| `Ixz` sign | tabulated `Ixz` is +ve; enters the tensor **negated** | (a) tensor structure; (b) `rdot/L` vs analytic `Ixz/(IxIz−Ixz²)`; (c) gyroscopic `−ω×(Iω)` for pure roll; (d) **round trip against CR-2144 Table IX-8's PRIMED values** | tensor `I[0,2] = −1315219.3` = −Ixz; symmetric, positive definite; `rdot/L` matches analytic to 1e-18; pure roll gives nose-**down** pitching moment `−Ixz p²` | OK |
| `Ixz` end to end | plant must reproduce Table IX-8 primed values | drove the plant with pure β, p, δa, δr at the exact CR-2144 condition (V=235.92 m/s, q̄=8474.81 Pa) and compared `[I⁻¹M]₀`, `[I⁻¹M]₂` to `L'`, `N'` | **all 8 values round-trip to ±0.000%** (machine precision) | OK |
| — negative control | test must have teeth | repeated the above with `−Ixz` in the tensor | `N'_p` off by **+57.3%** — the test bites | OK |
| `_unprime` | inverts Appendix A A-18 | re-applied the primed relation, written out independently, to the recovered raw values; plus inertia-unit-invariance check | recovers the table to <1e-12 for all four inputs; identical in slug·ft² and kg·m² | OK |
| `quat_to_dcm` | body → NED, `[w,x,y,z]` | vs `Rz(ψ)Ry(θ)Rx(φ)` built from elementary matrices at φ,θ,ψ = 0.37, −0.21, 1.13 (all non-zero, unequal) | **max error 2.2e-16**; `|C − Cᵀ|max = 1.76` so the sense is unambiguous | OK |
| — physical spot checks | | θ>0 ⇒ body-x has −D; φ>0 ⇒ body-y has +D; ψ>0 ⇒ body-x has +E | all three correct | OK |
| `quat_to_euler` | 3-2-1 | round trip through `euler_to_quat` | 2.2e-16 | OK |
| `quat_derivative` | `q̇ = ½ q⊗[0,ω]`, body rates | vs `R(t+dt) = R(t)·expm(skew(ω)dt)` built with Rodrigues only, at ω = (0.17, −0.29, 0.11); convergence order measured | errors 3.3e-12 / 3.3e-15 / 1.1e-16 at dt = 1e-3/1e-4/1e-5; **convergence order 3.00** | OK |
| — order-3 explanation | | Euler's O(dt²) term of `q·exp(ωdt/2)` is purely real and `quat_normalize` removes it exactly, so a *correct* derivative shows order 3, not 2 | my first expectation of order 2 was wrong; the code is right | NOTE |
| — negative control | | integrated with `−ω` | error 6.25e-4 ≈ `2|ω|dt` — rejected | OK |
| — right-multiplication | body rates ⇒ `q⊗Δq` | vs exact axis-angle quaternion product written out in the script | 1.5e-12 | OK |
| `pos_ned = DCM @ vel_body` | | θ=+0.10, α=+0.10 must give zero descent | NED-down velocity 1.4e-16 | OK |

### 3. Gust-rate signs

The decisive construction: an air mass in **rigid rotation** at a known rate
about **one body axis at a time**. An aircraft rotating *with* the air must see
an effective rate of exactly zero. Attitude deliberately non-trivial
(φ,θ,ψ = 0.23, −0.17, 0.9) so the body↔NED rotation inside `gust_rates` is
genuinely exercised.

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| rigid rotation, body **x** | `p_g = +∂w/∂y` | air field `v = Ω_ned × (r−r₀)` at Ω = 0.031 rad/s about body x, built in NED with no flightsim function | `gust_rates → (0.031, 2e-18, −1.6e-19)`; error 6.9e-18 | OK |
| rigid rotation, body **y** | `q_g = −∂w/∂x` | same, about body y | `→ (9.6e-19, 0.031, −3.7e-19)`; error 1.4e-17 | OK |
| rigid rotation, body **z** | `r_g = +∂v/∂x` | same, about body z | `→ (−3.9e-20, −1.9e-19, 0.031)`; error 1.0e-17 | OK |
| cross-axis leakage | none | the two off-axis components of each of the three cases | max 2e-18 in all three | OK |
| `ω_rel = ω − ω_gust` | rotating with the air ⇒ still air | set `state.omega = Ω` in each case and compared the **moments** to still air at zero rate | `ω_rel` < 1.4e-17; `|M − M_still|` ≤ 9.9e-11 N·m for all three axes | OK |
| the three gradient relations | `p=+∂w/∂y`, `q=−∂w/∂x`, `r=+∂v/∂x` | body-linear fields with exactly **one** non-zero gradient each | all three signs and magnitudes exact to 7e-18 | OK |
| `sampled_rates` | same quantity, better estimator | vs `gust_rates` on all three rigid rotations, at `airframe.stations(747)` | agree to ≤2.2e-16 | OK |
| `strip_roll_moment` sign | `−w_g/V`, not `+w_g/V` | built the gust field *equivalent* to a rigid roll rate (`w_g = −p·y`) and compared the strip Cl to `strip_clp_from_rate` | **identical to 3.3e-18**; the wrong convention would differ by `2×|Cl|` = 0.0029 | OK |
| `strip_roll_moment` direction | updraught right ⇒ roll left | `w_g = −y` (air rising on the starboard wing) | `Cl = −0.0367` (rolls **left**) | OK |
| `strip_roll_moment` symmetry | uniform gust ⇒ no roll | uniform 5 m/s body-z gust | `Cl = 5.1e-19` | OK |
| `strip_clp_from_rate` **magnitude** | "must return `Clp·p_hat`" | rigid roll rate at the **shipped** `N_SPAN = 9` | returns **0.826435 × Clp·p_hat** | **C-F1** |

### 4. Wind-axis → body-axis force rotation

Tested on a 5×5 grid of α ∈ {−0.25,−0.08,0,0.06,0.19} × β ∈ {−0.30,−0.11,0,0.07,0.22} —
non-zero α **and** β simultaneously, and unequal.

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| `air_data` inversion | — | velocities constructed to have exactly the target (α,β); `air_data` must return them | worst error 1.2e-16 over the whole grid | OK |
| Drag direction | anti-parallel to `v_rel` | isolated drag channel; measured `cos(F, v_rel)` | **exactly −1.000000000000000**, max deviation 3.3e-16, at every (α,β) | OK |
| Drag magnitude | `q̄SC_D` | same | max relative error 2.1e-16 | OK |
| Lift direction | perpendicular to `v_rel` | isolated lift channel (induced drag killed with `e=1e12`, wave drag with `a=1e6`); measured `F̂·v̂` | **max \|cos\| = 4.6e-14** including at β = ±0.30 | OK |
| Lift magnitude | `q̄SC_L` | same | max relative error 4.2e-16 | OK |
| Lift plane | wind-axis −z lies in the body xz plane | body-y component of the isolated lift | `Fy/\|F\| = 1.0e-14` | OK |
| Lift direction sense | `C_L>0` acts up | `Fz` at α=0.06, β=0.22 | `Fz = −4.41e6 N` | OK |
| Full assembly | `F = R_bw(−D,0,−L) + (0,Y_body,0)` | `R_bw` written out independently in the script; compared over the whole grid | **max error / q̄S = 2.1e-16** | OK |
| Side force axis | "already body-axis" | isolated side-force channel | exactly `(0, q̄S·C_Y, 0)`; `F̂·v̂ = +0.218` at β=0.22, i.e. **not** perpendicular to the relative wind | **NOTE** |

**On the side force.** This is the one component that behaves differently at
β ≠ 0, and it is a **stated convention, not a defect**: `aero.py` says "Side
force is already body-axis, as the lateral derivatives are defined", and
CR-2144's `C_Yβ` is indeed a body-axis derivative. Measured consequence: the
force is not perpendicular to `v_rel`, so it does work along the flight path,
with `cos(F,v) = sin β`. The gap between this and the fully-wind-axis convention
`F = R_bw(−D, Y, −L)` is at most **0.0778 in C_Y units** over the tested grid,
and is **exactly zero at β = 0** (measured 9.3e-10 N). No component was found
that is *incorrect* at β ≠ 0.

### 5. Per-radian

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| Control derivatives | per radian | 1 rad vs `DEG2RAD` rad elevator, ratio of moment increments, all 4 aircraft | ratio = `0.017453292520` = `DEG2RAD` to 1e-12 in all 4 | OK |
| All 13 primary derivatives | per radian | plausibility bands (a per-degree leak is a factor 57.3 and cannot hide) | every derivative of every aircraft inside band; `CLa` 4.94/5.70/4.68/5.73, `Cmq` −23.9/−20.8/−7.37/−6.23 | OK |
| Cessna table fit | per-degree table, per-radian fit | independent `np.polyfit` on the degrees column, converted with `RAD2DEG` | table slope 0.10000/deg = 5.72958/rad; `ac.CLa` = 5.72958 | OK |
| Cessna `alpha_ref` | radians | recomputed independently; `at_reference` interpolation reproduced against a radians abscissa | `alpha_ref` = 0.02907 rad (1.666°); `Clb` reproduced to 1e-9 | OK |
| Control limits | radians | all 4 aircraft | 0.4363 rad = 25.0° | OK |
| `sweep` | radians | all 4 aircraft | 0.6545 rad = 37.5° (747), 0.0 (light aircraft) | OK |
| No degree leak into the plant | — | grep for `radians/degrees/deg2rad` in the non-test package | conversions confined to `aircraft.py` (definition boundary) and display code (`panel`, `viz`, `checks`, `analysis/series`, `apps/sweep`) | OK |

### 6. Non-dimensionalisation of rate derivatives

| Quantity | Claimed convention | How tested | Result | Status |
|---|---|---|---|---|
| `q_hat = q·c/(2V)` | **chord** | drove the 747 plant with a pure `q` at the exact CR-2144 condition and recovered the dimensional `Zq`, `Mq` of Table IX-4 | `Zq` −1.572770 vs −1.572768 m/s (**+0.0002%**); `Mq` −0.339000 vs −0.339000 (**−0.0000%**) | OK |
| — discrimination | | what the wrong reference length would give | span would give `Mq = −2.4290` (×7.17); dropping the ½ would give `−0.6780` | OK |
| `p_hat, r_hat = ·b/(2V)` | **span** | the Table IX-8 primed round trip (section 2 above) | all 8 primed values to machine precision | OK |
| `Zw`, `Mw` (α = w/U₀) | per radian | same finite-difference recovery, with `Mw` converted from 1/(ft·s) to 1/(m·s) | `Mw` **−0.000%**; `Zw` **−0.461%** (this residual *is* C-F2) | OK / C-F2 |
| Helper ↔ `aero.coefficients` consistency | same reference lengths both sides | 18 derivatives round-tripped through a scratch aircraft | **all ratios 1.000000** | OK |
| `airframe`/`wind` `p_hat` | `pb/2V`, matching `aero` | `strip_clp_from_rate` uses `dα = 2·p_hat·y/b`; checked by the equivalent-gust identity | signs and the `b/2V` scaling agree exactly | OK |

---

## C-F1 in full

**Claim under test.** Two docstrings assert exactness:

- `airframe.calibrated_lift_slope`: *"What it guarantees is that a rigid roll
  rate through the strip integral reproduces the tabulated Clp exactly."*
- `wind.strip_clp_from_rate`: *"this **must** return `Clp * p_hat`."*

**Measured.** At the shipped `airframe.N_SPAN = 9`:

| n_span | strip `Cl/p_hat` | sourced `Clp` | error |
|---|---|---|---|
| **9 (shipped)** | −0.289400 | −0.350179 | **−17.357%** |
| 11 | −0.305981 | | −12.622% |
| 21 | −0.334066 | | −4.601% |
| 51 | −0.346031 | | −1.185% |
| 101 | −0.348705 | | −0.421% |
| 501 | −0.350047 | | −0.038% |
| **2001 (test uses this)** | −0.350163 | | −0.005% |
| 20001 | −0.350179 | | −0.000% |

It is a pure quadrature error: the ratio is **0.826435 for all four aircraft**,
because it depends only on the chord shape and the station count, not the
airframe.

**Why it is not caught.** `flightsim/tests/test_wind.py::_b747_and_stations` has
signature `def _b747_and_stations(n_span=2001)`. The production path is
`loads.strip_model` → `airframe.stations(ac)` → `n_span = N_SPAN = 9`.

**The default shape is the worst case.** Separating shape from quadrature by
taking n → ∞:

| shape | ratio at n=9 | ratio at n=200001 (shape effect) | quadrature error at n=9 |
|---|---|---|---|
| **elliptic (default)** | 0.826435 | 1.000000 | **−17.357%** |
| uniform | 1.375000 | 1.333333 | +3.125% |
| tapered | 0.971154 | 0.974359 | −0.329% |

The elliptic distribution is the only one with infinite chord slope at the tips
— which `_strip_rolling_coefficient`'s own comment names as the reason for using
the trapezoid rule — and it is the one whose 9-point quadrature is 5× worse than
the alternatives. A side effect: `airframe.LOADING_SHAPES`, whose stated purpose
is that "the spread between them measures the shape assumption **and nothing
else**", carries at n=9 a quadrature spread of −17.4% to +3.1% on top of the
shape spread.

**It reaches flight.** Production path, 747 at cruise, spanwise-linear gust
`∂w/∂y = 0.05 s⁻¹` (a case where strip and equivalent-rate should agree exactly,
since a linear gradient is precisely what an equivalent rate represents):

- equivalent-rate `Cl` = +0.00220604
- `strip_model` `Cl` (N_SPAN=9) = **+0.00182315** → ratio **0.826435**
- strip at n=2001 = +0.00220593 → ratio 0.999953

Flown 10 s: final bank angle **+17.381°** (point model), **+28.450°**
(strip, n=9), **+30.640°** (strip, n=2001). The shipped station count
understates the strip path's own departure from the point model by **16.5%**.

**Sign is not affected.** Both paths agree in sign at every point tested; C-F1
is a magnitude finding only.

---

## What I tried and failed to break

Recording these because a passing check is only worth what the attempt to break
it was worth.

1. **Transposed DCM.** `|C − Cᵀ|max = 1.76` at the test attitude, so
   `quat_to_dcm` cannot be passing by accident of symmetry.
2. **Negated body rates in `quat_derivative`.** Produces error 6.25e-4 ≈ 2|ω|dt
   where the correct version gives 3.3e-12 — the convergence test has teeth.
3. **Negated `Ixz` in the inertia tensor.** Breaks the Table IX-8 primed round
   trip by +57.3% on `N'_p`. The 0.000% agreement of the real tensor is
   therefore meaningful.
4. **Wrong reference length in `q_hat`.** Using span would give `Mq = −2.429`
   against a tabulated −0.339; dropping the factor 2 would give −0.678. Neither
   is within a factor of 2 of the observed −0.339000.
5. **`+w_g/V` instead of `−w_g/V` in `strip_roll_moment`.** Would make the
   equivalent-gust and rigid-rate answers differ by 2×|Cl| = 0.0029; observed
   difference 3.3e-18.
6. **Symmetric cases that would hide a sign error.** Every control test used a
   single surface at a time; every gust test used one body axis at a time; every
   force-rotation test used non-zero α *and* β, unequal, of both signs.
7. **Per-degree leak.** Searched by band-checking 13 derivatives × 4 aircraft
   (a factor-57.3 error cannot stay inside any of those bands) and by the exact
   `DEG2RAD` ratio test. Nothing found.
8. **Inlined conversion factors.** Regex sweep over `flightsim/` and `scripts/`
   for 13 distinct factor literals. Only `_B747_G` found.

Two of my own test expectations were wrong and the code was right; both are
recorded above rather than quietly corrected — the `quat_derivative` convergence
order (3, not 2, because of normalisation) and the `Mw` unit conversion (the
228% "failure" was exactly `M2FT`, which is how I diagnosed my own error).

---

## Notes and bounded edge cases (not defects)

| Item | Observation | Bound |
|---|---|---|
| `aero.V_MIN = 1.0` clamp | Below 1 m/s, `β = asin(v/V_clamped)` is **not** the true sideslip, and the drag direction stops being exactly anti-parallel to `v_rel` | measured `cos(F,v)`: −1.00000000 at ≥1 m/s; −0.988 at 0.5 m/s; −0.958 at 0.05 m/s. Far below any flight speed; documented as a NaN guard |
| Unused `units.py` constants | `M2FT`, `FTS2MS`, `SLUG2KG`, `SLUG_FT3_TO_KG_M3`, `PER_DEG_TO_PER_RAD` have no non-test consumer | none — reported, not removed |
| `aircraft.stability_to_body` | defined and tested, but no registry aircraft calls it (Tables IX-4/IX-8 print "BODY AXIS SYSTEM") | none |
| `_boeing_747_approach` `V0` | `V0 = 165.0 * KT2MS / FT2M` is computed and never used | dead assignment only |
| `Cnda` sign spread | 747 cruise +0.00129, 747 approach +0.00640 (proverse), Cessna −0.0198 (adverse), Cherokee 0.0 | a data question, not a convention question — flagged for the data agent |
| `NOTATION.md` §8 | states the `_B747_G` slip as "2.4e-6" | I measured **1.51e-6** relative on the resulting mass |

## Untestable

| Item | Why |
|---|---|
| Cessna 172 rudder convention | `CYdr = Cldr = Cndr = 0` by an explicit, reasoned source decision. There is no rudder response, so the sign convention for `δr` on this aircraft cannot be exercised at all. Stated rather than guessed. |
| `δa` as a physical deflection (747) | Table IX-2's footnote defines `δa` as a **compound** control ("right inboard plus left inboard with the effect of outboard ailerons included"). The *sign* is testable and correct; the mapping from `δa` to any single physical surface angle is not defined by the source and I did not attempt to infer one. |
| CG / moment reference coincidence | The code takes the aerodynamic moment reference, the inertia reference and the accelerometer station to be one point. That is an assumption about the *data*, not a unit or sign convention, and cannot be settled from within the code. (Already recorded in `NOTATION.md` §6.) |

---

## Files

```
audit_evidence/
  C-units-conventions.md          this document
  scripts/
    c1_units.py                   conversion factors vs defining standards
    c2_quaternion.py              quat_to_dcm, quat_derivative
    c3_signs_flown.py             alpha, beta, elevator, aileron, rudder -- flown
    c4_ixz.py                     Ixz sign + CR-2144 primed round trip + _unprime
    c5_wind_axis.py               wind->body force rotation at alpha,beta != 0
    c6_gust_rates.py              rigid-rotation gust rates + strip_roll_moment sign
    c7_nondim.py                  per-radian and c/2V vs b/2V
    c8_dimensional.py             ISA, scaling laws, gravity, Coriolis, thrust
    c9_strip_stations.py          C-F1, characterised end to end
    c10_cn_vs_cl.py               C-F2, quantified
  logs/*.log                      captured output of each of the above
```
