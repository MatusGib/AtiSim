# Analysis and plausibility-check UI — design

**Status: IMPLEMENTED, session 15.** Written as a design against `a0d791d`
(2026-08-15) and built out in the same session. `atisim/checks.py`,
`atisim/analysis/{artifact,series,figures}.py`, `atisim/apps/sweep.py`,
`integrate.logged_rollout`, and `scripts/vortex.py --artifacts`. Suite **434
passed / 1 skipped**, up from 377.

**What building it changed in this document.** Five things, each marked in place
below with the evidence:

| § | Designed | Built |
|---|---|---|
| §7.2 | camera held by `uirevision` alone | `uirevision` **plus** an explicit `figures.apply_camera` from `relayoutData` — two browser probes of `uirevision` contradicted each other, so the app does not rely on a mechanism this project cannot assert on (§9.2.4) |
| §2.4 | isosurface on a cubic 32³ grid | **32 × 8 × 32** — the Parks vortex fixes `dpsi = 0`, so the cores are infinite east-west lines and a cubic grid spends 3/4 of its points on a direction the field is constant in. Cubic came to 2.29 MB against the stated 2 MB budget; this is 4× cheaper at the same resolution in the plane that matters |
| §3.2 C5 | `gate`, \|n_z[0] − cos θ₀\| < 1e-3 | **`report` in a wind field, `gate` in still air.** At 40 r₀ the 1/r far field still leaves 0.0384 g, so there is no trimmed start to demand (§3.2) |
| §2.1 P5 | α gate on the linear band | condemns **INVALID only**. `band == "linear"` failed the manoeuvring Fig. 8 point at 10.31°, which is amber and which `PROJECT.md` §4 publishes |
| §2.2 P9 | residual order-1 blamed on quadrature | the **trajectory's own order** — which became §0.5, and is now measured, falsified and recorded in `PROJECT.md` §4 |
| §2.4 | (unstated) where the field grid is centred | **on the STRUCTURE, from the field spec** — not the trajectory. See below; this one was a live bug. |

**The bug the UI found in the UI.** The first working 3D panel centred its field
grid on the trajectory's midpoint. For the canonical vortex run — 40 core radii
of lead-in — that midpoint is about −2476 m, roughly 2.5 km upstream of both
cores, where a Rankine vortex is **exactly irrotational**. So the vorticity
isosurface was computed over a region of zero vorticity and came out empty, and
the panel drew a flight path through nothing while looking entirely normal.
Caught by reading `isomin` off the live page — it was `0` where it should have
been `0.9 × 2V₀/r₀ = 0.255` — not by any test.

`vortex_viz._field_panel` had already written the rule down: *"Zoom to the
structure, not the run. The lead-in is deliberately long … and plotting all of it
would shrink the cores to invisibility."* The 3D panel is now given
`field_centre` from the artifact's field spec, `isomin` reads 0.255, the
isosurface peaks at the in-core 0.28333 s⁻¹, and the scene got **cheaper** (0.556
MB against 1.194) because the grid is tight around the structure instead of
spanning a 9.7 km run. A regression test pins it.

This is worth more than the fix: it is an instance of §4's own thesis — a plot
that looks plausible while the data behind it is empty — landing on the very
instrument built to catch it.

**Not built, and why:** the comparison driver (Galilean, strip-vs-point, and the
h/2 Richardson panel P11 all need a second run and belong to a driver that writes
its own artifact), the multi-run sweep table of §7.3, and artifacts from
`leewave.py`/`microburst.py` — the schema and `artifact.rebuild_field` already
cover both fields, so that is the same wiring as `vortex.py` and nothing yet
needs it.

**Provenance convention, the same one the rest of the project uses.** Every
number below carries where it came from:

| Tag | Meaning |
|---|---|
| **[M]** | **Measured this session** on this machine, against this commit. Scripts are listed in §8. |
| **[R]** | Recorded by the repository already — `PROJECT.md` §4 or `ASSUMPTIONS.md`. Not re-measured unless stated. |
| **[C]** | Cited to an external source, named inline. |

An unmarked number is arithmetic on marked ones. There are no unmarked
performance claims.

---

## 0. Summary, and the five things measurement changed

The recommendation is **Dash + Plotly**, with the framework confined to layout
and callback wiring and every computed quantity living in `atisim/`. That
follows the repo's own established protocol rather than inventing one: the
notebook already holds no arithmetic, and session 13 exists entirely because
session 12 broke that rule.

Five results contradicted what the investigation brief assumed — or, in the last
case, what I assumed — and each one changes a recommendation:

1. **Data volume is not the constraint. Browser payload is.** The largest
   analysis run is 31,791 samples ≈ 6.4 MB of float64 **[M]**. Sampling the
   wind field on a 64³ grid takes 0.6–2.8 ms jitted **[M]**. But twelve
   full-rate `Scattergl` traces serialise to **8.5 MB of JSON in 58.6 ms**
   **[M]**, and a 48³ Plotly `Isosurface` is **5.0 MB** **[M]**. The engine can
   feed the UI far faster than the UI can ship a frame. Every performance
   decision below follows from that inversion.

2. **The h vs h/2 overlay does not find the wind seam.** The brief calls it
   "the most likely place for a silent error" and asks for a dedicated view.
   Measured on the canonical vortex run, the cumulative position difference
   peaks at **0.402 m at the very end of the run**, 23.7× larger than anything
   inside the first core **[M]**. It is accumulated phugoid phase drift, not a
   seam signal — the actual per-step local truncation error is **5.8×10⁻⁶ m**
   **[M]**, seventy thousand times smaller. The overlay would show a smooth
   growing curve and localise nothing. §3 replaces it with two instruments that
   do work.

3. **The divergence check and the `omega_gust`-vs-finite-difference check have
   no teeth.** All four fields are solenoidal to round-off — max |∇·w| is
   between 0 and 1.4×10⁻¹⁷ s⁻¹ **[M]**. And `gust_rates` *is* `jax.jacfwd` of
   the field, so comparing it against a finite difference of the same field
   agrees to 5×10⁻⁹ at h = 1 m **[M]** and is very nearly a tautology. Both are
   worth keeping as regression tripwires for *future* fields; neither is a
   diagnostic. The check that does bite is `gust_rates` vs `sampled_rates`, and
   the energy-closure residual (below).

4. **Something unexpected localises the field's one genuine non-smoothness.**
   The per-step energy-closure residual spikes to **22–354× the run median at
   each of the four Rankine core-boundary crossings** **[M]**. That is the
   `ASSUMPTIONS.md` §E2 gradient discontinuity showing up as an energy
   artefact, and it is the sharpest positional diagnostic found. It gets a
   panel.

5. **The integrator is first order, not fourth, in a spatially varying wind
   field.** Chased down because the energy residual converged at order 1.00 and
   the explanation I first wrote for it was wrong. Measured order of the
   trajectory itself, against a dt = 1/2048 reference over a 4 s window
   **[M]**:

   | Field | Smoothness | Fitted order |
   |---|---|---|
   | Lee wave, 1.2 km, 25 m/s | C^∞ | **1.015** |
   | Gaussian updraft column, 400 m | C^∞ | **1.015** |
   | Single Parks core, flown entirely outside it | C^∞ (1/r² far field) | **1.030** |
   | …the same, further out | C^∞ | **1.028** |
   | Single Parks core, flown **through** it | C⁰ at r = r₀ | **0.624, non-monotonic** |

   Four independent smooth configurations give a clean, uniform order of
   **1.01–1.03** where the still-air figure is **3.989** **[R]**. The cause is
   the once-per-step wind hold, which `integrate.py` documents as deliberate
   (*"the standard treatment for Dryden and von Karman"*). Crossing the Rankine
   core edge adds a grid-dependent error on top that destroys even first-order
   convergence.

   **This is not a defect and it is not a UI finding — but it was not
   previously measured, and it changes two panels.** `ASSUMPTIONS.md` §E4 says
   in as many words that *"the order-of-accuracy test flies at fixed controls in
   still air, so it cannot see a wind term evaluated at the wrong stage"*;
   session 12 closed the **body-force** half of E4 and the **order** half was
   left open. This appears to be the first measurement of it. Consequences are
   in §9.2.1; the panel consequences are that P11's expected slope is **1, not
   4**, and that P9's residual order of 1.00 is the trajectory faithfully
   reporting its own order rather than a quadrature artefact.

**Two costs stated up front.** The design adds `pyarrow` and `dash`+`plotly` to
a project whose entire dependency list is four packages, roughly tripling
install weight; and it asks for a new `atisim/checks.py` plus a run-artifact
writer, which is real engine work before any pixel is drawn. The alternative —
a UI that computes its own numbers — is cheaper and is the failure mode §3
argues hardest against.

---

## 1. Recommended stack

### 1.1 What the criteria actually are here

Feature lists are useless because the volumes are small. The criteria that
discriminate:

| Criterion | Why it discriminates here |
|---|---|
| Payload per interaction | The measured binding constraint (§0.1). |
| 3D camera survival across re-render | The linked time cursor fires a figure update on every click. If the camera resets, the whole deep-dive design fails. |
| Auditability of the update path | This repo's culture is that every number names its source. A magic reactive layer is a place for a number to appear without one. |
| Notebook reuse | Required. Solved by architecture, not framework — see §1.3. |
| Dependency weight | Four packages today. Every addition is a real cost. |
| Cold start | JAX import + trim is ~1 s; a first `rollout` compile is 0.6–0.9 s **[M]**. |

### 1.2 Comparison

| | Dash + Plotly | Panel + HoloViews | Streamlit | Bokeh server | marimo | PyVista/trame |
|---|---|---|---|---|---|---|
| 3D | Plotly `Scatter3d`/`Cone`/`Streamtube`/`Isosurface`, WebGL | none native; embeds a Plotly or VTK pane | embeds Plotly | **none** | embeds Plotly | **best available** — real VTK |
| Camera across re-render | `uirevision` held constant, one line **[C]** | inherits whichever pane | figure rebuilt on rerun | n/a | pane-dependent | native, server-side state |
| Linked panels | explicit callbacks; verbose but auditable | `hv.link_selections`, no callbacks **[C]** | rerun-everything | explicit | reactive-cell | hand-built |
| Panel-count scaling | one callback, N outputs — flat for a shared cursor | flat | **degrades badly**: whole script reruns | flat | flat | n/a |
| Notebook reuse | `app.run(jupyter_mode=…)`, or reuse the figure functions | components render natively | poor | moderate | native | native |
| Deps added | dash, plotly, flask (3) | panel, holoviews, hvplot, bokeh, param (5) **and still plotly for 3D** | streamlit + deps | bokeh, tornado | marimo | pyvista, vtk, trame (~200 MB) |
| Local launch | one command, hot reload | one command | one command | one command | one command | one command |

### 1.3 Recommendation, and the architecture that makes the choice cheap

**Dash + Plotly**, subject to one architectural rule that matters more than the
framework choice:

> The framework imports `atisim`. `atisim` never imports the framework.
> Every quantity the UI displays is computed by a function in `atisim/` that
> is asserted by a test.

This is not a new rule. `ASSUMPTIONS.md` states it verbatim for the notebook —
*"add the computation to `verification.py`, assert it in a test, then add a
notebook cell. Never the other way round"* — and `vortex_viz.figure()` already
returns a `Figure` and *"never shows and never saves"*. The UI is the third
front end over the same tested core, after the test suite and the notebook.

With that rule in place, the notebook-reuse requirement is satisfied by
importing the figure functions, not by the framework. That removes the single
strongest argument for Panel, and the decision falls to auditability and
dependency weight, where Dash wins on both.

**Rejected, with the reason each:**

- **Panel + HoloViews** — runner-up, and genuinely better at one thing:
  `hv.link_selections` cross-filters N panels with no hand-written callbacks
  **[C]**. Rejected because this UI has *one* shared control (the time cursor),
  which in Dash is one callback with N outputs, so the advantage does not
  materialise at this scale; because it needs five packages and still needs
  Plotly for 3D; and because `link_selections` does not work with `DynamicMap`
  **[C]**, which is where a field-resampling panel would naturally live.
  **Revisit if** panels ever need *independent* brushing rather than a shared
  cursor.
- **Streamlit** — rejected outright. Its execution model reruns the script on
  every interaction. With a JAX import, a trim solve, and a 0.5–8.5 MB figure
  set **[M]**, every cursor move pays for all of it. Caching helps and does not
  fix the model.
- **Bokeh server alone** — rejected: no 3D. The brief makes 3D essential.
- **marimo** — rejected, not dismissed. The reactive-DAG model is a genuinely
  good fit for "change the window, everything downstream updates," and its
  notebook/app duality is the cleanest of any candidate. Rejected because the
  3D path is still an embedded Plotly figure (so no 3D advantage) while the
  ecosystem and debugging story are younger than Dash's. Nothing in this design
  is marimo-hostile; the figure functions would port unchanged.
- **PyVista/vedo/K3D/ipygany as the shell** — rejected as the *shell*, retained
  as an escape hatch for the 3D pane (§2.4). PyVista is much better at 3D:
  marching cubes runs server-side, so an isosurface ships triangles rather than
  the scalar volume. Rejected as the shell because the 2D strip stack — which
  is where most diagnostic value lives — would be hand-built, and because
  vtk + pyvista + trame is a ~200 MB dependency for a project currently at four
  packages. ipygany is effectively unmaintained; K3D and vedo are both narrower
  than PyVista with no compensating advantage here.

### 1.4 The measured payload budget this buys

Everything below is **[M]**, Plotly 6.9.0, JSON serialisation of the figure.

| Scene | Size | `to_json` |
|---|---|---|
| `Scatter3d` trajectory, 4,018 pts | 0.186 MB | 1.9 ms |
| `Scatter3d` trajectory, 31,791 pts | 1.424 MB | 10.9 ms |
| `Cone` glyph field, 24³ = 13,824 glyphs | 0.941 MB | 5.3 ms |
| `Streamtube`, 24×5×24 grid, **any** seed count 20–120 | **0.20 MB** | 2.4 ms |
| `Isosurface`, 32³ | 1.465 MB | 11.7 ms |
| `Isosurface`, 48³ | **5.002 MB** | 27.2 ms |
| 12 × `Scattergl`, 2,000 pts each | 0.542 MB | 5.0 ms |
| 12 × `Scattergl`, 4,018 pts each | 1.081 MB | 7.9 ms |
| 12 × `Scattergl`, 31,791 pts each | **8.499 MB** | 58.6 ms |

**The budget: 2 MB per callback.** That yields, concretely: 2,000 points per 2D
strip, a 32³ isosurface *or* a streamtube of any complexity, and a full-rate 3D
trajectory up to ~4,000 points. Two observations worth carrying:

- **`Streamtube` is nearly free and independent of seed count** — the payload is
  the vector grid, and the integration happens client-side. It is the best value
  in the whole table.
- **`Isosurface` is the opposite** — marching cubes runs client-side, so the
  entire scalar volume ships. 32³ is the practical ceiling in Plotly. If a
  finer core boundary is ever needed, that is the trigger to swap the 3D pane
  for PyVista (which meshes server-side), and it is a pane swap, not a rewrite.

---

## 2. Plot inventory

Pruned rather than accumulated. Each entry states the error it detects; if it
detects nothing specific it is not here.

### 2.1 Tier 1 — the sweep view

Sign conventions throughout, from `state.py`: NED inertial, body x-forward /
y-right / z-down, altitude = `−pos_ned[2]`, quaternion `[w,x,y,z]` body→NED.
Vertical gust is plotted **positive up**, which is `−wind_ned[2]`.

---

**P1. Causal strip stack** — reuse `vortex_viz._trace_stack` essentially as-is.

Five strips, shared x = time (s), analysis window shaded:
`w_up` (m/s, up positive) with `q_gust` (deg/s) on a twin axis → `α_air` and
`α_inertial` (deg) overlaid → `n_z` (g) with the trim reference line →
`θ` (deg) with `q` (deg/s) → `elevator` (deg).

*Correct looks like:* gust in at the top, response propagating downward, and
`elevator` **flat** for any `fly()` encounter — the fixed controls are a
physical statement, not a convenience.

*Detects:* (a) **air-relative/inertial confusion** — the two α traces separate
by up to 7.0° in a Parks encounter **[R]**; if they coincide under wind, the
wind is not reaching the sensing path. (b) **Causal-order violations** — a
response leading its input. (c) A non-flat elevator on a turbulence run, which
means the discriminator's premise has been broken.

*Known failure mode:* both α traces identical under a non-zero gust — the
`viz.derived` bug of §6(b), which lived three sessions because still-air tests
cannot see it.

---

**P2. Load factor `n_z(t)`** — g, body-normal, with peak annotation and a
horizontal line at `n_z[0]`.

*Correct looks like:* `n_z[0] = cos θ₀`, not 1.0 — for the 747 at cruise that is
0.9967 **[R]**. The excursion is measured from `n_z[0]`, not from 1.

*Detects:* (a) **an inadequate lead-in.** A −6 r₀ lead-in starts the aircraft
0.20 g out of equilibrium and understates first-core Δθ by 15% **[R]**; the
`n_z[0]` line makes that visible without arithmetic. (b) A sign error in
`specific_force`'s z-negation.

*Countermeasure it needs:* the y-axis must **not** autoscale to the excursion,
or a 0.02 g offset at t = 0 fills the panel and looks like a defect. Pin to
`[min(n_z) − 0.1, max(1.05, max(n_z))]`.

---

**P3. `n_z` against α, scatter** — reuse `vortex_viz._load_vs_alpha_panel`.

x: α (deg), both air-relative (coloured by time) and inertial (grey). y: `n_z`
(g).

*This panel is an assertion, not a display.* The air-relative points must fall
on a straight line; the inertial ones must scatter. Recorded correlations:
**0.9990 air-relative, 0.5572 inertial** **[R]**.

*Detects:* any wind-sensing error, immediately and unambiguously — a scattered
air-relative cloud means the wind used for sensing is not the wind that was
flown.

*Known failure mode, and it is permanent:* the line is straight *because*
`aero.py` is `CL = CL0 + CLa·α` with no stall. A real aircraft's curve bends at
buffet onset. The panel title must say so — the existing one does.

---

**P4. Wingrove & Bach Fig. 8 discriminator** — reuse
`vortex_viz._discriminator_panel`.

x: pitch attitude excursion (deg) in the window. y: load excursion from trim
(g). Filled marker = windowed; hollow marker = whole run; dotted connector
between them; grey reference band and the paper's three category positions.

*This is the domain-standard plot* and the closest thing the project has to
one. The connector is the important part: it draws the windowing trap
explicitly. Measured, the vortex moves 2.24° → 8.33° between the two windows
while the manoeuvre moves 30.37° → 30.74° **[R]** — the asymmetry is the whole
argument for the window rule.

*Detects:* window-selection error, which is the largest single source of a
wrong Fig. 8 coordinate.

*What it must never become:* a claim of agreement. `PROJECT.md` §5 forbids it —
Wingrove & Bach never identifies an aircraft type. The x-axis must scale to the
data, not to the paper's range, or a cropped chart hides the model's manoeuvring
point sitting 2.5× to the right.

---

**P5. α validity gate** — α_air (deg) against time, with the declared bands
shaded: green to ±10°, amber to ±12°, red beyond
(`panel.ALPHA_LINEAR_DEG = 10.0`, `ALPHA_INVALID_DEG = 12.0`).

*Detects:* a run that has left the linear-aero range and therefore proves
nothing.

*Measured, and it forces a design decision.* On the canonical vortex run,
|α| peaks at **11.39° over the whole run** — amber — but only **8.27° inside
the first core**, which is green, and just **1.34% of the run is outside the
linear band** **[M]**. So a whole-run gate would flag amber on a run whose
reported result is entirely valid. **The gate must be window-aware and must
report both numbers**, exactly as the Fig. 8 panel reports both windows. This
is the same trap in a different channel.

*Bands are symmetric on purpose:* `aero.py` is odd-symmetric in Δα, so a
pushdown is as far out of range as an equal pull-up. That symmetry is what
latent bug (e) got wrong **[R]**.

---

**P6. F-factor** — for microburst and lee-wave runs only.

x: along-track distance (km, 0 = field axis). y: F, dimensionless, **positive
is hazardous**. Instantaneous F as a thin grey line; the **1 km average as a
thick line** (Proctor et al. Eq. 7); the thrust-authority band `(idle, full)`
shaded; the FAA 0.10/0.13 thresholds drawn **only with an aircraft-class
label**.

*Detects:* (a) **quoting the instantaneous peak instead of the averaged
metric** — measured 21% overstatement in the Cherokee microburst case (+0.2326
vs +0.1929) **[R]**. Drawing both, with the average heavier, makes the right
one the default read. (b) Applying jet-transport thresholds to a piston
aircraft — `scripts/microburst.py` already decides this per aircraft and the
panel must inherit that decision, not re-make it.

*Correct looks like:* the averaged curve strictly inside the instantaneous
envelope, and masked where the 1 km window would run off the end of the run —
`dynamics.average_f_factor` returns that `valid` mask and it must be honoured.
Plotting the clamped tail understates a rising hazard.

---

**P7. 3D: trajectory through the field.** Detailed in §2.4.

---

### 2.2 Tier 2 — deep-dive panels

**P8. Recovered wind vs analytic field.**

x: time. y: `wind_ned` components (m/s), recorded vs re-evaluated at the
recorded position.

*Detects:* whether the artifact's wind is the wind that was flown.

*Measured, and it is a trap that will be walked into.* `SimState.wind_ned`
caches what the **previous** step applied. So the recorded wind differs from the
analytic field at the recorded position by up to **0.3315 m/s, 1.13% of the peak
gust** — and after shifting by exactly one step the two agree to **7.1×10⁻¹⁵
m/s** **[M]**. A panel that overlays them unshifted shows a real, correct,
systematic lag that looks exactly like a bug. **The panel must plot the shifted
comparison as the primary trace and the unshifted residual as a separate
diagnostic, labelled "one-step cache lag, expected".**

---

**P9. Energy budget and its residual.** Two traces plus one scatter.

Definitions: `E = ½m|v_ned|² + mgh`. Power of the non-gravitational forces
`P = F_body · v_body`, with `F_body` recovered through
`dynamics.specific_force`, which inverts the derivative sum rather than
recomputing it. Closure residual `= (E − E₀) − ∫P dt`.

*Correct looks like:* **relative residual 8.8×10⁻⁴ at dt = 0.01** **[M]**, and
it halves as dt halves — measured order **1.00, 0.99** over dt = 0.02 → 0.005
**[M]**, and **1.015, 1.008, 1.004** over dt = 0.04 → 0.005 on a short window
**[M]**.

**That first-order convergence is the finding of §0.5, and I got its cause
wrong on the first pass.** The draft of this section attributed it to
trapezoidal quadrature crossing the core-boundary kink. It is not that: the
*trajectory itself* is first order in any spatially varying wind field
(measured 1.015–1.030 across four smooth configurations **[M]**), so the closure
residual is reporting the trajectory's own order faithfully. The panel should
therefore state **expected order 1 in a wind field, 4 in still air**, and a
measured order near 4 in a wind field would mean the closure was wrong, not
right. Recorded rather than quietly corrected, per `PROJECT.md` §4's rule.

*The second trace is the one that earns the panel.* Plot **per-step |Δresidual|
against along-track position**, log y. Measured: the run median is 12.7 J, and
within one step of the four Rankine core-boundary crossings it reaches
**282 / 3,054 / 3,721 / 4,493 J — 22×, 241×, 293× and 354× the median**
**[M]**, with the run peak of 22.1 kJ just inside core 2. That is
`ASSUMPTIONS.md` §E2's gradient discontinuity, located to within one step, from
an energy diagnostic. **This is the panel that does the job §0.2 says the h/2
overlay cannot.**

*Third element:* the wind's share of the power, `P_gust − P_still`. Measured
peak **6.36×10⁷ W, 76.6% of peak total power**, with a net **2.04×10⁸ J**
delivered over the run against 8.04×10⁹ J of trim kinetic energy, i.e. 2.5%
**[M]**.

---

**P10. Point-vs-strip gust rate.**

x: along-track position in units of r₀. y: `q_gust` from `wind.gust_rates`
(tangent at the CG) and from `wind.sampled_rates` (secant across the airframe),
plus their difference normalised by `V₀/r₀`.

*Detects:* whether the encounter is in the regime where the point model is
exact — which is the standing `ASSUMPTIONS.md` §E2 caveat on the project's
headline result.

*Measured on the two-core Hannibal array as `scripts/vortex.py` builds it*
**[M]**:

| station | correction ÷ (V₀/r₀) |
|---|---|
| 0.50 r₀ | 0.0012 |
| 0.99 r₀ | 0.0016 |
| **1.00 r₀** | **2.0016** — and the two estimators have **opposite signs** |
| 1.10 r₀ | 0.7488 |
| 1.25 r₀ | 0.1068 |
| 2.00 r₀ | 0.0219 |
| 3.00 r₀ | 0.0004 |

*Read the in-core value carefully.* `ASSUMPTIONS.md` §E2 records **exactly
0.0000** at 0.50 r₀ for a **single** core; this measurement gives 0.0012 because
the second core's 1/r far field is curved across the airframe at that station.
The difference is small and it is real, and it is a good illustration of a rule
the panel must follow: **a diagnostic that does not state its configuration is
not reproducible.** The panel footer carries the core count.

---

**P11. Step-size convergence — per-step Richardson restart, not a trajectory
overlay.**

From each recorded state, take one step at h and two at h/2 **from that same
state**, and plot `|Δpos|` against along-track position, log y.

*Why not the obvious thing.* Measured on the canonical vortex run **[M]**:

| Instrument | Peak | Where | In-core | Ratio |
|---|---|---|---|---|
| cumulative \|pos(h) − pos(h/2)\| | 0.402 m | **end of run, t = 40.18 s** | 0.0169 m | end 23.7× in-core |
| per-step growth of that difference | 7.6×10⁻⁴ m | t = 38.90 s | 1.3×10⁻⁴ m | in-core 0.165× peak |
| **per-step Richardson local error** | **5.77×10⁻⁶ m** | far field / in-core | 5.69×10⁻⁶ m | **peak/median 1150×** |

The cumulative overlay is dominated by accumulated phugoid phase drift and
peaks nowhere near the field. Even its per-step growth peaks at t = 38.9 s. Only
the restart estimate isolates local truncation, and it has a 1150× dynamic
range across the run, which is what makes a log-y panel readable.

*Cost, stated with the recommendation:* **4.7 ms warm for all 4,018 samples**
after a 1.7 s first compile **[M]**. Cheap enough to write into the artifact for
every run.

*The reference slope must be 1, not 4.* This is the direct panel consequence of
§0.5. In still air the expected slope is 4 (measured 3.989 **[R]**); **in any
spatially varying wind field it is 1** (measured 1.015–1.030 across four smooth
configurations **[M]**). A panel that draws a slope-4 reference line against a
wind run would show every correct run failing. The reference line is therefore
selected by *whether the run had a wind model*, and the panel says which.

*And it must expect non-convergence across a core traverse.* Over a window
containing a Rankine boundary crossing the error is **non-monotonic** — measured
pairwise orders +3.46, −1.06, +0.04 over dt = 1/16 → 1/128 **[M]**. That is
correct behaviour for RK4 across a C⁰ right-hand side, and a panel that fits a
single slope through it reports a meaningless number. **The panel plots the
points and the reference line and refuses to fit a slope when the sequence is
non-monotonic**, which is the same discipline `verification.fitted_order`
already applies when it raises on a saturated sequence.

*Honest limitation.* Even the restart estimate does **not** peak at the core
edges — it reads ~5×10⁻⁷ m there, roughly 10× *below* the run peak **[M]**. So
P11 establishes that the step size is adequate; it does not localise the field's
non-smoothness. **P9 does that.** They answer different questions and the brief
conflated them into one "seam" view.

---

**P12. Lateral symmetry check.**

y: β (deg), p (deg/s), φ (deg) against time, on a **fixed** axis pinned to a
declared tolerance.

*Detects:* spurious lateral response. This is stronger than it looks. The Parks
vortex has **no east variation** and the updraft column is **axisymmetric about
an axis the aircraft flies straight through**, so in both cases every strip on
the span sees the same vertical gust and the antisymmetric roll integral cancels
— which is why flying the strip path moves the vortex result by **0.000000 m**
**[R]**. So for both canonical turbulence encounters the lateral response should
be **identically zero**, and any non-zero value is a defect.

*Do not autoscale this panel.* An autoscaled plot of 10⁻¹⁴ rad/s of round-off
manufactures a dramatic-looking oscillation out of nothing. Fixed axis, and a
badge carrying the number.

---

### 2.3 Deliberately pruned

| Candidate | Why not |
|---|---|
| **PSD of gust input and response vs Dryden/von Kármán** | **There is no Dryden layer** (`PROJECT.md` §7 step 4, blocked on digitising MIL-F-8785C Fig. 7 and on `init_sim` seeding). Every field today is deterministic and either single-event or periodic, so its "spectrum" is the Fourier transform of a known analytic function — it tests nothing, and windowing a transient manufactures spectral structure that is a artefact of the window. **Build this the session Dryden lands, not before.** It is the highest-value future panel and the wrong panel today. |
| **Pole migration / mode phase-planes** | Keep, but on a separate *model* screen, never beside an encounter. `validation.longitudinal_modes` and `lateral_modes` already compute them and the notebook already shows the sweeps. Placing a pole plot next to an encounter time history invites attributing an encounter feature to a mode — which is precisely the error the whole-run Δθ of 8.33° represents: that number **is** the phugoid, not the encounter **[R]**. |
| **Quaternion-norm time history** | Measured max \|1 − ‖q‖\| = **2.22×10⁻¹⁶ over 4,018 steps** **[M]**. Plotting machine epsilon manufactures structure. It gets a **badge with a number**, not a panel. |
| **Divergence field render** | Measured 0 to 1.4×10⁻¹⁷ s⁻¹ across all four fields **[M]**. Same reasoning. Badge. |
| **Airspeed as its own panel** | Subsumed: it belongs on the F-factor screen (F goes as 1/Vₐ, which is why flown peak F exceeds the naive w₀/V **[R]**) and nowhere else. |
| **Ground track (`viz.post_flight`)** | Keep for hand-flown runs from `fly.py`; it carries no diagnostic content for a straight-line fixed-control encounter. |

### 2.4 The 3D panel, and why one treatment does not fit four fields

This is the part where the brief's instinct — "trajectory tube inside the vortex
structure" — needs splitting by field, and the measurements say how.

Sampling cost is not a consideration: a 64³ grid costs **0.56 ms (vortex) to
2.84 ms (microburst/updraft)** jitted **[M]**, so the field can be resampled on
every callback. Payload is the only consideration.

| Field | Structure | Representation | Why this one | Payload **[M]** |
|---|---|---|---|---|
| **Parks vortex array** | 2D — no east variation. Rankine: vorticity is **exactly 0 outside and exactly 2V₀/r₀ inside** (measured max \|curl\| = 0.283333 s⁻¹ against 2V₀/r₀ = 0.283333 **[M]**) | **Vorticity isosurface at 0.9 × 2V₀/r₀**, 32³, plus **one vertical slice plane** coloured by w_up | The core boundary *is* the physical claim, and an isosurface draws exactly it. Because the field has no east variation, a single slice loses nothing — and the existing `_field_panel` is already that slice. | isosurface 32³ **1.465 MB**; slice ~0.1 MB |
| **Updraft column** | axisymmetric, smooth, 39.6 spans **[R]** | **Streamtube** seeded on a lateral line, plus a translucent cylinder at the w₀/e radius | Vorticity is a smooth blob with no edge to draw. What matters is the *radius* and the *edge sharpness* — which is the DECLARED parameter — so draw the radius explicitly. | **0.20 MB** |
| **Lee wave** | 1D periodic in north, purely vertical | **Slice plane only. No 3D structure.** | It genuinely has none. Rendering a 1D field as a volume manufactures structure — the exact failure §4 warns about. Show it as the 2D field it is, in the 3D scene, and let the emptiness be the message. | ~0.1 MB |
| **Microburst** | axisymmetric stagnation flow with a **ground plane** | **Streamtube seeded on the axis**, plus an opaque ground plane and the one-wingspan clearance slab | The topology — descend, turn, run out radially, go to zero at the ground — is the paper's whole point, and only streamlines show a stagnation topology. | **0.20 MB** |

**Common to all four:** the trajectory as `Scatter3d`, `mode="lines"`,
colour-mapped by a **selectable scalar** (`n_z`, α_air, F-factor, energy
residual), with a marker at the time cursor. At 4,018 points that is 0.186 MB
**[M]**; runs longer than ~4,000 samples get min/max-envelope decimated per §4.

**Camera:** `layout.uirevision` set to a constant string and never changed by a
callback **[C]**. This is the single line that decides whether the linked-cursor
design works. It is also the first thing the vertical slice (§7) must prove.

**Axes:** all three in metres with `scene.aspectmode="data"`. A wrong core radius
is only visible if the cores are drawn round — the existing 2D `_field_panel`
already makes this argument for `set_aspect("equal")` and it carries over.

---

## 3. Plausibility checks

### 3.1 Where each is computed, and the defence

**Recommendation: in the engine, asserted by a test, written into the run
artifact. The UI renders and never computes.**

The defence is the repo's own history, not a preference. `ASSUMPTIONS.md`
states the protocol; session 12 put two checks inside a test file instead of in
`verification.py`; the consequence was that **the notebook told a reader the
`−m·dW/dt` term could not be detected** when it demonstrably could, and session
13 existed to fix it. A UI that computes its own numbers reintroduces exactly
that drift, with a worse blast radius, because a figure is more persuasive than
a print.

Concretely: a new **`atisim/checks.py`**, tier-0 in the same sense as
`verification.py`, each check a function returning a typed result, each asserted
by a test in `atisim/tests/test_checks.py`, each serialised into the
artifact.

**The stated exception.** Three checks need a *second run* and therefore cannot
be written by the run that produced the artifact:

- the h/2 Richardson restart (P11) — actually per-sample and cheap, so this one
  **can** go in the primary artifact: 4.7 ms warm **[M]**;
- the Galilean pair (still air vs uniform wind, offset start);
- strip vs point loads.

These produce a separate **comparison artifact** written by a comparison driver.
**The UI must not launch a simulation.** A callback that silently re-flies pays
a 0.6–0.9 s JAX compile **[M]** and makes the panel's contents depend on the
machine — and `n_steps` is a `static_argname`, so *every distinct dt recompiles*.

### 3.2 The checks, with pass criteria and measured current values

| # | Check | Computed by | Pass criterion | Current value **[M]** | Does it bite? |
|---|---|---|---|---|---|
| C1 | Quaternion norm drift | `checks` | max\|1−‖q‖‖ < 1e-12 | 2.22e-16 | **No.** Regression tripwire only. |
| C2 | Field divergence, max over a grid, normalised by the field's own max strain rate | `checks` | < 1e-12 | 0 – 2.4e-16 | **No** for these four fields. **Yes** for the next field added — which is the point. |
| C3 | Energy closure residual ÷ \|ΔE\|max | `checks` | < 2e-3 at dt = 0.01, **and halving with dt — order 1 in a wind field, not 4** (§0.5) | 8.8e-4; order 1.00 | **Yes** — see C4. |
| C4 | **Per-step energy residual against position** | `checks` | reported, not gated | 22–354× median at the four core crossings | **Yes. The sharpest positional diagnostic found.** |
| C5 | Trim residual and trimmed-start check | `trim` (exists) + `checks` | ‖r‖ < 1e-9 **and** `trim.is_physical`, **and** \|n_z[0] − cos θ₀\| < 1e-3 | residual 1.9e-20 | **Yes** — catches a short lead-in, which is a real historical defect **[R]**. |
| C6 | Galilean invariance | comparison driver | \|Δq\|, \|Δω\| < 1e-11; \|Δpos − W·t\| < 1e-6 m | 0, 0, 2.6e-13 m | **No**, and its blindness is *measured* — it passes with the `−m·dW/dt` bug injected **[R]**. Surface it **with that caveat printed**, or it misleads. |
| C7 | Step-size Richardson, per sample | `checks` | peak < 1e-4 m; report peak/median; **reference slope 1 with a wind model, 4 without** | 5.8e-6 m; 1150× | **Yes**, as an adequacy statement. |
| C8 | α validity band, **windowed and whole-run separately** | `checks` | window \|α\| < 10° green / 12° amber | window 8.27°, run 11.39° | **Yes.** |
| C9 | Lateral symmetry on a symmetric field | `checks` | max\|β\|, max\|p\| < 1e-10 | expected exactly 0 **[R]** | **Yes** for any new field. |
| C10 | Point vs strip gust rate along the track | `checks` | reported, not gated | table in P10 | **Yes** — it is the live §E2 caveat. |
| C11 | Recorded wind vs analytic, **one-step-shifted** | `checks` | < 1e-12 m/s shifted | 7.1e-15 shifted, 0.33 unshifted | **Yes** the moment a stochastic model lands. |

**Four of these currently have no teeth (C1, C2, C6 and, on today's fields,
part of C9), and the panel must say so rather than showing four green ticks that
mean nothing.** A badge reading "0 / tolerance 1e-12" alongside "this check has
never failed and is a tripwire for future fields" is honest; a green tick is
not. This is the same argument `PROJECT.md` makes about falsification — *"a
check which can only pass shows nothing"* — rendered into the UI.

### 3.3 Badge design

Each badge: **name · measured value · tolerance · verdict**, and a class of
`gate` (fails the run) or `tripwire` (recorded, never yet non-zero) or
`report` (a number with no threshold, like C4 and C10). Clicking a badge sets
the time cursor to the sample where the check is worst and scrolls the relevant
panel into view — borrowed from FDM exceedance workflows (§6).

---

## 4. How plots lie, and the countermeasure for each

| # | The lie | Measured magnitude here | Countermeasure |
|---|---|---|---|
| 1 | **Autoscaling hides slow drift** | A 0.02 g trim offset fills an autoscaled `n_z` panel | Axis ranges are a **declared property of each panel**, pinned across runs and across the whole time base. Borrowed directly from ParaView, which makes "rescale to data range over all timesteps" vs "current timestep" an explicit button precisely because the automatic mode is *"extremely misleading… with absolutely no indication that the average field value quintupled"* **[C]**. Any panel whose range was auto-derived carries a visible marker. |
| 2 | **Decimation destroys peaks** | **Measured [M]:** stride-decimating the vortex run from 4,018 to 250 points reports −1.164 g instead of the true **−1.235 g** — a **5.72% error on the project's headline number**. **Min/max-envelope decimation loses 0.0000% at every level tested (2000, 1000, 500, 250, 100 points).** | **Never stride-decimate. Always min/max envelope** — keep both extremes of each bucket. It is measurably free. |
| 2b | **…and you cannot fix it by using more points** | The stride error is **non-monotonic**: 4.07% at 500 points, **5.72% at 250**, 4.07% at 100 **[M]** | It is aliasing luck, not a convergent approximation, so no "enough points" threshold exists. This is the argument for (2) rather than for a larger budget. |
| 3 | **Interpolation smooths a discontinuity** | The Rankine core edge has genuinely discontinuous gradient: one-sided derivatives differ by 2·V₀/r₀ with **opposite signs** **[R]** | `line_shape="linear"` everywhere; **never `"spline"`**. A spline through the core edge draws a smooth curve across a real discontinuity. |
| 4 | **Aliasing when plot rate < physics rate** | Physics is 100 Hz on analysis runs; largest single-step field change is 0.3315 m/s **[M]** | Same as (2), plus: every strip footer prints its **effective sample rate**, so a decimated panel says it is decimated. |
| 5 | **A log axis makes any power law look like "a straight line going down"** | **The best example in this document is §0.5.** On log-log, a slope-1 sequence and a slope-4 sequence are both a tidy descending line. The engine's convergence in a wind field is **1.015** where a reader assumes 4 **[M]**, and no amount of staring at an unreferenced log plot reveals that. | Every log panel draws the **expected slope as a reference line** and prints the fitted slope numerically beside it. Never a log panel without both. |
| 5b | **…and fitting a slope through a non-monotonic sequence invents one** | Across a core traverse the pairwise orders are **+3.46, −1.06, +0.04** **[M]**; a least-squares fit through them returns 0.62, which describes nothing. An earlier draft of this document quoted a ratio of 18.77 from such a sequence as though it meant fourth order. | Refuse to fit when the sequence is non-monotonic; show the points and say so. `verification.fitted_order` already raises on a saturated sequence — same discipline, different failure. |
| 6 | **Colour maps manufacture structure** | — | Perceptually uniform sequential (viridis) for unsigned; **diverging only for genuinely signed quantities** — w_up, load *increment*, energy residual — pinned symmetric about zero. Cite Moreland's diverging-map work, which is ParaView's default for exactly this reason **[C]**. **The repo already does this correctly**: `_field_panel` pins `vmin/vmax` to ±V₀ *"because the physics claim under test is a SIGNED up-then-down doublet: a sign error reverses the colour order and is visible instantly"*. That is the precedent; generalise it. |
| 7 | **Shared axes make two runs look alike** | h and h/2 differ by 0.402 m over a 9.5 km track = **4×10⁻⁵ relative** **[M]** — indistinguishable overlaid | **Comparison panels plot the difference, never two overlaid absolutes**, and the difference panel carries its own axis and its own units. |
| 8 | **Re-evaluating a model at logged states substitutes a different quantity** | Exact today; wrong the moment Dryden lands | `vortex_viz._measure` re-invokes the wind model per sample — correct for a deterministic field, and for a stochastic one it would split the key again and yield a **different realisation from the one flown**. **The UI reads `wind_ned`/`omega_gust` from the artifact and never re-evaluates.** The artifact must therefore carry them, which is §5. |
| 9 | **A correct one-step lag looks like a bug** | 0.3315 m/s = **1.13% of peak gust**, → 7.1×10⁻¹⁵ after a one-step shift **[M]** | P8's shifted-primary/residual-secondary presentation, with the label. |
| 10 | **A green tick on a check that cannot fail** | C1, C2, C6 (§3.2) | Badges carry the measured value and a `tripwire` class, never a bare tick. |

---

## 5. Data layer

### 5.1 The finding that drives this section

**No analysis run is persisted at all.** `viz.save`/`viz.load` exist and work,
but the only caller is `scripts/fly.py` — the interactive panel. `vortex.py`,
`leewave.py` and `microburst.py` each build arrays in memory, draw a PNG, and
exit. Every number in `PROJECT.md` §4's encounter tables was produced by a run
that no longer exists.

And `viz.Trajectory` could not carry them anyway: it has nine array fields and
**no aircraft identity, no wind-field identity, no dt, no trim solution, no
declared parameters, and no git SHA**. `scripts/vortex.py` assembles exactly
that information into its `provenance` string and then renders it into a PNG
footer as text.

So the data layer is greenfield, and the cheapest correct move is to **promote
the provenance string that already exists into structured metadata**.

### 5.2 Format: Parquet, and the reason is not speed

Measured on the canonical 4,018-sample vortex run, 14 channels **[M]**:

| Format | Size | Read all 14 | Read 2 columns |
|---|---|---|---|
| Parquet, zstd | 0.263 MB | 0.81 ms | 0.61 ms |
| Parquet, snappy | 0.293 MB | 0.95 ms | 0.60 ms |
| npz, compressed | **0.198 MB** | 2.94 ms | 0.46 ms |
| npz, raw | 0.453 MB | 1.38 ms | **0.27 ms** |

**Neither is remotely a bottleneck, and npz is actually smaller.** This is a
metadata and ecosystem decision, not a performance one, and it should be stated
that way rather than dressed up with a speed argument.

**Parquet, because:** it carries typed, self-describing per-column metadata and
file-level key-value metadata natively, which is where units and sign
conventions must live; a directory of Parquet files is a dataset any tool can
open without this codebase; and column-subset reads are native, which matters
when a sweep view over 50 runs needs one channel from each.

**Cost, in the same sentence:** `pyarrow` is a ~90 MB wheel added to a project
whose entire runtime dependency list is `jax, numpy, scipy, matplotlib`. If that
is judged too expensive, **npz with a sidecar `run.json` is a legitimate
fallback** and loses only the per-column typing — say so rather than pretending
Parquet is required.

**Rejected:**

- **Zarr and netCDF** — both are for chunked N-dimensional arrays that do not
  fit in memory. The largest run here is 6.4 MB **[M]**. They solve a problem
  this project does not have, at the cost of a heavier dependency and a
  chunking scheme to get wrong.
- **HDF5** — same metadata story as Parquet with a C library, worse tooling, and
  a well-known concurrent-read story that this project would rather not learn.

### 5.3 **Do not store the field. Store its parameters.**

This is the most consequential data-layer decision and it is justified by
measurement rather than taste.

Sampling any of the four fields on a **64³ grid costs 0.56–2.84 ms jitted**
**[M]**. A 64³ grid of three float64 components is 6.3 MB. **Reading that back
from disk would be slower than recomputing it** — and it would be a second copy
of a quantity that is already exactly determined by a `VortexArray`, an
`UpdraftColumn`, a `LeeWave` or a `Microburst` NamedTuple of four to six floats.

So the artifact stores the **field specification**, and the UI reconstructs the
field on load. The brief asks about "time-series plus field-sample data"; the
measured answer is that there is no field-sample data worth persisting.

The one exception: `wind_ned` and `omega_gust` **at the aircraft**, which are
recorded because §4's lie #8 requires it — the flown realisation must not be
re-derivable-only.

### 5.4 Schema

```
runs/<field>-<aircraft>-<sha7>-<timestamp>/
    run.parquet          time series, one row per physics step
    meta.json            everything below
    checks.json          the C1–C11 results (§3.2)
    figure.png           optional, the existing matplotlib figure
```

`run.parquet` columns — one per scalar channel, never a packed vector, so a
column subset is meaningful:

```
t                                        s
pos_n, pos_e, pos_d                      m, NED
vel_u, vel_v, vel_w                      m/s, body
quat_w, quat_x, quat_y, quat_z           -, body->NED
omega_p, omega_q, omega_r                rad/s, body
elevator, aileron, rudder                rad
throttle                                 0-1
wind_n, wind_e, wind_d                   m/s, NED, AS APPLIED (one-step cache)
gust_p, gust_q, gust_r                   rad/s, body, AS APPLIED
inc_CL, inc_Cl, inc_Cm, inc_Cn           -, loads.CoeffIncrement as applied
mode                                     int, manual.Mode
```

Every column carries `units` and `frame` in its Parquet field metadata. The
`increment` columns are new to the log and are what make a strip run
distinguishable from a point run after the fact — without them, `viz.Trajectory`
cannot tell them apart, and `PROJECT.md` §4 records that the two are
bit-identical on the Parks vortex, so the distinction is *only* recoverable
from metadata.

`meta.json`:

```jsonc
{
  "schema_version": 1,
  "git_sha": "a0d791d1844aa0d5929260992ac3cd2e3d686173",
  "git_dirty": false,
  "created": "2026-08-16T...",
  "conventions": {
    "frame": "NED inertial; body x-fwd y-right z-down",
    "quaternion": "[w,x,y,z], body->NED",
    "altitude": "-pos_d",
    "vertical_gust_sign": "wind_d is positive DOWN; plots show -wind_d, up positive"
  },
  "aircraft": {
    "key": "boeing747",
    "source": "NASA CR-2144 Table IX-3/IX-4 flight condition 9",
    "derivative_hash": "<sha256 of the Aircraft NamedTuple, sorted>"
  },
  "flight_condition": { "airspeed_mps": 235.9, "altitude_m": 12192.0 },
  "trim": { "alpha_rad": 0.080918, "elevator_rad": ..., "throttle": ...,
            "residual_norm": 1.933e-20, "is_physical": true },
  "integrator": { "scheme": "RK4 fixed step", "dt_s": 0.01, "n_steps": 4018,
                  "wind_sampling": "once per step, held across four stages",
                  "float": "float64" },
  "wind_field": {
    "kind": "VortexArray",
    "source": "Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129",
    "case": "hannibal",
    "params": { "north": [0.0, 1066.8], "down": [-12192.0, -12192.0],
                "r0": 182.88, "v0": 25.908 },
    "model": "wind.field_model",           // vs sampled_field_model
    "omega_gust_estimator": "analytic tangent at CG"
  },
  "load_model": null,                       // or "loads.strip_model"
  "loading_shape": null,                    // "elliptic" when strip
  "declared_parameters": {                  // <- the load-bearing block
    "lead_in_core_radii": 40.0,
    "window": { "kind": "first core", "north_m": [-182.88, 182.88] },
    "window_rule": "the disturbance's own extent (PROJECT.md 8)"
  },
  "caveats": [
    "Load comparisons are ORDERING ONLY (PROJECT.md 5).",
    "Vortex core is 3.07 spans; point-gust assumption E2 is marginal here."
  ],
  "config_hash": "<sha256 of this object minus created/git_dirty>"
}
```

**`declared_parameters` and `caveats` are not decoration.** The project's
standing rule is *"flag, never invent"*, and every figure this UI draws must be
able to reproduce the provenance footer that `scripts/vortex.py` already writes
by hand. A declared parameter that does not travel with its result is the exact
failure the ledger exists to prevent. `caveats` is a list of strings rendered
verbatim in the footer — so an ordering-only result cannot be displayed as
though it were a value claim.

**`derivative_hash`** answers "is this the same aircraft?" across runs without
comparing 30 floats by eye. **`config_hash`** answers "is this the same
experiment?" and is what a comparison view keys on.

### 5.5 Caching, lazy loading, and multi-run comparison

- **Cache the compile, not the data.** Loading a run is 0.81 ms **[M]**;
  compiling a rollout is 0.6–0.9 s **[M]** and recurs for every distinct
  `n_steps` because it is a `static_argname`. If the UI ever computes anything
  through JAX, that is the thing to memoise.
- **Lazy loading:** for the multi-run sweep, read `meta.json` and `checks.json`
  only — a few kB each. Read `run.parquet` on selection. Column-subset reads make
  a "peak n_z across 50 runs" table cost one column per run.
- **Overlay:** per §4 lie #7, **comparison is a difference panel**, not two
  traces. Runs with different `dt` have different sample grids, so the artifact
  must state its grid and the interpolation used must be **declared on the
  panel** (linear, onto the coarser grid). Comparing runs whose `derivative_hash`
  or `wind_field.params` differ shows a **banner naming exactly which fields
  differ** — an aircraft-A-vs-B comparison that silently also changed the
  altitude is the kind of thing that survives a long time.

---

## 6. What was borrowed, and from where

- **ParaView / Tecplot (CFD post-processing).** The pipeline separation —
  source, filter, view — maps onto field-spec → sampled scalar → render, and is
  why §5.3 stores parameters rather than samples. Directly borrowed: the
  explicit distinction between rescaling colour over the current timestep and
  over all timesteps, which ParaView surfaces as a *button* because the
  automatic mode is misleading **[C]**; and the diverging-colormap default,
  Moreland's work, which is why §4 lie #6 restricts diverging maps to signed
  quantities **[C]**. Also borrowed: ParaView's Spreadsheet View — the numbers
  beside the render — which is §7's readout panel.
- **IADS (Curtiss-Wright), the flight-test standard.** Borrowed: **strip charts
  with threshold checking** (P1 plus the shaded validity bands of P5); the
  **test-point system** for marking events; and above all **Scrollback**, where
  *"all data and displays are updated at the same time to display data from the
  exact same moment"* **[C]** — that is exactly the shared time cursor of §7,
  and the reason it must be a single piece of state rather than per-panel.
- **FDM / DFDR exceedance analysis.** Borrowed: events are *computed* and the
  analyst is *taken to them*, rather than being asked to scroll and look. That
  is §3.3's clickable badge. The whole premise of §3 — that the checks run
  whether or not anyone looks — is the FDM model rather than the dashboard
  model.
- **The papers' own figures.** Wingrove & Bach 1994 Fig. 8 is P4 and the repo
  already implements it. Parks 1985's vertical-wind traces with core positions
  marked are P1's top strip and the 3D slice. Nothing new was invented here; the
  contribution is the hollow whole-run marker and its connector, which the repo
  added and which no paper draws.
- **This repository itself**, which is the most-borrowed-from source. The
  verification/validation tier split becomes §3's `gate`/`tripwire`/`report`
  badge classes. The provenance-footer habit becomes §5.4's `meta.json`. The
  "compute in a module, assert in a test, display in a thin front end" protocol
  becomes §1.3's architecture rule. The ±V₀-pinned diverging colour map becomes
  §4 lie #6.

---

## 7. Screen layouts

### 7.1 Sweep view — one run, judged in seconds, no interaction required

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ boeing747 · Parks hannibal · CR-2144 FC9 · 12192 m · 235.9 m/s · dt 0.01     │
│ trim α 4.636° resid 1.9e-20 · lead-in 40 r₀ · window: first core · a0d791d   │
│ ORDERING ONLY (PROJECT.md §5) · core = 3.07 spans, E2 marginal               │
├──────────────────────────────────────────────────────────────────────────────┤
│ [C5 trim ✓ 1.9e-20] [C8 α window 8.27° green | run 11.39° AMBER]            │
│ [C3 energy 8.8e-4 ✓ order 1.00, expected 1 (wind model present)]            │
│ [C7 step 5.8e-6 m ✓] [C1 quat 2.2e-16 tripwire] [C2 div 0 tripwire]         │
│ [C9 lateral 0 ✓] [C10 point-vs-strip: 2.00 at r₀ ⚠]                         │
├───────────────────────────────────────────┬──────────────────────────────────┤
│ P1  causal strip stack (5 strips)         │ P7  3D: trajectory in the field  │
│     ┌───────────────────────────────────┐ │     vorticity isosurface 32³     │
│     │ w_up  m/s ·············· q_gust   │ │     + w_up slice plane           │
│     ├───────────────────────────────────┤ │     tube coloured by [n_z ▾]     │
│     │ α_air / α_inertial  deg           │ │                                  │
│     ├───────────────────────────────────┤ ├──────────────────────────────────┤
│     │ n_z  g       ── n_z[0] = 0.9967   │ │ P4  Fig. 8 discriminator         │
│     ├───────────────────────────────────┤ │     ● windowed  ○ whole run      │
│     │ θ  deg ················· q        │ │                                  │
│     ├───────────────────────────────────┤ ├──────────────────────────────────┤
│     │ elevator  deg   (flat = correct)  │ │ P3  n_z vs α scatter             │
│     └───────────────────────────────────┘ │     air-relative must be a line  │
│      window shaded on every strip         │                                  │
└───────────────────────────────────────────┴──────────────────────────────────┘
```

Everything above is static on load. **No click is needed to judge the run**,
which is the brief's first job. Total payload: five strips at ≤2,000 points
(0.23 MB), 3D isosurface + slice + tube (1.75 MB), two small 2D panels
(0.05 MB) — **≈2.0 MB, inside budget** **[M]**.

### 7.2 Deep-dive view — "that peak looks wrong" → "here is the geometry", in one click

Same layout, plus a live time cursor and a tab strip of diagnostic panels.

**The one-click path:**

1. **Click the peak on the `n_z` strip.** That sets `t*`, the single piece of
   shared state, in one callback.
2. Everything updates from `t*` **in the same callback**, so nothing tears:
   - a vertical rule appears at `t*` on all five strips;
   - the 3D scene drops a marker on the trajectory at `pos(t*)`, moves the slice
     plane to `east(t*)`, and **does not move the camera** (`uirevision` held);
   - P3 and P4 highlight the corresponding point;
   - the readout panel fills.
3. **The readout panel** — ParaView's Spreadsheet View, borrowed — shows at
   `t*`: `t`, `north` in metres **and in units of r₀**, altitude, `w_up`,
   `q_gust`, `α_air`, `α_inertial`, `n_z`, and the per-sample values of C4, C7
   and C10. The `north/r₀` figure is what turns "that peak" into "0.98 core
   radii, just inside the boundary" without the user doing arithmetic.

That is **one click** from anomaly to field geometry, which is what the brief
asked for.

**Diagnostic tabs** (each renders an artifact field; none computes):

| Tab | Panels |
|---|---|
| **Wind** | P8 recovered vs analytic (shifted + residual), P10 point-vs-strip |
| **Numerics** | P9 energy budget + per-step residual vs position, P11 Richardson vs position |
| **Comparison** | difference panels against a second selected run, with the banner of §5.5 |
| **Model** | the mode/pole and coefficient-sweep views from `validation.py` — deliberately not on the encounter screen (§2.3) |

### 7.3 Multi-run sweep — the table before the picture

A table, one row per run, reading `meta.json` and `checks.json` only: field,
aircraft, dt, the declared parameters, peak `n_z`, Δθ, worst check, and a
verdict column. Sort by any of them. Click a row to open §7.1. Select two rows
to open the Comparison tab.

This is the FDM model rather than the dashboard model: the checks have already
run, and the table's job is to route attention to the run that failed one.

---

## 8. Minimal vertical slice

The smallest thing that renders one real run end-to-end, ordered so the design
gets falsified as early as possible.

**Step 0 — the falsification, before anything else is built.** A 30-line Dash
script: a `Scatter3d` of any 4,000-point array plus a slider whose callback
returns a new figure with `uirevision` held constant. **Verify by hand that
rotating the scene and then moving the slider does not reset the camera.** If
`uirevision` does not hold, the entire linked-cursor design in §7.2 is wrong and
this is discovered in an hour rather than after the data layer is built. *Verify:
camera eye vector unchanged across ten slider moves.*

**Step 1 — artifact writer.** `atisim/analysis/artifact.py` with
`write_run(path, traj, meta, checks)` and `read_run(path)`. Parquet + two JSON
files, §5.4. *Verify: a round-trip test asserting bit-identical arrays and an
exact metadata match; and that reading a run written by a different `git_sha`
raises nothing but records the mismatch.*

**Step 2 — three checks only.** `atisim/checks.py` with C1 (quaternion norm),
C3 (energy closure), C8 (α band, windowed and whole-run). Each with a test in
`atisim/tests/test_checks.py`. Three, not eleven, because three is enough to
establish the pattern and the eleventh adds nothing to the design's falsifiability.
*Verify: each test asserts the measured value in this document — 2.22e-16,
8.8e-4, 8.27°/11.39° — and each is falsified by injecting a defect, per
`PROJECT.md`'s standing rule that a check which can only pass shows nothing.*

**Step 3 — one line in `scripts/vortex.py`.** Write an artifact beside the PNG.
The script already assembles every field `meta.json` needs into its `provenance`
string; this is a restructuring, not new information. *Verify: the existing
printed Δθ/Δn values are unchanged — 2.240°, −1.235 g — and `np.array_equal`
against the in-memory arrays.*

**Step 4 — three pure figure functions.** `atisim/analysis/figures.py`:
`strip_stack(run) -> go.Figure`, `discriminator(runs) -> go.Figure`,
`field_3d(run, scalar) -> go.Figure`. No Dash import. *Verify: each renders in a
notebook cell from a saved run with no simulator in scope — the same standard
`viz.load` already meets.*

**Step 5 — the shell.** `atisim/apps/sweep.py`, one callback:
`clickData → t*` fanning out to every panel.
`python -m atisim.apps.sweep runs/vortex-a0d791d-.../`
*Verify: the sweep view of §7.1 renders inside the 2 MB payload budget — assert
`len(fig.to_json())` per callback — and the one-click path of §7.2 works.*

**What is deliberately not in the slice:** the multi-run table, the comparison
driver, the Model tab, the microburst and lee-wave 3D treatments, and eight of
the eleven checks. Each is additive and none of them can falsify the design.

---

## 9. Assumptions, open questions, and what could not be determined

### 9.1 Assumptions made

1. **The analysis workload is single-user, local, one run at a time.** Nothing
   here would survive a shared server or concurrent writers. `PROJECT.md` §10's
   entry points are all single-shot local commands, so this matches, but it is
   an assumption rather than a stated requirement.
2. **Runs stay in the 2k–32k sample range.** Everything in §1.4 and §5 follows
   from that. A Monte-Carlo ensemble (`batched_rollout` over 100 keys, which
   §7 step 6 of `PROJECT.md` plans) is 100× the volume and **would break the
   payload budget** — see 9.2.
3. **The four current fields are representative.** They are all deterministic,
   position-only, and solenoidal. Dryden is none of those and is planned.
4. **Measurements are single-machine.** All **[M]** figures are medians on this
   Windows machine, jitted JAX on CPU, Plotly 6.9.0, pyarrow 25.0.1. The
   project's own §8 warns that its suite runtime has varied 3× on one machine;
   treat absolute timings as order-of-magnitude and the *ratios* as the finding.
5. **The reader wants correctness first.** Every trade-off above resolves toward
   the option that makes an error visible, at some cost in polish.

### 9.2 Open questions

1. **The first-order-in-wind result (§0.5) needs the project's own treatment,
   and it is not this document's to give.** What was measured is clear: order
   **1.015–1.030** across four independent smooth configurations against
   **3.989** in still air **[R]**. What is *not* settled:

   - **Is it harmful?** Arithmetic says no, at the step sizes in use. Inside the
     first core the h-vs-h/2 position difference is **0.0169 m** **[M]** over a
     366 m traverse. Inside the core `∂w/∂x = V₀/r₀ = 0.1417 s⁻¹`, so that
     displacement is worth **0.0024 m/s of gust error out of a ~26 m/s peak**,
     about 1×10⁻⁴ relative — three orders below the ±25% band `PROJECT.md` §5
     places on the identified vortex parameters. **The headline Δθ = 2.240° and
     Δn = −1.235 g are not threatened.** But that is an estimate made while
     writing a UI document, not a measurement of the headline numbers'
     sensitivity, and it should be redone properly.
   - **Is it avoidable?** Sampling the wind at each RK4 stage would restore the
     order for a deterministic field and would cost four field evaluations per
     step instead of one — cheap, given §1.4's sampling numbers. For a
     *stochastic* field it is the wrong thing to do, which is exactly why
     `integrate.py` holds the wind, and why this is a real design tension rather
     than a bug to fix. A per-model choice may be the answer.
   - **Where does it belong?** By the project's own protocol: a
     `verification.py` function, a test asserting the measured orders, then a
     notebook cell — and an `ASSUMPTIONS.md` §E4 entry recording that the order
     half is now measured. Not in `atisim/checks.py`, because it is a
     property of the scheme rather than of a run.

2. **What does the ensemble view look like?** §7 step 6 wants Fig. 8 with error
   bars from a vmapped ensemble. 100 members × 4,018 samples is 100× the
   measured payload and cannot be drawn as 100 traces. The likely answer is a
   percentile band plus the median trace, which is a different panel design, not
   a scaled one — and it should be designed when Dryden makes the spread
   non-zero, since today every member meets the same field by construction.
3. **Should the UI ever be allowed to run the simulator?** §3.1 says no, on the
   grounds that a hidden 0.6–0.9 s compile makes the panel machine-dependent.
   But the natural interaction "re-fly this at half the step size" is exactly
   what a user will want. The design's answer is a comparison driver invoked
   from the command line; a defensible alternative is an explicit, blocking,
   progress-reporting "re-fly" button that writes a real artifact. **Not
   resolved.**
4. ~~**Does `uirevision` survive a scene whose trace *count* changes?**~~
   **ANSWERED — both halves pass.** Step 0 was run against Dash 4.4.1 /
   Plotly 6.9.0 **[M]**:

   | Probe | Result |
   |---|---|
   | camera across a callback-driven figure replacement, `uirevision='constant'` | eye (0.4, −1.9, 0.8) → **unchanged** |
   | the control, same callback, **no** `uirevision` | → reset to (1.25, 1.25, 1.25) |
   | camera across a **trace-count change**, 2 traces → 1, same `uirevision` | eye (0.4, −1.9, 0.8) → **unchanged** |

   So the linked-cursor design stands, and the 3D pane may switch field
   representation without losing the camera.

   **One methodological note, because it nearly produced a false negative.** The
   first attempt drove the rotation with synthetic `MouseEvent`s on the WebGL
   canvas. Those fired a `plotly_relayout` and populated `_preGUI` — so the
   GUI-edit bookkeeping engaged and the probe *looked* valid — while
   `_scene.getCamera()` showed the camera had not actually moved. The apparent
   "camera reset" was the probe failing, not the mechanism. The reliable
   instrument is the scene's own `setViewport` + `saveLayout`, which is the code
   path plotly's drag handler itself takes. **Check that the thing you perturbed
   actually moved before believing it did not survive.**
5. **What is the right pass threshold for C3?** The measured 8.8×10⁻⁴ at
   dt = 0.01 is a *measurement*, not a bound derived from anything. §4's rule
   forbids picking a tolerance to make a check pass. The honest interim answer
   is to report it as `report` class and only promote it to `gate` once its
   dt-scaling has been established across all four fields — this document only
   established it for the vortex.
6. **Interpolation for cross-`dt` comparison is undecided.** Linear onto the
   coarser grid is proposed; for a fourth-order trajectory that is a first-order
   comparison and may dominate the difference being measured. Not investigated.

### 9.3 What could not be determined from the repository

1. **Where a saved analysis run should live.** `runs/` exists and is
   `.gitignore`d, and it currently holds two `.npz` from `fly.py` plus a PNG.
   Whether artifacts belong there, whether they should be retained, and what the
   retention policy is are not addressed anywhere in the repo.
2. **Whether any interactive-backend performance figure is still valid.**
   `PROJECT.md` §8 explicitly leaves open whether the panel holds 20 fps on
   TkAgg; the only recorded figure is 26.4 fps on Agg. Nothing in this design
   depends on it, but a reader comparing "matplotlib is fast enough" against
   these payload numbers should know that comparison is not available.
3. **The Cessna's status in an analysis UI.** `PROJECT.md` §5 says it is out of
   scope and no result should be quoted from it, but it is in `REGISTRY` and
   every script accepts it via `--aircraft`. Whether the UI should refuse to
   display it, display it with a banner, or ignore the question is a policy call
   the repo has not made. The schema's `caveats` list is the mechanism; the
   policy is not mine to set.
4. **Whether `test_viz.py::test_derived_agrees_with_the_aero_module` is still
   wanted.** `PROJECT.md` §9 session 6 flags it as the structurally-cannot-fail
   test that §6(b) says was replaced, and records that it was never deleted.
   Unrelated to this design; noted because §3's badge classes formalise exactly
   that distinction and someone may want to act on it.

### 9.4 Measurement scripts

All **[M]** figures were produced by seven scripts, run against `a0d791d` with
`PYTHONPATH` set to this worktree:

| Script | Produces |
|---|---|
| `measure_engine.py` | §0.1 volumes, field-sampling throughput, vorticity; §3.2 C2 divergence, finite-difference `omega_gust`; P10 point-vs-strip |
| `measure_checks.py` | rollout compile/warm cost; C1, C6; the first (superseded) energy and format numbers |
| `measure_ui.py` | §5.2 energy closure by integration and its dt-order; §0.2 seam local-vs-cumulative; §5.2 fair format comparison; §1.4 all Plotly payloads |
| `measure_lies.py` | §3.2 C4 residual localisation at core boundaries; P11 Richardson restart; §4 lie #2 decimation loss |
| `measure_lag.py` | P8 one-step cache lag; P5 α-band gate |
| `measure_order.py` | §0.5 first pass — trajectory order with and without a field; energy-residual order on a short window |
| `measure_order2/3.py` | §0.5 the smooth-field discriminator and the corrected single-core kink control |

They live in the session scratchpad, not in the repository — they are
measurements taken to write this document, not tests. **Anything from them that
should not be allowed to rot belongs in `atisim/checks.py` with a test, per
§3.1 — except §0.5, which belongs in `verification.py` per §9.2.1.**

**Three results were superseded during the work and are recorded rather than
quietly dropped**, per `PROJECT.md` §4's rule:

1. The energy residual was first computed by finite-differencing `E`, giving a
   meaningless 2.68%; redone as `ΔE − ∫P dt`, giving 8.8×10⁻⁴.
2. The format comparison first read all npz columns against a two-column
   Parquet read. Redone fairly, npz turns out to be *smaller* than Parquet,
   which is why §5.2 argues from metadata rather than speed.
3. **The `measure_order2.py` "outside the core" control was invalid** — it
   started at 6 r₀ = 1097.3 m, which is 30.5 m from the *second* core at
   1066.8 m, i.e. inside it. It crossed a kink after all and its non-monotonic
   result proved nothing. Redone in `measure_order3.py` with a single core; the
   corrected result is the clean 1.030 in §0.5. This is the same class of error
   as the too-short vortex lead-in of `PROJECT.md` §9 session 3: a control that
   was not controlling what it claimed.
