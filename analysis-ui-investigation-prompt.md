# Investigation prompt — analysis & plausibility-check UI for the turbulence-encounter simulator

## Role

You are investigating the design of a data-analysis interface. **Do not build the application.** Your deliverable is a design document that makes a defensible recommendation and shows its work. Assume the reader will implement it afterwards and will hold you to every claim.

## Context — the system this UI sits on top of

A JAX-based rigid-body flight-dynamics simulator:

Read the documentation Summary.md and assumptions and any other docs for information about the engine 

## What the UI is for

Two jobs, both first-class:

1. **Fast sanity sweep** — after any run, see every relevant channel plotted at once and judge in seconds whether the result is physically sensible. This is a bug-detection instrument, not a presentation layer.
2. **Deep interactive exploration** — drill into a single encounter: link a moment in the time history to a position in the 3D vortex field, compare runs, isolate why a load factor spiked.

Hard requirements:

- **Python-native** stack (Dash / Panel / Streamlit / Bokeh server / marimo / something else you justify). Not a JS frontend.
- **3D visualisation of the trajectory through the wind field is essential**, not decorative. The user must be able to see where in the vortex structure the aircraft was when a given thing happened.
- Nobody flies anything. There is no cockpit, no controls, no real-time piloting.

## Investigate and report on

### 1. Framework selection

Evaluate the realistic Python candidates against criteria that actually matter here, not against feature-list marketing:

- Linked/brushed interaction across many panels (does selecting a time window in one plot update all others, and at what cost in code complexity?).
- 3D rendering quality and interaction: Plotly `Scatter3d`/`Cone`/`Streamtube`, PyVista/VTK, vedo, K3D, ipygany — including how each embeds in the chosen framework and what happens to camera state on callback re-render.
- Redraw latency at realistic data volumes. State the volumes you assume (sample rate × duration × channels; field-sampling grid resolution) and measure or cite rather than guess.
- Callback/reactivity model and how badly it degrades as panel count grows.
- Whether the same components can be reused inside a Jupyter notebook, since exploratory work will happen there too.
- Local-process deployment: startup time, hot reload, one command to launch against a run directory.

Deliver a comparison table plus a single recommendation with the rejected alternatives named and the reason each was rejected.

### 2. Which plots are actually informative

This is the core of the investigation. Do not default to "plot every state variable." Work out, from the physics and the literature, which views carry diagnostic information for a turbulence encounter, and for each one state **what specific error it would reveal**. Candidates to evaluate, extend, and prune:

- Load factor time history (n_z, n_y) with peak annotation.
- The Wingrove & Bach Fig. 8 discriminator (manoeuvre versus turbulence g-loads) — this is a scatter/phase-plane view, not a time series, and it is the closest thing the project has to a domain-standard plot.
- Recovered wind components (u, v, w) against the analytic field values at the aircraft position — i.e. sampled truth versus what the aircraft "felt".
- α, β, airspeed, and the gust-induced increments in each.
- Bowles F-factor time history for the microburst case.
- Power spectral density of the gust input and of the aircraft response, overlaid on the theoretical Dryden/von Kármán spectrum once that layer exists.
- Phase-plane and pole-migration views for the linearised modes (phugoid, short-period, dutch roll) — relevant given the known damping discrepancies.
- Energy budget: kinetic + potential + work done by the wind field.
- 3D: trajectory tube coloured by a selectable scalar (n_z, α, F-factor), rendered inside the vortex structure shown as streamlines, vorticity isosurfaces, glyph field, or slice planes. Evaluate which of those field representations is legible for a line vortex versus a toroidal ring versus a microburst — they may need different treatments.

For each recommended plot, specify: axes, units, sign convention (NED is easy to get backwards), what "correct" looks like, and what a known failure mode looks like.

### 3. Plausibility checking as a built-in feature

The UI is supposed to catch nonsense, so build the checking in rather than relying on the user's eye. Investigate rendering automated invariant checks as pass/fail indicators next to the relevant plot:

- Quaternion norm drift over the run.
- Energy conservation in still air; energy budget closure in a wind field.
- Divergence of the wind field where it should be solenoidal, and the consistency of `omega_gust` against a numerical gradient of the same field.
- The per-step-hold wind seam: an overlay of results at halved step size, showing where the two diverge. This is the most likely place for a silent error and deserves a dedicated view.
- Galilean invariance and torque-free rigid-body checks already in the verification tier, surfaced visually rather than as test output.
- Symmetry checks: a symmetric encounter should produce symmetric lateral response.
- Trim residual and whether the run started genuinely trimmed.

Recommend how these are computed — in the UI, or in the engine and written to the run artifact — and defend the choice.

### 4. How plots lie

Explicitly enumerate the ways a plot can look plausible while the underlying data is wrong, and specify the UI countermeasure for each. At minimum: autoscaling that hides slow drift, interpolation that smooths away a discontinuity, aliasing when the plot's sample rate is below the physics rate, log axes flattering exponential divergence, colour maps that manufacture structure (use perceptually uniform maps; diverging maps only for genuinely signed quantities), and shared-axis defaults that make two different runs look more alike than they are.

### 5. Data layer

The UI must be decoupled from the engine. Investigate:

- Run artifact format (Parquet, Zarr, netCDF, HDF5) and the trade-offs for time-series plus field-sample data.
- A schema that carries units, sign conventions, aircraft identity, wind-field identity and parameters, integrator settings, git SHA and config hash, so any figure is traceable to the run that produced it.
- Caching and lazy loading so the sweep view over many runs stays fast.
- How multiple runs are overlaid for comparison (aircraft A vs B, step size h vs h/2, field variant vs variant).

### 6. Layout

Propose a concrete screen-by-screen design: what the sweep view shows for a single run at a glance, what the deep-dive view contains, how the time cursor is shared between 2D and 3D panels, and how a user gets from "that peak looks wrong" to "here is the field geometry that caused it" in as few actions as possible.

## Method

- Read the actual repository before recommending anything — the wind-field contract, the state layout, what the run loop currently emits, and what `verification.py` / `validation.py` already compute. Recommendations that don't fit the existing interfaces are worthless.
- Look at how adjacent fields solve this: flight-test data review tools, DFDR analysis practice, CFD post-processors (ParaView, Tecplot), and turbulence-encounter papers' own figures. Say what you borrowed and from where.
- Where you assert a performance number, either measure it or cite it. Say which you did.

## Deliverable

A single design document containing:

1. Recommended stack, with the comparison table and the rejected options.
2. The prioritised plot inventory, each entry stating the error it detects.
3. The plausibility-check panel design and where each check is computed.
4. The run-artifact schema.
5. Screen layouts for sweep mode and deep-dive mode.
6. A minimal vertical slice: the smallest implementation that renders one real run end-to-end, so the design can be falsified early.
7. An explicit list of assumptions made and open questions, plus anything you were unable to determine from the repository.

Do not produce the full implementation. If a recommendation carries a cost — performance, complexity, a dependency, a maintenance burden — state the cost in the same sentence as the recommendation.
