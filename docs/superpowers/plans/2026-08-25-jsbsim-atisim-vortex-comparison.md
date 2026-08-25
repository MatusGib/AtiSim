# JSBSim / AtiSim Vortex-Encounter Comparison — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fly AtiSim and JSBSim through the *same* Kelvin–Helmholtz vortex field and report where the two engines' load-factor and pitch responses agree, disagree, and diverge from Wingrove & Bach 1994's own measured numbers.

**Architecture:** JSBSim is driven per-step through `atmosphere/wind-{north,east,down}-fps` from `atisim.wind.vortex_wind`, evaluated at JSBSim's own reported position, with fixed controls. The results are frozen to a checked-in XML by a generator that runs under a *separate* interpreter, exactly as the existing 737 verification does; the test suite and the report read the frozen file and never import jsbsim. AtiSim is then flown through the identical field and the two time histories are compared inside the core window.

**Tech stack:** JSBSim 1.3.1 (system Python 3.10), JAX/AtiSim (project `.venv`), pytest, matplotlib.

---

## Context an engineer needs before touching anything

### The interpreters — there are two, and they are not interchangeable

| Which | Path | Has |
|---|---|---|
| Project venv | `C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe` | jax, atisim, matplotlib, pytest. **No jsbsim, deliberately.** |
| Reference interpreter | `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe` | jsbsim 1.3.1, numpy 2.2.6, scipy 1.15.3, jax 0.6.2. **No matplotlib.** |

Measured, session 19. The generator scripts run under the *reference* interpreter and
`sys.path.insert(0, ROOT)` so they can import `atisim.units` / `atisim.wind`; everything
else runs under the project venv.

### Check which tree you imported, every time

Standing project rule (PROJECT.md §10). From a worktree this **must** be an absolute
`PYTHONPATH`, because the editable install now maps `atisim` to the **main checkout**:

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -c "import atisim; print(atisim.__file__)"
```

If that prints the main checkout while you are in a worktree, every result below is about
someone else's code and reads exactly like a real one.

### The three facts that shaped this design (all measured, session 19)

**1. JSBSim has no writable gust-rate input.** Its property catalog reports
`atmosphere/p-turb-rad_sec`, `q-turb-rad_sec`, `r-turb-rad_sec` as **(R)** — read-only —
and a write to `q-turb-rad_sec` reads back `0.0` after one step. Every writable wind input
(`wind-*-fps`, `gust-*-fps`, `turb-*-fps`, `cosine-gust/*`) is **translational only**, and
JSBSim samples the wind at a single point. So JSBSim *structurally cannot* carry a gust
gradient across the airframe. AtiSim can (`wind.gust_rates`, `wind.sampled_rates`,
`loads.strip_model`). This is a finding to report, not a bug to work around: the gradient
arm of this comparison measures what JSBSim omits, using AtiSim as the instrument.

**2. Translational wind injection works and is validated.** Writing
`atmosphere/wind-down-fps = -50` at M 0.78 / 30,000 ft moved α from 1.925° to 5.612°.
Predicted `atan(50 / 776 ft/s) = 3.69°`; measured change `3.687°`. The hook is correct.

**3. The two 737s are the same aeroplane; the two 747s are not.**

| | AtiSim | JSBSim | Δ |
|---|---|---|---|
| 737 mass | 48,534 kg | 48,534 kg | **0%** |
| 737 S, b, c | 108.8 m², 28.86 m, 3.75 m | 108.79 m², 28.86 m, 3.752 m | **<0.1%** |
| 737 Ixx/Iyy/Izz/Ixz | 8.021e5 / 2.087e6 / 2.693e6 / 2.591e4 | same to 5 s.f. | **<0.1%** |
| 747 mass | 288,773 kg | 249,974 kg | **15.5%** |
| 747 S | 511.0 m² | 524.7 m² | **2.7%** |
| 747 b | 59.64 m | 64.46 m | **8.1%** |
| 747 Iyy | 4.488e7 | 4.489e7 | **<0.1%** |

AtiSim's 737 was recovered *from* JSBSim, so the 737 arm isolates solver and aero model
with mass and geometry held identical. AtiSim's 747 comes from NASA CR-2144 and JSBSim's
B747 is its own model: same inertia tensor, 15.5% different mass. **The 747 arm therefore
compares two different aeroplanes**, and must be reported as such. Task 7 runs it both
as-shipped and mass-matched and reports the spread, following the sensitivity convention
already chosen for the Hannibal radius.

### Which case flies on which airframe, and why

| Case | Altitude | Airframe | Reason |
|---|---|---|---|
| Cimarron | 33,000 ft | **737** | Inside the 737 entry's declared `valid_altitude` = [25,000, 35,000] ft. The only case that is. Also the only case with published time histories (Fig. 3, Fig. 6a) and a published model-vs-data overlay (Fig. 4). |
| Hannibal | 37,000 ft | **747** | Above the 737's band. AtiSim's 747 is defined at 40,000 ft. |
| Morton | 39,000 ft | **747** | Same. |

### The source numbers, and the conflict in them

Wingrove & Bach 1994 Fig. 4 (read off the rendered page, p. 755) gives vortex **diameter**
and tangential velocity:

| Case | Diameter | → r0 | V0 |
|---|---|---|---|
| Hannibal | 1000 ft | **500 ft** | 85 ft/s |
| Morton | 900 ft | **450 ft** | 70 ft/s |
| Cimarron | 900 ft | **450 ft** | 50 ft/s |

`atisim/wind.py:113` has Hannibal `r0 = 600 ft`, citing Parks et al. 1985. Morton
cross-checks exactly (900/2 = 450 = the code), which confirms the column is a diameter and
the code's `r0` is a radius — so Hannibal is a genuine 20% conflict between the two
sources. Parks 1985 is paywalled and has never been obtained (AUDIT.md row 19). **Decision:
run both radii and report the spread.** Do not silently pick one.

Comparison targets from the same paper:

- Cimarron, 33,000 ft: Δn = **+0.73 / −1.20 g**; Δθ = **−0.5° then +1.8°** (p. 755, Fig. 6a)
- Fig. 8 extreme vortex point: Δθ = **1.4°** at Δn = **1.99 g**
- Wave/vortex envelope over all six cases: **+1.70 / −2.01 g** (p. 760)

---

## File structure

| File | Responsibility |
|---|---|
| `scripts/gen_jsbsim_vortex_reference.py` | **New.** Second and last file that imports jsbsim. Flies JSBSim through each case, freezes trajectories to XML. |
| `atisim/tests/data/jsbsim_vortex_reference.xml` | **New, generated, checked in.** |
| `atisim/jsbsim_vortex_ref.py` | **New.** Parser for that XML. Imports no jsbsim, mirrors `jsbsim_ref.py`. |
| `atisim/wind.py` | **Modify.** Add `WINGROVE_FIG4_CASES` beside `PARKS_CASES`. Separate dict, separate citation — do not merge. |
| `scripts/vortex_compare.py` | **New.** Flies AtiSim through the identical field, compares, draws the figure. |
| `atisim/tests/test_jsbsim_vortex.py` | **New.** Tests for the parser, the field agreement, and the comparison gates. |
| `docs/PROJECT.md` | **Modify.** §4 ledger rows, §9 session log, §10 entry points. |
| `audit/AUDIT.md` | **Modify.** Close the Wingrove & Bach `unverifiable` rows — the paper is now in hand. |

---

## Task 1: Record the reference interpreter so the generator is runnable from main

**Files:**
- Modify: `scripts/gen_jsbsim_reference.py:1-27` (docstring)
- Modify: `docs/PROJECT.md` §10

- [ ] **Step 1: Replace the generator's docstring run-line with the measured path**

In `scripts/gen_jsbsim_reference.py`, replace:

```
Run with an interpreter that has JSBSim installed (NOT the project venv, which
deliberately does not):

    python scripts/gen_jsbsim_reference.py
```

with:

```
Run with an interpreter that has JSBSim installed (NOT the project venv, which
deliberately does not). On this machine that interpreter is, measured session 19:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_reference.py

It carries jsbsim 1.3.1, numpy 2.2.6, scipy 1.15.3 and jax 0.6.2 -- everything this
file imports. It does NOT carry matplotlib, so nothing here may import it.

This is no longer the only file that imports jsbsim: scripts/gen_jsbsim_vortex_reference.py
is the second, and there are exactly two. Both freeze their output to XML under
atisim/tests/data/ so the suite runs with no JSBSim present.
```

- [ ] **Step 2: Verify the path in the docstring actually runs**

Run:

```bash
C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe -c "import sys; sys.path.insert(0,'C:/Users/mateusz/UROP/Claude_Flight_Sim'); import jsbsim, atisim.wind; print('both import OK', jsbsim.FGJSBBase().get_version())"
```

Expected: `both import OK 1.3.1 [GitHub build 1837/...]`

- [ ] **Step 3: Add the row to PROJECT.md §10's entry-point table**

```markdown
| `<ref-python> scripts/gen_jsbsim_vortex_reference.py` | **Regenerates the vortex reference.** Needs jsbsim; `<ref-python>` is `C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe`. Writes `atisim/tests/data/jsbsim_vortex_reference.xml`. Run only when the cases or the injection change — drift shows up in `git diff`. |
```

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_jsbsim_reference.py docs/PROJECT.md
git commit -m "Record the interpreter the JSBSim generator actually needs"
```

---

## Task 2: Add the Wingrove Fig. 4 case parameters, separately from Parks

The Fig. 4 numbers are a **different source** from `PARKS_CASES`. Merging them would put two
citations on one dict and lose which number came from where — the failure `PARKS_CASES`'
own comment exists to prevent.

**Files:**
- Modify: `atisim/wind.py` (after `PARKS_CASES`, ~line 115)
- Test: `atisim/tests/test_wind.py`

- [ ] **Step 1: Write the failing test**

Add to `atisim/tests/test_wind.py`:

```python
def test_wingrove_fig4_cases_are_radii_and_agree_with_parks_on_morton():
    """Fig. 4 quotes DIAMETERS; the dict must hold radii.

    Morton is the cross-check that fixes the interpretation: Fig. 4's 900 ft
    diameter is 450 ft of radius, which is PARKS_CASES['morton']['r0'] exactly.
    Hannibal is the conflict -- Fig. 4 says 500 ft where Parks says 600 -- and
    the test pins the disagreement so it cannot be quietly resolved.
    """
    from atisim.wind import PARKS_CASES, WINGROVE_FIG4_CASES
    from atisim.units import FT2M

    assert WINGROVE_FIG4_CASES["morton"]["r0"] == pytest.approx(450.0 * FT2M)
    assert WINGROVE_FIG4_CASES["morton"]["r0"] == pytest.approx(
        PARKS_CASES["morton"]["r0"]
    )
    assert WINGROVE_FIG4_CASES["morton"]["v0"] == pytest.approx(
        PARKS_CASES["morton"]["v0"]
    )
    # The conflict, pinned.
    assert WINGROVE_FIG4_CASES["hannibal"]["r0"] == pytest.approx(500.0 * FT2M)
    assert PARKS_CASES["hannibal"]["r0"] == pytest.approx(600.0 * FT2M)
    # Cimarron exists only in this source.
    assert "cimarron" not in PARKS_CASES
    assert WINGROVE_FIG4_CASES["cimarron"]["r0"] == pytest.approx(450.0 * FT2M)
    assert WINGROVE_FIG4_CASES["cimarron"]["v0"] == pytest.approx(50.0 * FT2M)
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_wind.py::test_wingrove_fig4_cases_are_radii_and_agree_with_parks_on_morton -v
```

Expected: FAIL — `ImportError: cannot import name 'WINGROVE_FIG4_CASES'`

- [ ] **Step 3: Add the dict**

In `atisim/wind.py`, immediately after `PARKS_CASES`:

```python
# ---------------------------------------------------------------------------
# The same vortices as PARKS_CASES, as the LATER paper reports them.
#
# Source: R. C. Wingrove, R. E. Bach Jr., "Severe Turbulence and Maneuvering
# from Airline Flight Records", J. Aircraft 31(4), Jul-Aug 1994, pp. 753-760.
# Fig. 4, p. 755, "Models for vortex-induced turbulence".
#
# Fig. 4's columns are headed "Vortex diameter (feet)" and "Tangential velocity
# (ft/sec)". The values there are 1000/85, 900/70 and 900/50. They are stored
# here as RADII, halved, so this dict has the same units as PARKS_CASES.
#
# MORTON FIXES THE INTERPRETATION: 900 ft of diameter is 450 ft of radius, and
# PARKS_CASES["morton"]["r0"] is 450 ft to the digit. So the column really is a
# diameter and `r0` really is a radius.
#
# *** HANNIBAL DISAGREES AND THAT IS NOT RESOLVED HERE. *** Fig. 4's 1000 ft
# diameter is a 500 ft radius; PARKS_CASES says 600 ft, citing Parks et al.
# 1985, which has never been obtained (AUDIT.md row 19). Both are kept, both are
# flown, and the spread is reported. Neither is deleted in favour of the other.
#
# CIMARRON APPEARS ONLY HERE. Parks 1985 identifies two cases; this paper adds a
# third, and it is the one with published time histories (Fig. 3, Fig. 6a) and a
# published model-vs-data overlay (Fig. 4), which is why it is the 737's case.
#
# `spacing` is deliberately ABSENT: Fig. 4 gives core size and strength and says
# nothing about the array spacing, so anything here would be invented. Callers
# that need an array take the spacing from PARKS_CASES and say so.
# ---------------------------------------------------------------------------
WINGROVE_FIG4_CASES: dict[str, dict[str, float]] = {
    "hannibal": {"r0": 500.0 * FT2M, "v0": 85.0 * FT2M},
    "morton": {"r0": 450.0 * FT2M, "v0": 70.0 * FT2M},
    "cimarron": {"r0": 450.0 * FT2M, "v0": 50.0 * FT2M},
}

# The altitude each case was flown at, Table 1 p. 754, in metres. Needed because
# these three cases are NOT at one altitude and the density drives every force.
WINGROVE_CASE_ALTITUDE: dict[str, float] = {
    "hannibal": 37000.0 * FT2M,
    "morton": 39000.0 * FT2M,
    "cimarron": 33000.0 * FT2M,
}
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_wind.py -v -k wingrove
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add atisim/wind.py atisim/tests/test_wind.py
git commit -m "Add Wingrove Fig. 4's own vortex numbers, and pin where they disagree with Parks"
```

---

## Task 3: The JSBSim vortex generator

**Files:**
- Create: `scripts/gen_jsbsim_vortex_reference.py`
- Creates (generated): `atisim/tests/data/jsbsim_vortex_reference.xml`

- [ ] **Step 1: Write the generator**

Create `scripts/gen_jsbsim_vortex_reference.py`:

```python
"""Freeze JSBSim's response to the Wingrove & Bach vortex cases.

Run with the interpreter that has JSBSim installed -- NOT the project venv:

    C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe \
        scripts/gen_jsbsim_vortex_reference.py

Writes atisim/tests/data/jsbsim_vortex_reference.xml. The suite and the report
read that file and never import jsbsim, so reference drift shows up in git diff
rather than as a mysterious change in test results.

This is the SECOND of exactly two files in the project that import jsbsim; the
other is scripts/gen_jsbsim_reference.py.

WHAT IS INJECTED, AND WHAT CANNOT BE
------------------------------------
JSBSim's wind is sampled at ONE point and enters only as a translational
velocity. Its property catalog reports atmosphere/{p,q,r}-turb-rad_sec as
READ-ONLY, and a write to q-turb-rad_sec reads back 0.0 after a step (measured,
session 19). So the gust GRADIENT across the airframe -- which AtiSim carries
via wind.gust_rates -- has no injection point here and is structurally absent
from the JSBSim side. That asymmetry is a result of this comparison, not a
defect in it, and scripts/vortex_compare.py measures it.

Both engines are therefore run with the TRANSLATIONAL FIELD ONLY for the
headline comparison. AtiSim's gradient terms are then switched on separately.
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import jsbsim
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from atisim.units import FT2M, LB2KG, SLUG_FT2_TO_KG_M2  # noqa: E402
from atisim.wind import (  # noqa: E402
    PARKS_CASES, WINGROVE_CASE_ALTITUDE, WINGROVE_FIG4_CASES,
)

DATA = ROOT / "atisim" / "tests" / "data"
OUT = DATA / "jsbsim_vortex_reference.xml"

MS2FPS = 1.0 / FT2M
DT = 1.0 / 120.0
SAMPLE_EVERY = 0.02      # s. The core traverse is ~1 s (Fig. 4's scale bar), so
                         # 0.05 as used by the 737 generator would put only 20
                         # samples across the whole event.
LEAD_IN_RADII = 15.0     # core radii upstream of the first core. See docstring
                         # of run_case for why not 40.

# (case, jsbsim model, atisim registry name, mach). The altitude comes from
# WINGROVE_CASE_ALTITUDE -- Table 1's own value for that incident.
CASES = [
    ("cimarron", "737",  "boeing737", 0.78),
    ("hannibal", "B747", "boeing747", 0.80),
    ("morton",   "B747", "boeing747", 0.80),
]

# Both radii for every case, so the Hannibal conflict is flown rather than
# argued. "wingrove" is Fig. 4's; "parks" is wind.py's incumbent.
RADIUS_SOURCES = ("wingrove", "parks")


def rankine(north, down, core_north, core_down, r0, v0):
    """Parks Eqs. (3)-(6), in plain numpy.

    Deliberately NOT imported from atisim.wind: that function is JAX and this
    interpreter's jax is a different build. More importantly, a bug shared by
    both sides of a cross-code comparison is invisible to it, so the field is
    written twice and Task 5 asserts the two agree to 1e-12.
    """
    along = north - core_north
    above = core_down - down
    r2 = along**2 + above**2
    if r2 < r0**2:
        w_h = v0 * above / r0
        w_up = -v0 * along / r0
    else:
        w_h = v0 * r0 * above / r2
        w_up = -v0 * r0 * along / r2
    return np.array([w_h, 0.0, -w_up])   # NED, z down


def trimmed(model, altitude_m, mach):
    fdm = jsbsim.FGFDMExec(None)
    fdm.set_debug_level(0)
    if not fdm.load_model(model):
        raise SystemExit(f"JSBSim could not load {model}")
    fdm["ic/h-sl-ft"] = altitude_m / FT2M
    fdm["ic/mach"] = mach
    fdm["ic/gamma-deg"] = 0.0
    fdm["ic/psi-true-deg"] = 0.0
    fdm["ic/lat-gc-deg"] = 39.0
    fdm["ic/long-gc-deg"] = -105.0
    fdm.run_ic()
    fdm["simulation/do_simple_trim"] = 1
    fdm["atmosphere/turb-type"] = 0          # no internal turbulence
    fdm["simulation/gravity-model"] = 0      # constant g, matching atisim
    fdm.set_dt(DT)
    return fdm


def run_case(case, model, mach, radius_source, seconds_pad=6.0):
    """Fly one case with fixed controls and per-step wind injection.

    LEAD-IN. scripts/vortex.py defaults to 40 core radii because below ~12 the
    1/r far field launches the aircraft out of equilibrium. 40 radii is ~5.5 km,
    which at 236 m/s is 23 s of fixed-control flight BEFORE the core -- and
    JSBSim sheds ~2.6 m/s of airspeed over the first 3 s at fixed throttle
    (measured). Both engines would arrive at the core off-trim, by different
    amounts, and that drift would be read as a solver difference. 15 radii is
    above the far-field threshold and a third of the drift. The drift is
    recorded as a diagnostic and Task 6 gates on it.
    """
    altitude = WINGROVE_CASE_ALTITUDE[case]
    src = WINGROVE_FIG4_CASES[case] if radius_source == "wingrove" else PARKS_CASES[case]
    r0, v0 = src["r0"], src["v0"]

    fdm = trimmed(model, altitude, mach)
    de_trim = fdm["fcs/elevator-pos-rad"]
    core_north = LEAD_IN_RADII * r0
    core_down = -altitude
    V0 = fdm["velocities/vtrue-fps"] * FT2M
    duration = (core_north + seconds_pad * r0) / V0

    entry = {
        "mass": fdm["inertia/mass-slugs"] * 14.593903,
        "S": fdm["metrics/Sw-sqft"] * FT2M**2,
        "b": fdm["metrics/bw-ft"] * FT2M,
        "c": fdm["metrics/cbarw-ft"] * FT2M,
        "Iyy": fdm["inertia/iyy-slugs_ft2"] * SLUG_FT2_TO_KG_M2,
        "density": fdm["atmosphere/rho-slugs_ft3"] * 515.378818,
        "airspeed": V0,
        "alpha_trim": fdm["aero/alpha-deg"],
        "theta_trim": fdm["attitude/theta-deg"],
        "Nz_trim": fdm["accelerations/Nz"],
    }

    samples, next_sample = [], 0.0
    while fdm["simulation/sim-time-sec"] <= duration:
        north = fdm["position/distance-from-start-lat-mt"]
        down = -fdm["position/h-sl-meters"]
        w = rankine(north, down, core_north, core_down, r0, v0)
        fdm["atmosphere/wind-north-fps"] = w[0] * MS2FPS
        fdm["atmosphere/wind-east-fps"] = w[1] * MS2FPS
        fdm["atmosphere/wind-down-fps"] = w[2] * MS2FPS
        # Fixed controls. The Fig. 8 discriminator separates turbulence from
        # MANOEUVRING by whether pitch correlates with elevator, so any closed
        # loop here would make the categories ambiguous.
        fdm["fcs/pitch-trim-cmd-norm"] = de_trim / 0.3
        t = fdm["simulation/sim-time-sec"]
        if t >= next_sample:
            samples.append({
                "t": t, "north": north, "altitude": -down,
                "wind": w.copy(),
                "alpha": fdm["aero/alpha-deg"],
                "theta": fdm["attitude/theta-deg"],
                "q": fdm["velocities/q-rad_sec"],
                "Nz": fdm["accelerations/Nz"],
                "vtrue": fdm["velocities/vtrue-fps"] * FT2M,
                "elevator": fdm["fcs/elevator-pos-rad"],
            })
            next_sample += SAMPLE_EVERY
        fdm.run()

    return entry, samples, {"r0": r0, "v0": v0, "core_north": core_north,
                            "altitude": altitude, "duration": duration}


def main():
    root = ET.Element("jsbsim_vortex_reference")
    prov = ET.SubElement(root, "provenance")
    ET.SubElement(prov, "jsbsim_version").text = jsbsim.FGJSBBase().get_version()
    ET.SubElement(prov, "dt").text = repr(DT)
    ET.SubElement(prov, "sample_every").text = repr(SAMPLE_EVERY)
    ET.SubElement(prov, "lead_in_radii").text = repr(LEAD_IN_RADII)
    ET.SubElement(prov, "gradient_injected").text = "false"

    for case, model, ac_name, mach in CASES:
        for source in RADIUS_SOURCES:
            entry, samples, geom = run_case(case, model, mach, source)
            e = ET.SubElement(root, "encounter",
                              case=case, model=model, aircraft=ac_name,
                              radius_source=source, mach=repr(mach))
            for k, v in {**entry, **geom}.items():
                ET.SubElement(e, "value", name=k).text = repr(float(v))
            for s in samples:
                ET.SubElement(
                    e, "sample",
                    t=repr(s["t"]), north=repr(s["north"]),
                    altitude=repr(s["altitude"]),
                    wind=" ".join(repr(float(x)) for x in s["wind"]),
                    alpha=repr(s["alpha"]), theta=repr(s["theta"]),
                    q=repr(s["q"]), Nz=repr(s["Nz"]),
                    vtrue=repr(s["vtrue"]), elevator=repr(s["elevator"]),
                )
            print(f"{case:10} {model:5} r0={source:8} "
                  f"{len(samples):5d} samples  "
                  f"Nz in [{min(s['Nz'] for s in samples):+.4f}, "
                  f"{max(s['Nz'] for s in samples):+.4f}]")

    ET.indent(root)
    OUT.write_text(ET.tostring(root, encoding="unicode"), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

```bash
C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe scripts/gen_jsbsim_vortex_reference.py
```

Expected: six lines (three cases × two radius sources), each reporting a sample count and
an Nz range that brackets 1.0, then `wrote .../jsbsim_vortex_reference.xml`.

**If Nz never leaves ±0.01 of 1.0, the wind is not reaching the aero.** Check that
`atmosphere/wind-down-fps` reads back non-zero mid-run before changing anything else.

- [ ] **Step 3: Commit the generator and the reference together**

```bash
git add scripts/gen_jsbsim_vortex_reference.py atisim/tests/data/jsbsim_vortex_reference.xml
git commit -m "Freeze JSBSim's answer to the three vortex cases, both radii"
```

---

## Task 4: The reader

**Files:**
- Create: `atisim/jsbsim_vortex_ref.py`
- Test: `atisim/tests/test_jsbsim_vortex.py`

- [ ] **Step 1: Write the failing test**

Create `atisim/tests/test_jsbsim_vortex.py`:

```python
"""Tests for the frozen JSBSim vortex reference."""

import numpy as np
import pytest

from atisim import jsbsim_vortex_ref


def test_reference_has_all_six_encounters():
    ref = jsbsim_vortex_ref.load()
    keys = set(ref.encounters)
    assert keys == {
        ("cimarron", "wingrove"), ("cimarron", "parks"),
        ("hannibal", "wingrove"), ("hannibal", "parks"),
        ("morton", "wingrove"), ("morton", "parks"),
    }


def test_cimarron_flew_the_737_and_hannibal_the_747():
    ref = jsbsim_vortex_ref.load()
    assert ref.encounters[("cimarron", "wingrove")].model == "737"
    assert ref.encounters[("hannibal", "wingrove")].model == "B747"


def test_every_encounter_actually_saw_the_vortex():
    """A run where the wind never reached the aero is the silent failure here."""
    ref = jsbsim_vortex_ref.load()
    for key, enc in ref.encounters.items():
        wz = np.array([s.wind[2] for s in enc.samples])
        assert np.abs(wz).max() > 5.0, f"{key}: peak |w_down| only {np.abs(wz).max()}"
        nz = np.array([s.Nz for s in enc.samples])
        assert nz.max() - nz.min() > 0.2, f"{key}: Nz barely moved"
```

- [ ] **Step 2: Run it to verify it fails**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_jsbsim_vortex.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'atisim.jsbsim_vortex_ref'`

- [ ] **Step 3: Write the reader**

Create `atisim/jsbsim_vortex_ref.py`:

```python
"""Reader for the frozen JSBSim vortex reference.

Deliberately does NOT import jsbsim, and nothing that imports this module may.
The reference is generated by scripts/gen_jsbsim_vortex_reference.py under a
separate interpreter and checked in, so the suite runs on a machine with no
JSBSim -- which is this project's own venv.

Mirrors atisim/jsbsim_ref.py; kept separate because that file's schema is a
trim-point derivative recovery and this one is a set of time histories.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_vortex_reference.xml"


class Sample(NamedTuple):
    t: float
    north: float        # m, along-track from the start point
    altitude: float     # m
    wind: np.ndarray    # (3,) NED m/s, as injected
    alpha: float        # deg, as JSBSim reported
    theta: float        # deg
    q: float            # rad/s
    Nz: float           # g, JSBSim's own normal load factor
    vtrue: float        # m/s
    elevator: float     # rad


class Encounter(NamedTuple):
    case: str
    model: str          # JSBSim's model name
    aircraft: str       # the atisim REGISTRY name it must be compared against
    radius_source: str  # "wingrove" or "parks"
    mach: float
    values: dict        # mass, S, b, c, Iyy, density, airspeed, r0, v0, ...
    samples: list


class Reference(NamedTuple):
    jsbsim_version: str
    dt: float
    sample_every: float
    lead_in_radii: float
    gradient_injected: bool
    encounters: dict    # (case, radius_source) -> Encounter


def load(path: Path = REFERENCE) -> Reference:
    root = ET.parse(path).getroot()
    encounters = {}
    for e in root.findall("encounter"):
        samples = [
            Sample(
                t=float(s.get("t")), north=float(s.get("north")),
                altitude=float(s.get("altitude")),
                wind=np.array([float(v) for v in s.get("wind").split()]),
                alpha=float(s.get("alpha")), theta=float(s.get("theta")),
                q=float(s.get("q")), Nz=float(s.get("Nz")),
                vtrue=float(s.get("vtrue")), elevator=float(s.get("elevator")),
            )
            for s in e.findall("sample")
        ]
        key = (e.get("case"), e.get("radius_source"))
        encounters[key] = Encounter(
            case=e.get("case"), model=e.get("model"),
            aircraft=e.get("aircraft"), radius_source=e.get("radius_source"),
            mach=float(e.get("mach")),
            values={v.get("name"): float(v.text) for v in e.findall("value")},
            samples=samples,
        )
    return Reference(
        jsbsim_version=root.findtext("provenance/jsbsim_version"),
        dt=float(root.findtext("provenance/dt")),
        sample_every=float(root.findtext("provenance/sample_every")),
        lead_in_radii=float(root.findtext("provenance/lead_in_radii")),
        gradient_injected=root.findtext("provenance/gradient_injected") == "true",
        encounters=encounters,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_jsbsim_vortex.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add atisim/jsbsim_vortex_ref.py atisim/tests/test_jsbsim_vortex.py
git commit -m "Read the vortex reference without importing jsbsim"
```

---

## Task 5: Prove both engines flew the same wind field

The comparison is worthless if the two sides saw different air. The generator writes the
field in plain numpy; `atisim.wind.vortex_wind` computes it in JAX. Task 3's docstring says
these are written twice on purpose — this task cashes that in.

**Files:**
- Modify: `atisim/tests/test_jsbsim_vortex.py`

- [ ] **Step 1: Write the failing test**

```python
def test_injected_field_matches_atisim_field_at_every_sample():
    """The two engines must have flown the same air, to floating-point.

    The generator recomputes the Rankine field in numpy rather than calling
    atisim.wind.vortex_wind, precisely so that a bug in that function cannot
    hide by being present on both sides. This is where the two are reconciled.
    """
    import jax.numpy as jnp
    from atisim import wind

    ref = jsbsim_vortex_ref.load()
    for key, enc in ref.encounters.items():
        array = wind.VortexArray(
            north=jnp.array([enc.values["core_north"]]),
            down=jnp.array([-enc.values["altitude"]]),
            r0=jnp.array(enc.values["r0"]),
            v0=jnp.array(enc.values["v0"]),
        )
        worst = 0.0
        for s in enc.samples:
            pos = jnp.array([s.north, 0.0, -s.altitude])
            mine = np.asarray(wind.vortex_wind(pos, array))
            worst = max(worst, float(np.abs(mine - s.wind).max()))
        assert worst < 1e-9, f"{key}: fields differ by {worst} m/s"
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_jsbsim_vortex.py::test_injected_field_matches_atisim_field_at_every_sample -v
```

Expected: PASS. **If it fails, stop.** Either the generator's `rankine` or `wind.vortex_wind`
is wrong, and nothing downstream means anything until that is settled.

- [ ] **Step 3: Commit**

```bash
git add atisim/tests/test_jsbsim_vortex.py
git commit -m "Reconcile the two independently-written vortex fields"
```

---

## Task 6: Fly AtiSim through the identical field and compare

**Files:**
- Create: `scripts/vortex_compare.py`
- Modify: `atisim/tests/test_jsbsim_vortex.py`

- [ ] **Step 1: Write the comparison script**

Create `scripts/vortex_compare.py`. It must, for each of the six encounters:

1. Build the same single-core `VortexArray` from the reference's own `core_north`,
   `altitude`, `r0`, `v0` — never from the case dicts, so the two sides cannot drift apart.
2. **Density-match the altitude**, exactly as `jsbsim_ref.Condition` documents: AtiSim's ISA
   uses geometric altitude where the standard uses geopotential, so a nominal match puts a
   same-signed bias on every force. Solve for the geometric altitude where
   `atisim.atmosphere.density(h)` equals the reference's `density`, with `scipy.optimize.brentq`.
3. Fly with `vortex_viz.fly(..., strip=False)` at the reference's own `airspeed`, starting at
   `start_north = 0.0` (the generator's own origin), for the reference's `duration`.
4. Extract, over the window `north ∈ [core_north − r0, core_north + r0]`: `Δn` max and min,
   `Δθ` max and min, peak `|α|`.
5. Re-fly with AtiSim's gradient terms on (`wind.gust_rates` via `field_model`, then
   `strip=True`) and report the delta as **the term JSBSim cannot carry**.
6. Print a table and, with `--png`, draw: vertical wind, Δn, Δθ and α for both engines
   overlaid, plus a Fig. 8 panel with the paper's three published points marked.

Report columns, per encounter: `Δn⁺`, `Δn⁻`, `Δθ⁺`, `Δθ⁻` for JSBSim, for AtiSim
translational-only, for AtiSim with gradients, and the paper's value where one exists.

- [ ] **Step 2: Write the gate test**

```python
def test_pre_core_drift_is_small_enough_to_attribute_differences():
    """Fixed-control flight sheds airspeed before the core is reached.

    If the two engines arrive at the core at materially different speeds, any
    difference in the core response is a drift artifact, not a solver result.
    This gates the whole comparison. The bound is 0.5% of true airspeed --
    below that, qbar differs by under 1% and the load-factor difference it
    can explain is smaller than the effects being measured.
    """
    ref = jsbsim_vortex_ref.load()
    for key, enc in ref.encounters.items():
        entry_speed = enc.values["airspeed"]
        core_north = enc.values["core_north"] - enc.values["r0"]
        at_core = min((s for s in enc.samples if s.north >= core_north),
                      key=lambda s: s.north, default=None)
        assert at_core is not None, f"{key}: no sample reached the core"
        drift = abs(at_core.vtrue - entry_speed) / entry_speed
        assert drift < 0.005, f"{key}: JSBSim shed {drift:.2%} before the core"
```

- [ ] **Step 3: Run both**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest atisim/tests/test_jsbsim_vortex.py -v
PYTHONPATH="$PWD" .venv/Scripts/python.exe scripts/vortex_compare.py --png runs/vortex_compare.png
```

Expected: tests pass; the script prints six rows and writes the figure.

**If the drift gate fails,** lower `LEAD_IN_RADII` in the generator toward 12 and
regenerate. Do not widen the bound — the bound is what makes the comparison mean anything.

- [ ] **Step 4: Commit**

```bash
git add scripts/vortex_compare.py atisim/tests/test_jsbsim_vortex.py runs/vortex_compare.png
git commit -m "Fly both engines through one vortex and report where they part"
```

---

## Task 7: The 747 mass conflict, measured rather than argued

AtiSim's 747 is 288,773 kg; JSBSim's is 249,974 kg, with the same inertia tensor. Since
`n = L/W`, mass is first-order for Δn, and `Iyy` is first-order for Δθ — the paper's own
explanation for the small vortex Δθ is "the large pitch moment-of-inertia and the short
time required to traverse". So neither can be waved off.

**Files:**
- Modify: `scripts/gen_jsbsim_vortex_reference.py`
- Modify: `scripts/vortex_compare.py`

- [ ] **Step 1: Add a mass-matched variant to the generator**

Add to `trimmed()`, guarded by a new parameter `match_mass_kg=None`:

```python
    if match_mass_kg is not None:
        # Trim the difference out of the fuel tanks. JSBSim recomputes the
        # inertia tensor from mass_balance plus tank contents, so this MATCHES
        # THE MASS AT THE COST OF PERTURBING Iyy -- which is the whole point of
        # running it as a second variant rather than a correction. Both are
        # reported; neither is the answer on its own.
        want_lbs = match_mass_kg / LB2KG
        have_lbs = fdm["inertia/weight-lbs"]
        delta = want_lbs - have_lbs
        fdm["inertia/pointmass-weight-lbs"] = max(0.0, delta)
        fdm.run_ic()
        fdm["simulation/do_simple_trim"] = 1
```

Extend `CASES` so each 747 case runs twice, `mass="shipped"` and `mass="matched"`, and add
`mass` to the encounter key and XML attributes. Record the achieved `Iyy` in `values` both
times so the perturbation is visible in the reference itself.

- [ ] **Step 2: Regenerate and inspect the inertia cost**

```bash
C:/Users/mateusz/AppData/Local/Programs/Python/Python310/python.exe scripts/gen_jsbsim_vortex_reference.py
```

Expected: ten encounters. Read off the `Iyy` for `shipped` vs `matched` — that number is
the price of matching the mass, and it goes in the report verbatim.

- [ ] **Step 3: Report the spread rather than a single 747 number**

In `scripts/vortex_compare.py`, the 747 rows print as a range across the two mass variants,
with a footnote naming the two masses and the two `Iyy`. The 737 rows print a single value
and a note that its mass and geometry match JSBSim's to under 0.1%.

- [ ] **Step 4: Commit**

```bash
git add scripts/gen_jsbsim_vortex_reference.py scripts/vortex_compare.py atisim/tests/data/jsbsim_vortex_reference.xml
git commit -m "Fly the 747 at both masses, because the two models are not the same aeroplane"
```

---

## Task 8: Close the audit rows this paper now settles

`audit/AUDIT.md` marks a set of numbers `unverifiable — source not available` because
Wingrove & Bach 1994 was paywalled. It is now in hand.

**Files:**
- Modify: `audit/AUDIT.md` (rows 19, 175, 388, 524, 657, 658)
- Modify: `audit/audit_evidence/B-reference-tracing.md` (lines 168, 393)
- Modify: `docs/PROJECT.md` §4

- [ ] **Step 1: Verify each claim against the paper, one at a time**

| Claim in the code | Paper says | Verdict |
|---|---|---|
| `UPDRAFT_W0 = 80 ft/s` | p. 756: "vertical winds over 80 ft/s", Bermuda 12 Oct 1983 | **verified** |
| `UPDRAFT_SECONDS = 20.0` | p. 756: "a 20-s encounter with an 80 ft/s-updraft" | **verified** |
| 5.2° pitch variation | p. 756: "resulted in a 5.2-deg variation in pitch angle" | **verified** |
| +0.66 / −1.58 g | p. 756: "positive load change of 0.66 g … negative load change of 1.58 g" | **verified** |
| 50 / 100 ft/s updraft statement | p. 756: "updrafts of 50 ft/s … can reach values of 100 ft/s" | **verified** |
| Fig. 8 discriminator, ±g asymmetry | p. 755 and Fig. 8 | **verified** |

- [ ] **Step 2: Rewrite the rows, superseding rather than deleting**

PROJECT.md §4's rule is that a ledger row is never deleted, only superseded. Add new rows
citing "Wingrove & Bach 1994, obtained session 19" and mark the old ones superseded.

- [ ] **Step 3: Add the Hannibal conflict as a NEW open row**

It is not resolved — Parks 1985 is still unobtained. Record it in §8 open questions with
both values and the note that both are flown.

- [ ] **Step 4: Commit**

```bash
git add audit/AUDIT.md audit/audit_evidence/B-reference-tracing.md docs/PROJECT.md
git commit -m "Close six audit rows the Wingrove paper settles, and open one it does not"
```

---

## Task 9: The report

**Files:**
- Create: `scripts/vortex_compare_report.py` (follow `scripts/jsbsim_report.py`'s structure)
- Creates: `docs/summary/jsbsim-atisim-vortex-report.pdf`

- [ ] **Step 1: Build the report**

Sections, in order:

1. **What was compared, and what could not be.** The read-only gust-rate finding, stated
   plainly with the property flags as evidence.
2. **The source.** Wingrove & Bach 1994, the three cases, Fig. 4's table, and the note that
   the paper names no aircraft — so the airframes come from the JSBSim ∩ AtiSim
   intersection, not from the paper.
3. **Input agreement.** The field reconciliation (Task 5) and the drift gate (Task 6).
4. **737 / Cimarron.** The headline: identical mass and geometry, so the difference is
   solver and aero model alone. Against the paper's +0.73 / −1.20 g.
5. **747 / Hannibal and Morton.** Both radii, both masses, reported as spreads.
6. **What the gradient adds**, and therefore what JSBSim omits.
7. **Fig. 8.** Both engines' points against the paper's three published categories.

- [ ] **Step 2: Run the full suite**

```bash
PYTHONPATH="$PWD" .venv/Scripts/python.exe -m pytest -q
```

Expected: all pass. Record the exact count in PROJECT.md §9 — do not write "all pass"
without the number.

- [ ] **Step 3: Commit**

```bash
git add scripts/vortex_compare_report.py docs/summary/jsbsim-atisim-vortex-report.pdf docs/PROJECT.md
git commit -m "Report where the two engines part in a vortex, and where neither matches the DFDR"
```

---

## Self-review notes

- **Every task has a gate that can fail for a real reason.** Task 3 fails if the wind never
  reaches the aero, Task 5 if the two fields differ, Task 6 if drift dominates.
- **Nothing here silently resolves the Hannibal conflict**; both radii are flown throughout
  and the plan never collapses them.
- **The 747 caveat propagates** from Task 7 into Task 9 §5 rather than being noted once and
  forgotten.
- **`WINGROVE_FIG4_CASES` has no `spacing` key**, and Task 3 uses a single core, so nothing
  reads a key that does not exist.
- **Known thin spot:** Task 6 step 1 is prose, not code, because the shape of
  `vortex_viz.fly`'s return value must be read before the extraction can be written
  correctly. Read `atisim/vortex_viz.py`'s `Encounter` before starting it.
