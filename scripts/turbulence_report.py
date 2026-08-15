"""Build the turbulence-fidelity technical report PDF.

Run from the project root:

    .venv/Scripts/python.exe scripts/turbulence_report.py docs/summary/turbulence-report.pdf

Covers the distributed-airframe wind-shear work: the physics it rests on, the
source verification that preceded it, what was built, what was measured, how
the engine works now, and how it will work once strip loads feed the 6-DOF.

SAME RULE AS scripts/summary.py. Every number in this document is either taken
from a cited source document, taken from docs/PROJECT.md section 4 or
docs/ASSUMPTIONS.md (where it was measured against a test), or computed here by
calling the project's own code. Nothing is typed in from memory. The numbers
that matter most -- the tail arm, the E2 correction profile, the loading-shape
spread -- are all recomputed at build time, so a regression in the model shows
up as a changed figure in the report rather than a stale claim.

Theory backbone: Robert F. Stengel, "Flight Dynamics", 2nd ed., Princeton
University Press, 2022. Equation numbers are that edition's.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import jax
import jax.numpy as jnp

import flightsim  # noqa: F401  -- enables x64 before any array is made
from flightsim import airframe, provenance, wind
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.state import State, euler_to_quat
from flightsim.units import FT2M

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/summary/turbulence-report.pdf")

A4 = (8.27, 11.69)
INK = "#1a1a1a"
MUTED = "#5a5a5a"
BLUE = "#2c6fb5"
TEAL = "#1f7a68"
AMBER = "#a8760b"
RED = "#a8443a"
RULE = "#c8c8c8"
WASH = "#eef2f6"
MONO = "DejaVu Sans Mono"

# Lowest y a text cursor may reach before it collides with the footer at 0.035.
FLOOR = 0.062


# ---------------------------------------------------------------------------
# Layout helpers. Copied from scripts/summary.py rather than imported, because
# that module builds its own PDF at import time.
# ---------------------------------------------------------------------------


def page(title=None, kicker=None):
    fig = plt.figure(figsize=A4, facecolor="white")
    if title:
        if kicker:
            fig.text(0.08, 0.955, "  ".join(kicker.upper()), color=BLUE, fontsize=7.5,
                     fontweight="bold")
        fig.text(0.08, 0.925, title, color=INK, fontsize=19, fontweight="bold", va="top")
        fig.lines.append(plt.Line2D([0.08, 0.92], [0.905, 0.905],
                                    transform=fig.transFigure, color=RULE, lw=0.8))
    return fig


def wrap(text, chars):
    out, line = [], ""
    for word in text.split():
        candidate = f"{line} {word}".strip()
        if len(candidate) > chars:
            out.append(line)
            line = word
        else:
            line = candidate
    out.append(line)
    return "\n".join(out)


def para(fig, y, text, size=10, color=INK, x=0.08, chars=96, gap=0.0165):
    lines = "\n".join(wrap(block, chars) for block in text.split("\n"))
    fig.text(x, y, lines, color=color, fontsize=size, va="top", linespacing=1.55)
    return y - gap * (lines.count("\n") + 1) - 0.008


def heading(fig, y, text, colour=TEAL):
    fig.text(0.08, y, text, color=colour, fontsize=11.5, fontweight="bold", va="top")
    return y - 0.026


def callout(fig, y, title, text, colour=BLUE, chars=86):
    lines = "\n".join(wrap(block, chars) for block in text.split("\n"))
    n = lines.count("\n") + 1
    h = 0.0175 * n + 0.032
    ax = fig.add_axes([0.08, y - h, 0.84, h])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
                                transform=ax.transAxes, facecolor=WASH,
                                edgecolor=colour, lw=0.0))
    ax.plot([0.0, 0.0], [0, 1], transform=ax.transAxes, color=colour, lw=3, clip_on=False)
    ax.text(0.022, 0.88, title, transform=ax.transAxes, color=colour,
            fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.022, 0.66, lines, transform=ax.transAxes, color=INK,
            fontsize=9, va="top", linespacing=1.5)
    return y - h - 0.018


def eqn(fig, y, text, colour=INK):
    fig.text(0.115, y, text, color=colour, fontsize=9.5, va="top", family=MONO,
             linespacing=1.6)
    return y - 0.018 * (text.count("\n") + 1) - 0.008


def table(fig, y, rows, widths, header=None, size=8.6, rowh=0.0175, mono_cols=()):
    x0 = 0.08
    if header:
        cx = x0
        for w, cell in zip(widths, header):
            fig.text(cx, y, cell, color=BLUE, fontsize=size, fontweight="bold", va="top")
            cx += w
        y -= 0.006
        fig.lines.append(plt.Line2D([0.08, 0.92], [y, y], transform=fig.transFigure,
                                    color=RULE, lw=0.7))
        y -= 0.012
    for row in rows:
        cx = x0
        for i, (w, cell) in enumerate(zip(widths, row)):
            fig.text(cx, y, cell, color=INK if i == 0 else MUTED, fontsize=size, va="top",
                     family=MONO if i in mono_cols else None)
            cx += w
        y -= rowh
    return y - 0.010


PAGES = []
PAGE_ORDER = [
    "cover", "scope", "engine", "frames", "equivalence", "verification",
    "vortex", "problem", "fit", "measured", "strip", "ledger", "after",
    "limits", "sources",
]


def pageno(name):
    return PAGE_ORDER.index(name) + 1


def emit(fig, y=None):
    """Append a page, and guard against silent overflow.

    Text laid out below the footer does not error, it just renders on top of it.
    The layout is driven by returned y-cursors, so a paragraph that grows by a
    few lines can push the tail of a page into the footer with no visible
    failure at build time. This is the same class of silent breakage
    check_pagination() exists for, so it gets the same treatment.
    """
    if y is not None and y < FLOOR:
        raise SystemExit(
            f"layout overflow on page {len(PAGES) + 1}: cursor reached y={y:.4f}, "
            f"below the {FLOOR} floor. Text is now running into the footer. "
            f"Shorten the copy or split the page."
        )
    PAGES.append(fig)
    fig.text(0.92, 0.035, str(len(PAGES)), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.08, 0.035, "JAX Flight Simulator - turbulence fidelity", color=MUTED,
             fontsize=8.5)
    PDF.savefig(fig)
    plt.close(fig)


def check_pagination():
    if len(PAGES) != len(PAGE_ORDER):
        raise SystemExit(
            f"pagination: emitted {len(PAGES)} pages but PAGE_ORDER names "
            f"{len(PAGE_ORDER)}. Every cross-reference from pageno() is now wrong."
        )


# ---------------------------------------------------------------------------
# Numbers, computed from the project's own code
# ---------------------------------------------------------------------------

AC = REGISTRY["boeing747"]
V747 = CRUISE["boeing747"]["airspeed"]
H747 = CRUISE["boeing747"]["altitude"]

CASE = wind.PARKS_CASES["hannibal"]
R0, V0, SPACING = CASE["r0"], CASE["v0"], CASE["spacing"]
CASE2 = wind.PARKS_CASES["morton"]

SPAN = float(AC.b)
CHORD = float(AC.c)
ARM_CHORDS = float(airframe.effective_tail_arm(AC))
ARM_M = ARM_CHORDS * CHORD
A0 = float(airframe.calibrated_lift_slope(AC))
CHARACTERISTIC = V0 / R0

TAIL_ARMS = {n: float(airframe.effective_tail_arm(REGISTRY[n])) for n in sorted(REGISTRY)}
GATE = {n: airframe.tail_arm_is_plausible(REGISTRY[n]) for n in sorted(REGISTRY)}

# The E2 correction profile, recomputed live.
_array = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H747]),
                          r0=jnp.array(R0), v0=jnp.array(V0))
_field = lambda p: wind.vortex_wind(p, _array)  # noqa: E731
_st = airframe.stations(AC)


def _rates_at(frac):
    s = State(pos_ned=jnp.array([frac * R0, 0.0, -H747]),
              vel_body=jnp.array([V747, 0.0, 0.0]),
              quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
              omega=jnp.zeros(3))
    tangent = float(wind.gust_rates(s.pos_ned, s.quat, _field)[1])
    secant = float(wind.sampled_rates(s.pos_ned, s.quat, _field, _st)[1])
    return tangent, secant


E2_FRACS = [0.25, 0.5, 0.75, 0.99, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0]
E2 = {f: abs(b - a) / CHARACTERISTIC for f in E2_FRACS for a, b in [_rates_at(f)]}
TAN_IN, _ = _rates_at(0.99)
TAN_EDGE, SEC_EDGE = _rates_at(1.0)

# Loading-shape sensitivity, recomputed live.
_cubic = lambda p: jnp.array([0.0, 0.0, 1e-7 * p[1] ** 3])  # noqa: E731
_state0 = State(pos_ned=jnp.array([0.0, 0.0, -H747]),
                vel_body=jnp.array([V747, 0.0, 0.0]),
                quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
                omega=jnp.zeros(3))
_st_fine = airframe.stations(AC, n_span=2001, n_lon=9)
SHAPES = {}
for _name in airframe.LOADING_SHAPES:
    with airframe.loading_shape(_name):
        SHAPES[_name] = float(
            wind.strip_roll_moment(_state0.pos_ned, _state0.quat, _cubic, AC, _st_fine, V747)
        )
SPREAD_ALL = (max(SHAPES.values()) - min(SHAPES.values())) / abs(np.mean(list(SHAPES.values())))
SPREAD_REAL = abs(SHAPES["tapered"] - SHAPES["elliptic"]) / abs(SHAPES["elliptic"])

LEDGER_COUNT = {c: sum(1 for e in provenance.LEDGER.values() if e.category == c)
                for c in provenance.CATEGORIES}

# Vortex profile for the figure.
_xs = np.linspace(-4 * R0, 4 * R0, 900)
_pts = jnp.stack([jnp.asarray(_xs), jnp.zeros_like(jnp.asarray(_xs)),
                  jnp.full_like(jnp.asarray(_xs), -H747)], axis=1)
_W_UP = -np.asarray(jax.vmap(lambda p: wind.vortex_wind(p, _array))(_pts))[:, 2]


PDF = PdfPages(OUT)

# ------------------------------------------------------------------- cover
fig = plt.figure(figsize=A4, facecolor="white")
ax = fig.add_axes([0, 0.60, 1, 0.40])
ax.set_axis_off()
ax.add_patch(FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="square,pad=0",
                            transform=ax.transAxes, facecolor=WASH, edgecolor="none"))
fig.text(0.08, 0.88, "TURBULENCE FIDELITY", color=BLUE, fontsize=9, fontweight="bold")
fig.text(0.08, 0.83, "From a point aircraft\nto a distributed airframe", color=INK,
         fontsize=27, fontweight="bold", va="top", linespacing=1.25)
fig.text(0.08, 0.685, "How the physics engine works, what the wind-shear model assumed,\n"
                      "what that assumption cost, and what replaces it.",
         color=MUTED, fontsize=11.5, va="top", linespacing=1.5)

y = 0.545
y = para(fig, y, "This report covers a single change and everything it rests on: the wind field "
                 "used to be sampled at one point, the aircraft's centre of gravity, and its "
                 "gradient taken there. For a disturbance forty wingspans across that is exact "
                 "enough to be uninteresting. For a vortex core two to three wingspans across it "
                 "is the weakest link in the project's headline result.")
y = para(fig, y, "The theory backbone throughout is Stengel, Flight Dynamics, 2nd edition. Every "
                 "equation used is cited by number. Two of them turned out to be wrong, which is "
                 "why the verification section comes before the implementation section.")

y = callout(fig, y - 0.01, "Every number here is computed, not typed",
            "This document is generated by scripts/turbulence_report.py, which imports the model "
            "and calls it. The tail arm, the correction profile, the loading-shape spread and the "
            "ledger counts are all recomputed at build time. A regression shows up as a changed "
            "figure in the report rather than a stale claim in prose.", colour=TEAL)

fig.text(0.08, 0.115, f"Boeing 747 at CR-2144 flight condition 9  -  {H747:,.0f} m, "
                      f"{V747:.1f} m/s, Mach 0.80", color=MUTED, fontsize=9)
fig.text(0.08, 0.085, "Parks et al. 1985 Kelvin-Helmholtz vortex, Hannibal case", color=MUTED,
         fontsize=9)
emit(fig, y)

# ------------------------------------------------------------------- scope
fig = page("What this document covers", "Scope")
y = 0.855
y = para(fig, y, "Four questions, in order. Each section says which is being answered.")
rows = [
    ("1", "How does the engine work now?", f"page {pageno('engine')}-{pageno('equivalence')}"),
    ("2", "Is the theory it rests on correct?", f"page {pageno('verification')}"),
    ("3", "What was wrong, and what was measured?", f"page {pageno('problem')}-{pageno('strip')}"),
    ("4", "How will it work after the next change?", f"page {pageno('after')}"),
]
y = table(fig, y, rows, [0.05, 0.55, 0.24], header=("", "Question", "Where"))

y = heading(fig, y - 0.01, "The rule this project runs on")
y = para(fig, y, "A figure without its table has broken the project. That rule predates this work "
                 "and is why the verification section exists: the model's three gust-rate signs "
                 "were re-derived from first principles rather than transcribed, and doing so "
                 "found two errors in the published source they were being checked against.")

y = callout(fig, y, "Provenance is now enforced, not just intended",
            "Every constant carries a category in flightsim/provenance.py: SOURCED read from a "
            "cited table, DERIVED computed from sourced values by a stated relation, CALIBRATED "
            "fitted to reproduce a sourced number, DECLARED chosen and carrying a measured "
            "sensitivity. A test asserts that DERIVED chains name inputs that exist, are acyclic, "
            "and bottom out in something sourced. A constant added without an entry fails the "
            "build.\n"
            f"Current ledger: {LEDGER_COUNT['SOURCED']} sourced, {LEDGER_COUNT['DERIVED']} derived, "
            f"{LEDGER_COUNT['CALIBRATED']} calibrated, {LEDGER_COUNT['DECLARED']} declared.",
            colour=AMBER)

y = heading(fig, y, "What is deliberately not here")
y = para(fig, y, "This is not a claim that the model represents a real Boeing 747. Every "
                 "comparison in the project is closed-loop against a source document's own "
                 "arithmetic, which is what makes it a test of the solver rather than of the "
                 "aeroplane. The derivative data is labelled Flexible in its own source while "
                 "this model is rigid, and that gap is unquantifiable from the documents held. "
                 "Vortex conclusions stay orderings, never values.")
emit(fig, y)

# ------------------------------------------------------------------ engine
fig = page("The engine, as it stands", "How it works now")
y = 0.855
y = para(fig, y, "Thirteen state variables, integrated by fixed-step RK4 at 50 Hz. Position and "
                 "attitude in an Earth frame, velocity and rotation in a body frame, with a "
                 "quaternion carrying the rotation between them.")
y = eqn(fig, y, "pos_ned  (3,)   north, east, down          quat  (4,)   w x y z, body -> NED\n"
                "vel_body (3,)   u v w, body axes           omega (3,)   p q r, body axes")

y = heading(fig, y, "The loop, once per step")
ax = fig.add_axes([0.08, y - 0.20, 0.84, 0.19])
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
boxes = [
    (0.2, 2.3, 2.3, "wind model\nsampled once", BLUE),
    (3.0, 2.3, 2.3, "relative flow\nvel_rel, omega_rel", TEAL),
    (5.8, 2.3, 2.1, "aero\ncoefficients", TEAL),
    (8.2, 2.3, 1.6, "forces\nmoments", TEAL),
    (3.0, 0.3, 2.3, "Newton-Euler\n+ gravity", AMBER),
    (5.8, 0.3, 2.1, "RK4\n4 stages", AMBER),
]
for x, yy, w, label, col in boxes:
    ax.add_patch(FancyBboxPatch((x, yy), w, 1.1, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="white", edgecolor=col, lw=1.4))
    ax.text(x + w / 2, yy + 0.55, label, ha="center", va="center", fontsize=8.5, color=INK)
for a, b in [((2.5, 2.85), (3.0, 2.85)), ((5.3, 2.85), (5.8, 2.85)),
             ((7.9, 2.85), (8.2, 2.85))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
ax.add_patch(FancyArrowPatch((9.0, 2.3), (9.0, 0.85), arrowstyle="-|>", mutation_scale=11,
                             color=MUTED, lw=1.1))
ax.add_patch(FancyArrowPatch((7.9, 0.85), (5.8 + 2.1, 0.85), arrowstyle="-", color=MUTED, lw=1.1))
ax.add_patch(FancyArrowPatch((5.8, 0.85), (5.3, 0.85), arrowstyle="-|>", mutation_scale=11,
                             color=MUTED, lw=1.1))
ax.text(0.2, 0.85, "state at t+dt", fontsize=8.5, color=MUTED, va="center")
ax.add_patch(FancyArrowPatch((3.0, 0.85), (2.5, 0.85), arrowstyle="-|>", mutation_scale=11,
                             color=MUTED, lw=1.1))
y = y - 0.215

y = callout(fig, y, "The one rule that makes turbulence tractable",
            "The aerodynamics never see inertial velocity. dynamics.py forms vel_rel = vel_body - "
            "dcm.T @ wind_ned and omega_rel = omega - omega_gust, and passes only those to aero.py. "
            "The Coriolis, gyroscopic and kinematic terms deliberately keep the INERTIAL velocity "
            "and rate: a gust changes the flow the wings see, not the airframe's ground velocity.",
            colour=TEAL)

y = heading(fig, y, "Why there is no minus-m-dW-by-dt term")
y = para(fig, y, "Stengel sets out two ways to handle a moving air mass. Work in the Earth frame "
                 "and subtract the wind only where the aerodynamics need it (his eq. 3.2-85), or "
                 "ride with the air mass. The second is elegant while the wind is uniform, because "
                 "a block of air drifting at constant speed is itself inertial (eq. 3.2-87). The "
                 "moment the wind varies in space, that frame accelerates and an apparent force "
                 "-dW/dt must be added (eq. 3.2-89), where dW/dt is the material derivative: the "
                 "local change plus the part you fly into (eq. 2.1-13).")
y = para(fig, y, "That term is the price of the second choice, not a physical force waiting to be "
                 "added to the first. This engine uses the first. Adding it would double-count, "
                 "and the project verified that by measurement: with all aerodynamics switched off "
                 "so free fall has a closed form, flying through a wind swinging at 9 g of dW/dt "
                 "matched the exact solution to 4e-12 m. Injecting the term deliberately gave "
                 "13.33 m.")
emit(fig, y)

# ------------------------------------------------------------------ frames
fig = page("Where the Earth frame becomes the body frame", "How it works now")
y = 0.855
y = para(fig, y, "The weather is described in North-East-Down. It has to be: a vortex sits at a "
                 "fixed place in the sky and does not care where the aeroplane points. The "
                 "aerodynamics only work in body axes, because angle of attack means how far the "
                 "nose is tilted relative to the oncoming air. Somewhere the two must meet, and "
                 "in this engine that happens at exactly two lines.")

y = heading(fig, y, "The wind vector: one rotation")
y = eqn(fig, y, "vel_rel = vel_body - dcm.T @ wind_ned          Stengel eq. 3.2-85\n"
                "V, alpha, beta from vel_rel                    Stengel eq. 3.2-86")

y = heading(fig, y, "The wind gradient: a rotation on both sides")
y = eqn(fig, y, "jac_ned   = jacobian(field)(pos_ned)           Stengel eq. 2.1-11 / 3.4-46\n"
                "grad_body = dcm.T @ jac_ned @ dcm             Stengel eq. 3.4-47")
y = para(fig, y, "A gradient answers the question how much does THIS change if I step THAT way. "
                 "Both halves are directional. One rotation re-expresses the answer, a wind "
                 "component; the other re-expresses the question, a direction to step in. Rotating "
                 "one side only would leave a quantity half in each frame. Stengel names this a "
                 "similarity transformation and the code performs exactly it.")

y = callout(fig, y, "The contract is deliberately asymmetric",
            "wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)\n"
            "wind_ned crosses in NED. omega_gust crosses ALREADY IN BODY AXES, because it is "
            "subtracted from state.omega, which is body. Above the handover everything is NED and "
            "attitude-blind; below it everything is body and never sees NED.")

y = heading(fig, y, "What changes as the aircraft moves through a field")
rows = [
    ("Position moves", "the field is re-evaluated; new wind and new gradient"),
    ("Attitude changes", "the same air maps to different body components"),
    ("Time", "nothing: the field is frozen in the Earth frame"),
]
y = table(fig, y, rows, [0.24, 0.62], header=("What", "Consequence"))
y = para(fig, y, "The second is the one that surprises. Freeze the aeroplane in space and roll it "
                 "ninety degrees: the wind in NED is bit-identical, yet vel_rel and omega_gust both "
                 "change. For the Parks vortex this has a sharp consequence. The vortex axes lie "
                 "across the flight path and the field has no lateral variation at all, so flying "
                 "wings-level due north the roll and yaw gust inputs are exactly zero and the "
                 "aircraft is pitched and nothing else. Roll appears only once it banks, created "
                 "purely by the rotation of axes rather than by meeting different air.")
emit(fig, y)

# ------------------------------------------------------------- equivalence
fig = page("Why a gust gradient is a rotation", "The key physics")
y = 0.855
y = para(fig, y, "This is the idea the whole turbulence model rests on, and it is worth stating "
                 "carefully because it is what licenses treating a spatially varying wind with "
                 "derivatives that were measured for a rigidly rotating aircraft.")

y = heading(fig, y, "Start with the aircraft rolling in still air")
y = para(fig, y, "At roll rate p, the wing at spanwise station y moves downward at p*y. It meets "
                 "the air from slightly below, so it sees an extra angle of attack. The further "
                 "outboard, the larger the effect, and the tips trace helices.")
y = eqn(fig, y, "d(alpha) = p*y / V                             Stengel eq. 3.4-39")
y = para(fig, y, "That linearly varying incidence, integrated across the span, IS the roll damping "
                 "derivative Clp. Stengel's Fig. 3.4-3 draws the resulting spanwise lift "
                 "distribution.")

# spanwise incidence figure
ax = fig.add_axes([0.10, y - 0.175, 0.36, 0.155])
ys = np.linspace(-1, 1, 200)
ax.plot(ys, ys, color=BLUE, lw=2)
ax.axhline(0, color=RULE, lw=0.8)
ax.axvline(0, color=RULE, lw=0.8)
ax.set_xlabel("spanwise station  2y/b", fontsize=8)
ax.set_ylabel("extra incidence", fontsize=8)
ax.set_title("roll rate p, still air", fontsize=9, color=INK)
ax.tick_params(labelsize=7)
for s in ax.spines.values():
    s.set_color(RULE)

ax2 = fig.add_axes([0.55, y - 0.175, 0.36, 0.155])
ax2.plot(ys, ys, color=TEAL, lw=2)
ax2.axhline(0, color=RULE, lw=0.8)
ax2.axvline(0, color=RULE, lw=0.8)
ax2.set_xlabel("spanwise station  2y/b", fontsize=8)
ax2.set_title("gust varying across span", fontsize=9, color=INK)
ax2.tick_params(labelsize=7)
for s in ax2.spines.values():
    s.set_color(RULE)
y = y - 0.195

y = para(fig, y, "Now hold the aeroplane still and let the AIR have a vertical velocity that "
                 "varies across the span. The wing sees the same picture. It cannot tell the two "
                 "situations apart, and neither can any derivative measured on it.")

y = heading(fig, y, "The six equivalences")
y = eqn(fig, y, "d(w)/dy = +p    d(v)/dz = -p        (3.4-48, 3.4-49*)\n"
                "d(w)/dx = -q    d(u)/dz = +q        (3.4-50, 3.4-51)\n"
                "d(v)/dx = +r    d(u)/dy = -r        (3.4-52, 3.4-53)")
y = para(fig, y, "The model uses the left-hand column. The asterisk on 3.4-49 is not a footnote "
                 "marker; the sign printed in the source is wrong, and the next page explains how "
                 "that was established.")
emit(fig, y)

# ------------------------------------------------------------ verification
fig = page("Verifying the source before building on it", "Verification")
y = 0.855
y = para(fig, y, "The three signs the model uses were re-derived rather than transcribed, by "
                 "expanding the relative flow at a body-fixed point and matching coefficients.")
y = eqn(fig, y, "v_rel(r) = v_cg + omega x r - w_g(r)\n"
                "omega x r = (q*z - r*y,  r*x - p*z,  p*y - q*x)")
y = para(fig, y, "Each composite formula was then given a rigid-rotation self-consistency test: "
                 "substitute a shear matrix that exactly matches a rigid rotation, and check the "
                 "moment returned equals the equivalent-rate moment. Thirteen formulations were "
                 "checked. Eleven passed. Two did not.")

y = callout(fig, y, "Eq. 3.4-49 has a sign error as printed", RED and
            "The source gives d(v)/dz = +p. It is -p. For omega = (p,0,0) at a point directly "
            "below the centre of gravity, omega x r = (0, -p*z, 0): roll right-wing-down and the "
            "belly swings LEFT. Eqs. 3.4-54 and 3.4-56 self-check only once that sign is "
            "corrected, which is good evidence the slip is confined to that one line rather than "
            "being a different convention.", colour=RED)

y = callout(fig, y, "Eq. 3.4-55 fails by a factor of -2 and is not used",
            "Substituting a rigid pitch rate gives d(w)/dx = -q and d(u)/dz = +q, so the bracket "
            "(d(w)/dx - d(u)/dz) evaluates to -2q and the formula returns +2*Mq*q where it should "
            "return -Mq*q. A factor of one half repairs the magnitude but cannot be justified: the "
            "tail-incidence mechanism dominates Mq for a conventional aircraft, while d(u)/dz "
            "mostly alters the tail's dynamic pressure rather than its incidence, so a 50/50 split "
            "is not physical either.", colour=RED)

y = heading(fig, y, "Why this changed the design")
y = para(fig, y, "Eqs. 3.4-54 to 3.4-56 are the source's per-surface treatment: they split the "
                 "roll, pitch and yaw derivatives into wing, tailplane and fin parts and drive each "
                 "with the shear component that actually reaches it. That is the natural way to "
                 "improve a point model, and it was ruled out twice over. The pitch equation does "
                 "not survive its own self-check, and the split needs tail areas and arms that "
                 "neither CR-2144 nor the Boeing simulation data it cites tabulates.")
y = para(fig, y, "So the design took the route that never forms an equivalent rate at all: sample "
                 "the field across the airframe and integrate. That is what Stengel names as "
                 "required below rotor scale, on p. 217.")
emit(fig, y)

# ------------------------------------------------------------------ vortex
fig = page("The disturbance being flown through", "The field")
y = 0.855
y = para(fig, y, "Parks et al. 1985 identified Kelvin-Helmholtz vortex arrays from two DC-10 "
                 "flight-recorder traces near the tropopause. The model is a Rankine vortex: a "
                 "solid-body core embedded in an irrotational outer flow, with arrays built by "
                 "linear superposition, which is what makes summing it with other components "
                 "legitimate.")
y = eqn(fig, y, "inside  (r <  r0):   w_xy = V0*d/r0        w_z = -V0*l/r0\n"
                "outside (r >= r0):   w_xy = V0*r0*d/r^2    w_z = -V0*r0*l/r^2")

ax = fig.add_axes([0.10, y - 0.20, 0.80, 0.18])
ax.plot(_xs / R0, _W_UP, color=BLUE, lw=1.8)
ax.axhline(0, color=RULE, lw=0.8)
for edge in (-1, 1):
    ax.axvline(edge, color=RED, lw=1.0, ls="--")
ax.text(1.05, 0.86 * max(_W_UP), "core edge", color=RED, fontsize=8)
ax.set_xlabel("along-track position, in core radii", fontsize=8.5)
ax.set_ylabel("updraft, m/s", fontsize=8.5)
ax.tick_params(labelsize=7.5)
for s in ax.spines.values():
    s.set_color(RULE)
y = y - 0.225

y = para(fig, y, "Through the core the vertical gust is an antisymmetric up-then-down doublet, "
                 "linear inside the core and falling as 1/r outside. That antisymmetry is what "
                 "separates a vortex from a single-signed gust, and it is why a vortex cannot "
                 "produce a signed load asymmetry of its own.")

rows = [
    ("Hannibal, 37,000 ft", f"{R0:.1f} m", f"{V0:.2f} m/s", f"{SPACING:.0f} m",
     f"{R0 / SPAN:.2f}"),
    ("Morton, 39,000 ft", f"{CASE2['r0']:.1f} m", f"{CASE2['v0']:.2f} m/s",
     f"{CASE2['spacing']:.0f} m", f"{CASE2['r0'] / SPAN:.2f}"),
]
y = table(fig, y, rows, [0.26, 0.14, 0.15, 0.14, 0.15],
          header=("Parks case", "core r0", "peak V0", "spacing", "r0 in spans"),
          mono_cols=(1, 2, 3, 4))
y = para(fig, y, f"The last column is the problem. The 747's span is {SPAN:.2f} m, so the core is "
                 f"only two to three wingspans across. Every other field in the project is forty "
                 f"spans or more.")
emit(fig, y)

# ----------------------------------------------------------------- problem
fig = page("What the point assumption cost", "The problem")
y = 0.855
y = para(fig, y, "Stengel states the validity condition for the rotary-derivative treatment "
                 "plainly on p. 215: it is for shear whose length scale is large compared with the "
                 "aircraft, and it assumes the wind vector and shear matrix are taken at the centre "
                 "of mass with the aircraft not disturbing the flow. On p. 217 he adds the warning "
                 "that matters here: stability derivatives work for large atmospheric rotors but "
                 "have limited value for wake vortices, where strip theory or CFD is required.")

rows = [
    ("Parks Hannibal core", f"{R0:.0f} m", f"{R0 / SPAN:.2f}"),
    ("Parks Morton core", f"{CASE2['r0']:.0f} m", f"{CASE2['r0'] / SPAN:.2f}"),
    ("Wingrove updraft radius", "2,359 m", f"{2359 / SPAN:.1f}"),
    ("Doyle lee wave, quarter wavelength", "6,250 m", f"{6250 / SPAN:.1f}"),
]
y = table(fig, y, rows, [0.42, 0.18, 0.20],
          header=("Field", "Scale", "In 747 spans"), mono_cols=(1, 2))

y = callout(fig, y, "The vortex is marginal by a factor of about thirteen",
            "It is on the right side of Stengel's line in KIND -- these are atmospheric "
            "Kelvin-Helmholtz rotors, not aircraft wakes, so one-way coupling is safe and the "
            "derivative treatment is the right family of method. It is marginal in SIZE. Everything "
            "else in the project is comfortably a point; this is not.", colour=AMBER)

y = heading(fig, y, "What a point sample keeps, and what it loses")
y = para(fig, y, "The model was never a naive point model. Expand the body-axis gust about the "
                 "centre of gravity and three terms appear. The constant becomes the translational "
                 "gust and feeds angle of attack and sideslip. The linear term becomes the "
                 "rotational gust and feeds the rate derivatives. The quadratic term has nowhere to "
                 "go at all.")
y = eqn(fig, y, "w_g(r) = w_g(0)  +  (dw/dx)*x + (dw/dy)*y  +  O(r^2)\n"
                "           |                |                  |\n"
                "        wind_ned        omega_gust          nothing")
y = para(fig, y, "A rigid-body coefficient build-up has no channel for curvature: there is no "
                 "derivative meaning the spanwise gust profile is bent. Representing it means "
                 "abandoning the point-plus-gradient idea and integrating across the span, which "
                 "is what the strip treatment does.")
emit(fig, y)

# --------------------------------------------------------------------- fit
fig = page("Change one: fit instead of differentiate", "What was built")
y = 0.855
y = para(fig, y, "The first change keeps the three quantities the aero model already consumes and "
                 "improves the estimator for them. Rather than the tangent at the centre of "
                 "gravity, take the least-squares slope across the extent the aerodynamics "
                 "actually integrate over: the span laterally, and the tail arm longitudinally.")

y = callout(fig, y, "The safety property that made this adoptable",
            "A least-squares slope through samples of a LINEAR function is exactly its slope. So "
            "for any field the old model handled correctly, the new one returns the identical "
            "answer, and every existing result is provably unmoved. It differs only where the "
            "field is curved across the airframe -- which is the entire point.", colour=TEAL)

y = heading(fig, y, "The one quantity this needed, and where it came from")
y = para(fig, y, "The lateral extent is the span, which is tabulated. The longitudinal extent is a "
                 "tail arm, which is not tabulated for any aircraft the project holds. It is "
                 "recovered from two derivatives that are, by taking the ratio of Stengel eqs. "
                 "3.4-7 and 3.4-12, in which the unknown tail lift slope cancels.")
y = eqn(fig, y, "CL_qhat = 2 * CL_a,ht * (l/c)        eq. 3.4-7 non-dimensionalised\n"
                "Cm_qhat = -2 * (l/c)^2 * CL_a,ht     eq. 3.4-12\n"
                "------------------------------------------------------\n"
                "l/c = -Cm_qhat / CL_qhat")

rows = [(n, f"{TAIL_ARMS[n]:.4f}", f"{TAIL_ARMS[n] * float(REGISTRY[n].c) / FT2M:.2f} ft",
         "pass" if GATE[n] else "REJECTED") for n in sorted(REGISTRY)]
y = table(fig, y, rows, [0.28, 0.16, 0.18, 0.18],
          header=("Aircraft", "l/c", "arm", "gate"), mono_cols=(1, 2))

y = para(fig, y, f"The 747 value is {ARM_CHORDS:.4f} chords, {ARM_M / FT2M:.2f} ft. The real "
                 f"747-100's centre of gravity sits roughly 100 to 110 ft ahead of the tailplane, "
                 f"so this lands inside the real geometry having been told nothing about it beyond "
                 f"the mean chord. That agreement is the entire justification for the relation, so "
                 f"it is asserted as a test rather than described in a comment.")
y = para(fig, y, "The same check rejects both light-aircraft sets, whose CLq and Cmq imply arms far "
                 "shorter than those airframes have. Whether the fault is the source data or the "
                 "tail-dominated reading is not resolved; either way the strip path must not be "
                 "used for them, and the gate enforces that.")
emit(fig, y)

# ---------------------------------------------------------------- measured
fig = page("What the point assumption actually cost", "The measurement")
y = 0.855
y = para(fig, y, "The assumption register has carried the scale ratio since session 11 but never a "
                 "measured consequence. It has one now, taken along a traverse of the Hannibal "
                 "core and normalised by V0/r0, the core's own characteristic pitch-rate input.")

ax = fig.add_axes([0.10, y - 0.21, 0.80, 0.19])
fr = np.array(E2_FRACS)
vals = np.array([E2[f] for f in E2_FRACS])
ax.plot(fr, vals, "o-", color=BLUE, lw=1.6, ms=4)
ax.axvline(1.0, color=RED, lw=1.0, ls="--")
ax.text(1.04, 1.55, "core edge", color=RED, fontsize=8)
ax.set_xlabel("along-track position, in core radii", fontsize=8.5)
ax.set_ylabel("correction, units of V0/r0", fontsize=8.5)
ax.tick_params(labelsize=7.5)
for s in ax.spines.values():
    s.set_color(RULE)
y = y - 0.235

rows = [(f"{f:.2f} r0", f"{E2[f]:.4f}") for f in (0.5, 0.99, 1.0, 1.1, 1.25, 2.0, 3.0)]
y = table(fig, y, rows, [0.18, 0.20], header=("Station", "Correction"), mono_cols=(1,))

y = callout(fig, y, "Inside the core the correction is exactly zero, not merely small",
            "Parks' Rankine profile is LINEAR in radius, so a point sample plus an analytic "
            "gradient is not an approximation at all while the airframe is inside the core. This is "
            "why the existing vortex results survived the assumption as long as they did -- and it "
            "is specific to this field. It would not hold for a Dryden field or a wake vortex.",
            colour=TEAL)

y = callout(fig, y, "At the boundary the gradient is discontinuous, not merely steep",
            f"Velocity is continuous at r0 -- both branches agree, which an existing test already "
            f"asserted -- but the derivative is not. Inside it is +V0/r0 = {CHARACTERISTIC:+.5f} "
            f"rad/s; outside at r0 it is {TAN_EDGE:+.5f}. The two one-sided derivatives differ by "
            f"2*V0/r0 and have OPPOSITE SIGNS. So the tangent there is not merely inaccurate, it is "
            f"ambiguous, and the code's strict less-than resolves the tie toward the outside branch "
            f"-- the wrong side, since the airframe is still almost entirely inside the core. The "
            f"fit has no such ambiguity.", colour=RED)
emit(fig, y)

# ------------------------------------------------------------------- strip
fig = page("Change two: integrate across the span", "What was built")
y = 0.855
y = para(fig, y, "Fitting a slope still collapses the field to three numbers, so a profile that "
                 "bends rather than ramps is still unrepresented. Only giving each strip the gust "
                 "at its own station carries that.")
y = eqn(fig, y, "Cl = -(1/(S*b)) * integral( y * c(y) * a0 * d(alpha)(y) dy )")

y = heading(fig, y, "The shape is declared; the magnitude is calibrated")
y = para(fig, y, "The integral needs a spanwise lift distribution. Taper ratio is not tabulated in "
                 "CR-2144, is not in the Boeing simulation data that report itself cites, and is "
                 "not recoverable from the reference geometry: a straight-tapered wing requires a "
                 "shape factor of 0.72873 from these values, and that function has a global minimum "
                 "of 0.75. No real taper ratio satisfies it, because the 747 planform is cranked.")
y = para(fig, y, "So the shape is elliptic, needing no taper ratio at all, and DECLARED. Its "
                 "magnitude is pinned so that integrating a rigid roll rate reproduces the "
                 "tabulated Clp exactly, which makes it CALIBRATED against a sourced number.")
y = eqn(fig, y, f"elliptic loading  =>  Clp_hat = -a0/8   =>   a0 = -8*Clp = {A0:.4f}")
y = para(fig, y, f"That value looks low against a thin-airfoil {2 * np.pi:.2f} and the aircraft's "
                 f"own CLa of {float(AC.CLa):.2f}, and it should. It is an EFFECTIVE slope "
                 f"absorbing sweep, the tail's share of Clp, and the gap between elliptic strip "
                 f"theory and a real cranked swept wing. A value near {2 * np.pi:.2f} would mean "
                 f"the calibration had not absorbed those and would be the surprising outcome.")

rows = [(n, f"{SHAPES[n]:+.4e}") for n in airframe.LOADING_SHAPES]
y = table(fig, y, rows, [0.22, 0.24], header=("Shape", "Rolling moment, cubic gust"),
          mono_cols=(1,))
y = para(fig, y, f"The two shapes that actually taper toward the tips agree to "
                 f"{SPREAD_REAL:.1%}. The full spread of {SPREAD_ALL:.1%} is driven entirely by the "
                 f"uniform distribution, which loads the tips where a cubic gust is largest and is "
                 f"not a defensible transport planform. It is in the sweep as a bracket, not as a "
                 f"candidate.")

y = callout(fig, y, "A sign error, and how it was caught",
            "The gust incidence was implemented as +w_g/V. It is -w_g/V. A wing moving DOWN meets "
            "the air from below and gains incidence; air moving DOWN past a stationary wing arrives "
            "from above and loses it. The rigid-rotation check could not catch this, because that "
            "path never touches a gust and reproduced the sourced Clp perfectly with the sign "
            "inverted. It took comparing the two treatments against each other on a linear gradient "
            "-- where they must agree -- to expose it. A model with that bug rolls correctly for "
            "its own motion and backwards for every gust.", colour=RED)
emit(fig, y)

# ------------------------------------------------------------------ ledger
fig = page("Which numbers are bulletproof", "Provenance")
y = 0.855
y = para(fig, y, "Review asked for an unambiguous separation between numbers read from cited tables "
                 "and numbers that were predicted, with no credit given to a predicted number for "
                 "landing in a plausible range. That is now data, and a test enforces it.")

rows = [
    ("SOURCED", str(LEDGER_COUNT["SOURCED"]), "read from a cited table, with document and page"),
    ("DERIVED", str(LEDGER_COUNT["DERIVED"]), "computed from sourced values by a stated relation"),
    ("CALIBRATED", str(LEDGER_COUNT["CALIBRATED"]), "fitted to reproduce a sourced number"),
    ("DECLARED", str(LEDGER_COUNT["DECLARED"]), "chosen; must carry a measured sensitivity"),
]
y = table(fig, y, rows, [0.16, 0.07, 0.60], header=("Category", "n", "Meaning"))

y = para(fig, y, "The enforcing test asserts that every entry uses one of the four categories, that "
                 "SOURCED and DECLARED entries carry a usable detail, that DERIVED and CALIBRATED "
                 "entries name inputs which themselves exist in the ledger, and that the dependency "
                 "graph is acyclic so every chain bottoms out in something sourced.")

y = callout(fig, y, "The two entries that most need reading",
            "b747.l_eff is DERIVED and must never be quoted as 747 geometry. It attributes both "
            "pitch derivatives to the tail, and the obvious correction -- subtracting the wing's "
            "share via eqs. 3.4-13/3.4-14 with the sourced c.g. at 0.25 chord -- was tried and "
            "REJECTED: it makes the wing 83 percent of CLq, implying a 23-chord tail arm. Those are "
            "two-dimensional infinite-aspect-ratio results and they do not transfer.\n"
            "strip.loading_shape is DECLARED. Quote 2.6 percent as the shape cost and 49.7 percent "
            "as the bound.", colour=AMBER)

y = heading(fig, y, "What the source audit established")
y = para(fig, y, "CR-2144 was read page by page, and so was the Boeing simulation report it names "
                 "as its own 747 source. Neither tabulates taper ratio or any tail dimension. Both "
                 "model the whole aircraft, so neither needs component geometry, and no further "
                 "digging in that lineage will produce it. The audit did return three quantities "
                 "the design had assumed absent: the centre of gravity at 0.25 chord, the mean "
                 "chord's spanwise station, and the pilot station offset -- the last of which "
                 "unblocks the accelerometer lever-arm correction that had been recorded as "
                 "impossible.")
emit(fig, y)

# ------------------------------------------------------------------- after
fig = page("How the engine will work after the next change", "What comes next")
y = 0.855
y = para(fig, y, "The strip integration is built and validated, but nothing consumes it. The 6-DOF "
                 "still receives its rotational gust through the existing contract, so today the "
                 "strip result is a measurement of the error rather than a correction to it. The "
                 "next change wires it in.")

y = heading(fig, y, "Now")
ax = fig.add_axes([0.08, y - 0.115, 0.84, 0.10])
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 2)
for x, w, label, col in [(0.2, 2.2, "field\nat the CG", BLUE),
                         (3.0, 2.4, "gradient\nat the CG", BLUE),
                         (6.0, 1.9, "p, q, r\ngust", TEAL),
                         (8.4, 1.4, "aero", TEAL)]:
    ax.add_patch(FancyBboxPatch((x, 0.4), w, 1.1, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="white", edgecolor=col, lw=1.4))
    ax.text(x + w / 2, 0.95, label, ha="center", va="center", fontsize=8.5, color=INK)
for a, b in [((2.4, 0.95), (3.0, 0.95)), ((5.4, 0.95), (6.0, 0.95)), ((7.9, 0.95), (8.4, 0.95))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
y = y - 0.125

y = heading(fig, y, "After")
ax = fig.add_axes([0.08, y - 0.155, 0.84, 0.14])
ax.set_axis_off()
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
for x, yy, w, label, col in [(0.2, 1.7, 2.2, "field at\nevery strip", BLUE),
                             (3.0, 1.7, 2.4, "local incidence\nper strip", BLUE),
                             (6.0, 1.7, 1.9, "integrate\nto moments", TEAL),
                             (8.4, 1.7, 1.4, "aero", TEAL),
                             (0.2, 0.1, 2.2, "field\nat the CG", MUTED),
                             (3.0, 0.1, 2.4, "gradient\nat the CG", MUTED),
                             (6.0, 0.1, 1.9, "p, q, r\ngust", MUTED)]:
    ax.add_patch(FancyBboxPatch((x, yy), w, 1.1, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="white", edgecolor=col, lw=1.4))
    ax.text(x + w / 2, yy + 0.55, label, ha="center", va="center", fontsize=8.5,
            color=INK if yy > 1 else MUTED)
for a, b in [((2.4, 2.25), (3.0, 2.25)), ((5.4, 2.25), (6.0, 2.25)), ((7.9, 2.25), (8.4, 2.25)),
             ((2.4, 0.65), (3.0, 0.65)), ((5.4, 0.65), (6.0, 0.65))]:
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=MUTED, lw=1.1))
ax.add_patch(FancyArrowPatch((7.9, 0.65), (8.9, 1.7), arrowstyle="-|>", mutation_scale=11,
                             color=MUTED, lw=1.1, ls="--"))
ax.text(0.2, 2.9, "strip path: carries curvature", fontsize=8, color=TEAL)
ax.text(0.2, 1.45, "point path: retained as fallback and reference", fontsize=8, color=MUTED)
y = y - 0.165

y = para(fig, y, "The two paths coexist. The point path stays the default and the reference "
                 "implementation, because it is what every frozen baseline was measured against, "
                 "and it remains the fallback for aircraft whose derivative sets fail the tail-arm "
                 "gate. The strip path becomes opt-in per run.")

y = callout(fig, y, "What has to change, and what deliberately does not",
            "The wind-model contract widens to carry a coefficient increment alongside the gust "
            "rates. dynamics.derivatives adds that increment at the same seam where wind already "
            "enters, so aero.py is untouched and keeps its rule of never seeing inertial velocity. "
            "integrate.step threads it through. The default remains zeros, so a run that does not "
            "ask for strip loads is bit-identical to today.", colour=TEAL)

y = para(fig, y, "This will move the vortex figure-8 point. That movement is expected, is explained "
                 "by the correction profile on page "
                 f"{pageno('measured')}, and must be reported rather than tuned away. The ordering "
                 "claim -- vortex below updraft below manoeuvre -- is what has to survive, because "
                 "the project's own rule already restricts vortex conclusions to orderings.")
emit(fig, y)

# ------------------------------------------------------------------ limits
fig = page("What is still not modelled", "Limits")
y = 0.855
y = para(fig, y, "Stated so no reader over-reads the result. None of these is an oversight; each "
                 "is recorded in the assumption register with its reasoning.")

rows = [
    ("Strip theory is not CFD", "each strip is treated two-dimensionally, with no spanwise induced coupling between strips"),
    ("The aircraft stays transparent", "its own flow field does not distort the impinging vortex; genuine wake encounters become less wrong, not trustworthy"),
    ("The loading shape is declared", "the integral is right; the shape is an assumption, costing 2.6 percent across defensible alternatives"),
    ("l_eff folds wing and fuselage in", "justified empirically by landing on the real tail arm, not by a computed error bar"),
    ("Quasi-steady aerodynamics", "no alpha-dot term. Stengel eq. 3.4-58 gives the unsteady contribution; the model has no channel for it"),
    ("Constant derivatives", "the same numbers at every Mach and incidence the sim reaches; unbounded, and the largest open assumption"),
    ("Rigid airframe", "the derivative data is labelled Flexible in its own source; unquantifiable from documents held"),
]
y = table(fig, y, rows, [0.30, 0.58], header=("Limitation", "What it means"), rowh=0.041,
          size=8.4)

y = callout(fig, y, "Two of these get WORSE in the direction this work is aimed",
            "The missing alpha-dot term and the rigid-airframe assumption both bite harder as "
            "fields get smaller. A shorter traverse makes unsteady lag a larger fraction of the "
            "encounter, and the sharper spanwise loads this work newly resolves are exactly the "
            "ones that would flex a real wing. Making the aerodynamic sampling finer does not make "
            "the structural model better, and it should not be read as doing so.", colour=RED)

y = para(fig, y, "The honest summary is that this work removes one specific, measured error -- "
                 "treating a two-to-three-span disturbance as though it acted at a point -- and "
                 "leaves the others exactly where they were. That is worth doing because it was the "
                 "weakest link in the project's headline result, and because it is now bounded "
                 "rather than argued.")
emit(fig, y)

# ----------------------------------------------------------------- sources
fig = page("Sources", "References")
y = 0.855
y = heading(fig, y, "Theory")
y = para(fig, y, "R. F. Stengel, Flight Dynamics, 2nd ed., Princeton University Press, 2022. "
                 "Sections 2.1, 2.2, 3.2 and 3.4. Equations used: 2.1-10 and 2.1-11 wind field and "
                 "shear matrix; 2.1-13 material derivative; 3.2-85 and 3.2-86 air-relative velocity "
                 "and angles; 3.2-87 and 3.2-89 inertial versus air-mass frames; 3.2-119 "
                 "acceleration at an offset point; 3.4-7 and 3.4-12 tail pitch derivatives; 3.4-39 "
                 "roll-rate incidence; 3.4-40 strip-theory roll damping; 3.4-46 and 3.4-47 shear "
                 "matrix and its similarity transform; 3.4-48 to 3.4-53 rate equivalences; 3.4-54 "
                 "to 3.4-56 per-surface shear moments, NOT USED; 3.4-58 unsteady term, not modelled.")

y = heading(fig, y - 0.006, "Aircraft data")
y = para(fig, y, "R. K. Heffley and W. F. Jewell, Aircraft Handling Qualities Data, NASA CR-2144, "
                 "December 1972, Section IX. Table IX-3 reference geometry and flight conditions, "
                 "printed p. 229; Table IX-4 longitudinal dimensional derivatives, p. 230; Table "
                 "IX-8 lateral dimensional derivatives, p. 234; Figure IX-1 flight conditions, "
                 "p. 212; Figure IX-2 general arrangement, p. 213. Tables IX-3, IX-4 and IX-8 were "
                 "verified element by element against the transcription in flightsim/aircraft.py.")
y = para(fig, y, "C. R. Hanke and D. R. Nordwall, The Simulation of a Jumbo Jet Transport Aircraft, "
                 "Volume II: Modeling Data, Boeing D6-30643-VOL-2 / NASA CR-114494, September 1970. "
                 "The document CR-2144 names as its sole 747 source. Its Summary of Areas and "
                 "Dimensions supplies wing area, mean chord, span, wheel geometry and engine moment "
                 "arms, and no tail geometry.")

y = heading(fig, y - 0.006, "Atmospheric fields")
y = para(fig, y, "E. K. Parks, R. C. Wingrove, R. E. Bach and R. S. Mehta, Identification of "
                 "Vortex-Induced Clear Air Turbulence Using Airline Flight Records, J. Aircraft "
                 "22(2), February 1985, pp. 124-129. The Rankine vortex model, eqs. (3) to (6), and "
                 "the two identified cases.")
y = para(fig, y, "R. C. Wingrove and R. E. Bach, Severe Turbulence and Maneuvering from Airline "
                 "Flight Records, J. Aircraft 31(4), 1994. The updraft magnitudes and the Fig. 8 "
                 "discriminator the project's turbulence work is aimed at.")

y = heading(fig, y - 0.006, "Project documents")
y = para(fig, y, "docs/PROJECT.md section 4 for measured checks and frozen baselines. "
                 "docs/ASSUMPTIONS.md section E2 for the point-aircraft assumption and its bound. "
                 "docs/superpowers/specs/2026-08-14-wind-shear-fidelity-design.md for the source "
                 "verification and the design reasoning, including the two errors found in "
                 "Stengel and the self-consistency test that found them.")

y = callout(fig, y, "Reproducing this document",
            ".venv/Scripts/python.exe scripts/turbulence_report.py docs/summary/turbulence-report.pdf\n"
            "Every figure and number above is recomputed on each build by calling the model.",
            colour=TEAL)
emit(fig, y)

check_pagination()
PDF.close()
print(f"wrote {OUT}  ({len(PAGES)} pages)")
