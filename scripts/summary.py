"""Build the plain-English project summary PDF.

Run from the project root:

    .venv/Scripts/python.exe scripts/summary.py         docs/summary/atisim-summary.pdf docs/summary/panel.png

Every number in the document is either taken from a cited source document, taken
from docs/PROJECT.md section 4 (where it was measured against a test), or
computed here by calling the project's own code. Nothing is typed in from
memory.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import jax.numpy as jnp
import atisim  # noqa: F401
from atisim import wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M, RAD2DEG

OUT = Path(sys.argv[1])
PANEL_PNG = Path(sys.argv[2])

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


def page(pdf, title=None, kicker=None):
    fig = plt.figure(figsize=A4, facecolor="white")
    if title:
        if kicker:
            fig.text(0.08, 0.955, "  ".join(kicker.upper()), color=BLUE, fontsize=7.5,
                     fontweight="bold")
        fig.text(0.08, 0.925, title, color=INK, fontsize=19, fontweight="bold", va="top")
        fig.lines.append(plt.Line2D([0.08, 0.92], [0.905, 0.905],
                                    transform=fig.transFigure, color=RULE, lw=0.8))
    return fig


def body(fig, y, text, size=10, color=INK, x=0.08, weight="normal", width=0.84,
         family=None, ha="left"):
    fig.text(x, y, text, color=color, fontsize=size, va="top", ha=ha,
             fontweight=weight, family=family, wrap=False,
             linespacing=1.55)
    return y


def wrap(text, chars):
    """Greedy wrap; matplotlib's own wrap ignores figure-relative widths."""
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
    lines = wrap(text, chars)
    body(fig, y, lines, size=size, color=color, x=x)
    return y - gap * (lines.count("\n") + 1) - 0.008


def callout(fig, y, title, text, colour=BLUE, height=None, chars=88):
    # Wrap each line separately so every deliberate break survives; one greedy
    # wrap over the whole string eats them all.
    lines = "\n".join(wrap(block, chars) for block in text.split("\n"))
    n = lines.count("\n") + 1
    h = height or (0.019 * n + 0.030)
    ax = fig.add_axes([0.08, y - h, 0.84, h])
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.0,rounding_size=0.02",
                                transform=ax.transAxes, facecolor=WASH,
                                edgecolor=colour, lw=0.0))
    ax.plot([0.0, 0.0], [0, 1], transform=ax.transAxes, color=colour, lw=3, clip_on=False)
    ax.text(0.022, 0.86, title, transform=ax.transAxes, color=colour,
            fontsize=9.5, fontweight="bold", va="top")
    ax.text(0.022, 0.60, lines, transform=ax.transAxes, color=INK,
            fontsize=9, va="top", linespacing=1.5)
    return y - h - 0.018


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


def footer(fig, n):
    fig.text(0.92, 0.035, str(n), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.08, 0.035, "AtiSim — project summary", color=MUTED, fontsize=8.5)


PAGES = []

# Page numbers cited in the body text come from here, never typed in. They were
# typed in until session 8, and FOUR of the five were wrong: a page inserted
# anywhere renumbers everything after it, the numbers are prose rather than
# code, and nothing renders an error. check_pagination() is the guard --
# add a page without naming it here and the build fails instead of quietly
# shifting every cross-reference by one.
PAGE_ORDER = [
    "cover", "idea", "physics", "structure", "data", "science", "result",
    "leewave", "bug", "panel", "honesty", "practicalities", "sources",
    "measurements",
]


def pageno(name: str) -> int:
    return PAGE_ORDER.index(name) + 1


def emit(pdf, fig):
    PAGES.append(fig)
    footer(fig, len(PAGES))
    pdf.savefig(fig)
    plt.close(fig)


def check_pagination() -> None:
    if len(PAGES) != len(PAGE_ORDER):
        raise SystemExit(
            f"pagination: emitted {len(PAGES)} pages but PAGE_ORDER names "
            f"{len(PAGE_ORDER)}. Every cross-reference written with pageno() is "
            f"now wrong. Update PAGE_ORDER to match the emission order."
        )


# ---------------------------------------------------------------------------
# Numbers computed from the project's own code, not typed in
# ---------------------------------------------------------------------------

CASE = wind.PARKS_CASES["hannibal"]
R0, V0, SPACING = CASE["r0"], CASE["v0"], CASE["spacing"]
V747 = CRUISE["boeing747"]["airspeed"]
H747 = CRUISE["boeing747"]["altitude"]
AC747 = REGISTRY["boeing747"]

single = wind.VortexArray(north=jnp.array([0.0]), down=jnp.array([-H747]),
                          r0=jnp.array(R0), v0=jnp.array(V0))
xs = np.linspace(-6 * R0, 6 * R0, 1200)
w_up = np.array([-float(wind.vortex_wind(jnp.array([x, 0.0, -H747]), single)[2]) for x in xs])

pair = wind.VortexArray(north=jnp.array([0.0, SPACING]), down=jnp.array([-H747, -H747]),
                        r0=jnp.array(R0), v0=jnp.array(V0))
xs2 = np.linspace(-3 * R0, SPACING + 3 * R0, 1400)
w_up2 = np.array([-float(wind.vortex_wind(jnp.array([x, 0.0, -H747]), pair)[2]) for x in xs2])

with PdfPages(OUT) as pdf:

    # ---------------------------------------------------------------- cover
    fig = plt.figure(figsize=A4, facecolor="white")
    ax = fig.add_axes([0, 0.62, 1, 0.38]); ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                                transform=ax.transAxes, facecolor="#12263a", lw=0))
    ax.text(0.08, 0.70, "AtiSim", transform=ax.transAxes,
            color="white", fontsize=31, fontweight="bold")
    ax.text(0.08, 0.56, "What the project is, what it proves, and what it does not",
            transform=ax.transAxes, color="#9fc0dd", fontsize=13)
    ax.text(0.08, 0.30, "A plain-English summary", transform=ax.transAxes,
            color="#6f93b3", fontsize=10.5)
    ax.text(0.08, 0.20, "7 August 2026", transform=ax.transAxes,
            color="#6f93b3", fontsize=10.5)

    y = 0.55
    y = para(fig, y, "This is a flight simulator written to do one specific job: find out how a "
                     "large aircraft responds when it flies into clear-air turbulence. It is not a "
                     "game and it has no scenery. It is a physics engine with an instrument panel "
                     "bolted on, so that a person can fly the aircraft into a turbulence pattern "
                     "measured from a real incident and watch what happens.", size=10.5)
    y = para(fig, y, "The project has an unusual rule that shapes everything in it: every number "
                     "must come from a published source, and must say which one. Where no source "
                     "exists, the gap is written down rather than filled with a plausible guess. "
                     "That rule is why this document can cite every figure in it.", size=10.5)

    y = callout(fig, y - 0.005, "The two rules the project runs on",
                "1.  Flag, never invent.  Every number carries the table it came from. A parameter "
                "no source supplies is named as a declared modelling choice, not given a plausible "
                "default.\n\n"
                "2.  Never edit a tolerance to make a test pass.  If a validated measurement moves, "
                "something real broke.", chars=92)

    rows = [
        ("Language", "Python 3.10 with JAX (a numerical library that compiles maths to fast code)"),
        ("Size", "About 4,600 lines of code and 260 automated tests"),
        ("Aircraft", "Boeing 747, Piper PA-28-180 Cherokee, Cessna 172"),
        ("Physics", "Six degrees of freedom, rigid body, fixed-step Runge-Kutta integration"),
        ("Purpose", "Clear-air turbulence response, compared against two published papers"),
        ("Status", "Flight model validated; turbulence layer partly built"),
    ]
    y = table(fig, y - 0.005, rows, [0.13, 0.71], header=("", ""), size=9.2, rowh=0.020)

    fig.text(0.08, 0.075, "Every figure in this document is marked with a source tag such as [S1] "
                          "or [M3].\nThe full list is on the last page.",
             color=MUTED, fontsize=8.5, va="top", linespacing=1.5)
    emit(pdf, fig)

    # ------------------------------------------------- what a sim must do
    fig = page(pdf, "What a flight simulator actually has to do", "The idea")
    y = 0.865
    y = para(fig, y, "An aircraft in flight is a solid object being pushed around by three things: "
                     "the air flowing over it, the thrust of its engines, and gravity. If you know "
                     "where it is, how fast it is going and which way it is pointing, you can work "
                     "out those forces. From the forces you get the acceleration. From the "
                     "acceleration you get the next position and attitude, a fraction of a second "
                     "later. Then you do it again.")
    y = para(fig, y, "That loop is the whole simulator. This one runs it fifty times a second.")

    ax = fig.add_axes([0.08, 0.500, 0.84, 0.235]); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.2)
    boxes = [
        (1.15, "State", "where it is,\nhow fast,\nwhich way up", BLUE),
        (3.75, "Forces", "lift, drag,\nsideforce,\nthrust, gravity", TEAL),
        (6.30, "Acceleration", "force / mass\nmoment / inertia", AMBER),
        (8.85, "New state", "1/50 second\nlater", BLUE),
    ]
    HW = 1.05  # half-width of a box
    for x, name, sub, colour in boxes:
        ax.add_patch(FancyBboxPatch((x - HW, 1.35), 2 * HW, 1.05,
                                    boxstyle="round,pad=0.02,rounding_size=0.10",
                                    facecolor="white", edgecolor=colour, lw=1.6))
        ax.text(x, 2.20, name, ha="center", fontsize=10, fontweight="bold", color=colour)
        ax.text(x, 1.94, sub, ha="center", va="top", fontsize=7.6, color=MUTED,
                linespacing=1.45)
    for x, nxt in zip([b[0] for b in boxes], [b[0] for b in boxes[1:]]):
        ax.add_patch(FancyArrowPatch((x + HW + 0.06, 1.875), (nxt - HW - 0.06, 1.875),
                                     arrowstyle="-|>", mutation_scale=12,
                                     color=MUTED, lw=1.3))
    ax.add_patch(FancyArrowPatch((8.85, 1.30), (8.85, 0.72), arrowstyle="-",
                                 mutation_scale=12, color=MUTED, lw=1.3))
    ax.add_patch(FancyArrowPatch((8.85, 0.72), (1.15, 0.72), arrowstyle="-",
                                 mutation_scale=12, color=MUTED, lw=1.3))
    ax.add_patch(FancyArrowPatch((1.15, 0.72), (1.15, 1.30), arrowstyle="-|>",
                                 mutation_scale=12, color=MUTED, lw=1.3))
    ax.text(5.0, 0.45, "repeat, 50 times per second", ha="center", fontsize=8.8,
            color=MUTED, style="italic")
    ax.text(3.75, 2.98, "the wind enters here, and nowhere else", ha="center", fontsize=8.6,
            color=RED, fontweight="bold")
    ax.add_patch(FancyArrowPatch((3.75, 2.88), (3.75, 2.48), arrowstyle="-|>",
                                 mutation_scale=11, color=RED, lw=1.2))

    y = 0.482
    y = para(fig, y, "\"Six degrees of freedom\" is the standard phrase for tracking all the ways a "
                     "rigid body can move: three of position (north, east, down) and three of "
                     "rotation (roll, pitch, yaw). Simpler simulators fix some of these. This one "
                     "does not, which is why an aircraft in it can be upset in a way that couples "
                     "roll into yaw and back, exactly as a real one is.")
    y = callout(fig, y, "Why the wind only enters in one place",
                "Turbulence changes the air the wings are flying through. It does not reach out and "
                "shove the aeroplane. So the wind is subtracted from the aircraft's velocity before "
                "the aerodynamic forces are computed, and is kept out of everything else. Putting "
                "it anywhere else is a classic modelling error, and this project has a test that "
                "fails if anyone does.", colour=RED, chars=92)
    y = para(fig, y, "The step size never changes. A variable step would make a run impossible to "
                     "reproduce exactly, and would leave the accuracy of the answer at the mercy of "
                     "whatever else the computer happened to be doing.")
    emit(pdf, fig)

    # ------------------------------------------------------- axes / forces
    fig = page(pdf, "The forces, and the axes they act in", "The physics")
    ax = fig.add_axes([0.06, 0.545, 0.54, 0.33])
    ax.set_axis_off(); ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75); ax.set_ylim(-1.30, 1.30)
    # Side view, nose to the right.
    fuse = plt.Polygon([[-0.62, -0.10], [-0.58, 0.09], [0.28, 0.11], [0.55, 0.05],
                        [0.60, -0.02], [0.50, -0.10]],
                       closed=True, facecolor="#dfe6ec", edgecolor=INK, lw=1.3)
    ax.add_patch(fuse)
    ax.add_patch(plt.Polygon([[-0.62, 0.06], [-0.72, 0.44], [-0.46, 0.44], [-0.40, 0.09]],
                             closed=True, facecolor="#c9d4de", edgecolor=INK, lw=1.1))
    ax.add_patch(plt.Polygon([[-0.66, 0.05], [-0.94, 0.16], [-0.50, 0.16]],
                             closed=True, facecolor="#c9d4de", edgecolor=INK, lw=1.1))
    ax.add_patch(plt.Polygon([[-0.16, 0.02], [-0.40, -0.20], [0.14, -0.20], [0.10, 0.02]],
                             closed=True, facecolor="#c9d4de", edgecolor=INK, lw=1.1))
    ax.plot([-0.04], [0.0], marker="o", ms=5, color=INK, zorder=5)
    arrows = [
        ((-0.04, 0.14), (-0.04, 1.00), TEAL, "Lift", (0.02, 1.06), "left", "bottom"),
        ((-0.04, -0.24), (-0.04, -1.02), BLUE,
         "Weight\nacts at the centre of gravity", (0.02, -1.06), "left", "top"),
        ((0.66, 0.0), (1.42, 0.0), AMBER, "Thrust", (1.46, 0.0), "left", "center"),
        ((-1.00, 0.0), (-1.62, 0.0), RED, "Drag", (-1.66, 0.0), "right", "center"),
    ]
    for (x0, y0), (x1, y1), colour, label, (lx, ly), ha, va in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, color=colour, lw=2.0))
        ax.text(lx, ly, label, color=colour, fontsize=9.5, fontweight="bold",
                ha=ha, va=va)

    ax2 = fig.add_axes([0.62, 0.545, 0.30, 0.33])
    ax2.set_axis_off(); ax2.set_aspect("equal")
    ax2.set_xlim(-1.15, 1.35); ax2.set_ylim(-1.35, 1.15)
    axes_def = [((0.95, 0.18), "x  forward", "roll about it", "left", "top"),
                ((0.62, -0.62), "y  right wing", "pitch about it", "left", "top"),
                ((0.0, -1.05), "z  down", "yaw about it", "center", "top")]
    for (dx, dy), label, rot, ha, va in axes_def:
        ax2.add_patch(FancyArrowPatch((0, 0), (dx, dy), arrowstyle="-|>",
                                      mutation_scale=12, color=INK, lw=1.5))
        ax2.text(dx * 1.10, dy * 1.10, label, fontsize=8.2, color=INK, ha=ha,
                 va=va, fontweight="bold")
        ax2.text(dx * 1.10, dy * 1.10 - (0.13 if va != "bottom" else -0.13), rot,
                 fontsize=7.2, color=MUTED, ha=ha, va=va)
    ax2.text(0.10, 1.05, "Body axes", fontsize=10, fontweight="bold", ha="center", color=INK)

    y = 0.500
    y = para(fig, y, "Everything is measured in axes fixed to the aircraft: x out of the nose, y "
                     "out of the right wing, z down through the floor. Rotating about those three "
                     "gives roll, pitch and yaw. The aircraft's position is tracked separately in "
                     "north-east-down axes fixed to the ground.")
    y = para(fig, y, "The aerodynamic forces come from a standard build-up: each force and moment "
                     "is a sum of contributions from angle of attack, from the rotation rates, and "
                     "from the control surfaces, each multiplied by a coefficient measured in a "
                     "wind tunnel or in flight. Those coefficients are the aircraft. They are what "
                     "the source documents supply, and they are what makes a Boeing 747 behave like "
                     "a Boeing 747 rather than like a scaled-up Cessna.")
    y = callout(fig, y, "Angle of attack, in one sentence",
                "The angle between where the nose is pointing and where the aircraft is actually "
                "going through the air — it is what produces lift, and a gust changes it "
                "instantly without the aircraft moving at all.", colour=TEAL, chars=92)
    y = para(fig, y, "Orientation is stored as a quaternion, a four-number way of representing a "
                     "rotation. The alternative — three angles — breaks down when the "
                     "aircraft points straight up or straight down. Quaternions do not, and the "
                     "project has a test confirming the representation stays valid over one hundred "
                     "thousand steps [M1].")
    emit(pdf, fig)

    # --------------------------------------------------------- module map
    fig = page(pdf, "How the code is put together", "The structure")
    y = 0.865
    y = para(fig, y, "Fourteen files, each with one job. The arrows show what depends on what: a "
                     "file only ever uses the ones below it, which is what stops a change in the "
                     "instrument panel from being able to affect the physics.")

    ax = fig.add_axes([0.08, 0.435, 0.84, 0.365]); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4)
    layers = [
        (6.4, "What you run", ["scripts/fly.py", "scripts/vortex.py", "scripts/checkpoint.py"], "#7a5230"),
        (4.9, "Displays and control", ["panel.py", "viz.py", "autopilot.py", "manual.py"], BLUE),
        (3.4, "Asking what the aircraft is doing", ["sensors.py", "trim.py"], TEAL),
        (1.9, "The physics", ["integrate.py", "dynamics.py", "aero.py", "wind.py"], AMBER),
        (0.4, "Foundations", ["state.py", "aircraft.py", "atmosphere.py", "units.py"], MUTED),
    ]
    for yy, label, names, colour in layers:
        ax.text(0.0, yy + 1.02, label, fontsize=8.6, color=colour, fontweight="bold")
        w = 9.9 / len(names)
        for i, name in enumerate(names):
            ax.add_patch(FancyBboxPatch((i * w + 0.04, yy), w - 0.16, 0.72,
                                        boxstyle="round,pad=0.03,rounding_size=0.08",
                                        facecolor="white", edgecolor=colour, lw=1.3))
            ax.text(i * w + (w - 0.12) / 2, yy + 0.36, name, ha="center", va="center",
                    fontsize=8.2, color=INK, family=MONO)
    for yy in (1.28, 2.78, 4.28, 5.78):
        ax.add_patch(FancyArrowPatch((5.0, yy + 0.60), (5.0, yy + 0.02), arrowstyle="-|>",
                                     mutation_scale=11, color=RULE, lw=1.4))

    y = 0.415
    rows = [
        ("units.py", "Conversion constants only. Feet to metres and so on, in one place."),
        ("state.py", "What the aircraft's state is, and the quaternion maths."),
        ("atmosphere.py", "Air density and temperature against altitude, to 20 km."),
        ("aero.py", "Turns motion into aerodynamic forces. Never sees ground speed."),
        ("dynamics.py", "Newton's laws for a rigid body. The wind enters here."),
        ("wind.py", "The turbulence patterns themselves."),
        ("integrate.py", "Takes one time step, accurately."),
        ("aircraft.py", "The three aircraft, every number citing its source table."),
        ("sensors.py", "The only supported way to ask what the aircraft is doing."),
        ("trim.py", "Solves for the controls that hold steady level flight."),
        ("autopilot.py", "Holds a height, a heading and a speed."),
        ("manual.py", "Hand flying, trim, and the switch between the two."),
        ("panel.py", "The live instrument panel."),
        ("viz.py", "Saves a flight, and draws the after-the-fact plots."),
    ]
    y = table(fig, y, rows, [0.155, 0.70], header=("File", "What it does"), size=8.6,
              rowh=0.0182, mono_cols=(0,))
    y = callout(fig, y - 0.004, "The one rule worth knowing",
                "Ask what the aircraft is doing through sensors.py, never by reaching into the "
                f"state directly. The reason is on page {pageno('bug')}.", colour=RED, chars=92)
    emit(pdf, fig)

    # ------------------------------------------------------- the aircraft
    fig = page(pdf, "The three aircraft, and their sources", "The data")
    y = 0.865
    y = para(fig, y, "An aircraft in this simulator is a list of about forty numbers. Every one of "
                     "them is transcribed from a published document, with a comment naming the "
                     "table it came from. Where a source is missing a number, the number is set to "
                     "zero and the gap is recorded — it is never guessed.")

    rows = [
        ("Boeing 747", "NASA CR-2144, section IX [S1]",
         "Fully validated. The only\naircraft used for turbulence work."),
        ("Piper PA-28-180\nCherokee", "McCormick, via a worked\nexample [S4]",
         "The validated light aircraft.\nNo second source for its\nlateral numbers."),
        ("Cessna 172", "Roskam / USAF DATCOM,\nvia PyFME [S5]",
         "OUT OF SCOPE. Its rudder data\nis missing, so its turns are\nwrong. No result may be\nquoted from it."),
    ]
    y = table(fig, y, rows, [0.20, 0.30, 0.34],
              header=("Aircraft", "Source", "Standing"), size=8.8, rowh=0.078)

    y = callout(fig, y + 0.030, "Cruise conditions used, computed from the code",
                f"Boeing 747:   {V747:.1f} m/s true airspeed at {H747:.0f} m "
                f"({H747 / FT2M:.0f} ft), about Mach 0.80\n"
                f"Cherokee:     {CRUISE['cherokee']['airspeed']:.1f} m/s at "
                f"{CRUISE['cherokee']['altitude']:.0f} m\n"
                f"Cessna 172:   {CRUISE['cessna172']['airspeed']:.1f} m/s at "
                f"{CRUISE['cessna172']['altitude']:.0f} m", colour=BLUE, chars=92)

    y = para(fig, y, "Does the 747 actually behave like a 747? The way to check is to compare its "
                     "natural oscillations against the ones the source document measured. An "
                     "aircraft disturbed in flight wobbles in characteristic ways, each with its "
                     "own period and damping, and those are a fingerprint of the aerodynamics.")

    rows = [
        ("Dutch roll frequency", "0.943", "0.947 rad/s", "0.4%"),
        ("Dutch roll damping", "0.0361", "0.0349", "3.4%"),
        ("Roll response time", "1.795 s", "1.779 s", "0.9%"),
        ("Spiral time", "138.0 s", "137.0 s", "0.8%"),
        ("Phugoid frequency", "0.0553", "0.0673 rad/s", "17.8%  explained"),
        ("Short-period damping", "0.3425", "0.387", "11.5%  explained"),
    ]
    y = table(fig, y, rows, [0.28, 0.16, 0.22, 0.22],
              header=("Motion", "This model", "CR-2144 [S1]", "Difference"),
              size=8.8, rowh=0.0195)

    y = para(fig, y - 0.004, "The four lateral numbers agree to within one percent. The last two do "
                             "not, and the project says why rather than hiding it: the simulator "
                             "deliberately leaves out five of the source's coefficients, and a "
                             "second model built with them restored closes both to about one "
                             "percent [M2]. That is an understood, documented omission, not an "
                             "unexplained error.", size=9.4)
    emit(pdf, fig)

    # ------------------------------------------------------- the vortex
    fig = page(pdf, "The turbulence: a real one, from a real incident", "The science")
    y = 0.865
    y = para(fig, y, "In 1985 four NASA researchers took the flight recorders from airliners that "
                     "had hit severe clear-air turbulence near the top of the troposphere, and "
                     "worked backwards to reconstruct the air the aircraft had flown through. What "
                     "they found was not random buffeting. It was a row of large, organised "
                     "spinning vortices [S2].")
    y = para(fig, y, "Each vortex is modelled as a spinning core with smooth circulation around it "
                     "— a rotating solid centre embedded in an irrotational flow. Inside the "
                     "core the air speed rises in a straight line from the middle outwards. "
                     "Outside, it falls away as one over the distance.")

    ax = fig.add_axes([0.10, 0.520, 0.80, 0.195])
    ax.plot(xs / R0, w_up, color=BLUE, lw=2.0)
    ax.axvspan(-1, 1, color=WASH, zorder=0)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_xlabel("distance along the flight path, in core radii", fontsize=8.6, color=MUTED)
    ax.set_ylabel("vertical air speed  m/s", fontsize=8.6, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE)
    ax.text(0, ax.get_ylim()[1] * 0.80, "core", ha="center", fontsize=8.5, color=MUTED)
    ax.text(2.9, w_up.max() * 0.55, "straight line inside,\n1/distance outside",
            fontsize=8.2, color=BLUE, linespacing=1.5)
    ax.set_title(f"One vortex, Hannibal case: peak vertical air speed "
                 f"{np.abs(w_up).max():.1f} m/s  [computed]",
                 fontsize=9, color=INK, pad=6)

    y = 0.452
    rows = [
        ("Hannibal, Missouri", "37,000 ft", f"{R0:.0f} m ({R0/FT2M:.0f} ft)",
         f"{V0:.1f} m/s ({V0/FT2M:.0f} ft/s)", f"{SPACING:.0f} m"),
        ("Morton, Wyoming", "39,000 ft",
         f"{wind.PARKS_CASES['morton']['r0']:.0f} m ({wind.PARKS_CASES['morton']['r0']/FT2M:.0f} ft)",
         f"{wind.PARKS_CASES['morton']['v0']:.1f} m/s ({wind.PARKS_CASES['morton']['v0']/FT2M:.0f} ft/s)",
         f"{wind.PARKS_CASES['morton']['spacing']:.0f} m"),
    ]
    y = table(fig, y, rows, [0.23, 0.13, 0.19, 0.19, 0.14],
              header=("Case [S2]", "Altitude", "Core radius", "Edge speed", "Spacing"),
              size=8.4, rowh=0.020)

    y = callout(fig, y, "Why the spacing is not a free parameter",
                "The two cases give vortices spaced 2.92 and 3.56 core diameters apart. The paper "
                "checks that against Scorer's theoretical prediction of 2.7 for this kind of wave "
                "instability, and it agrees. That check is what turns the spacing from a number "
                "somebody chose into a number the physics requires. [S2]", colour=TEAL, chars=92)

    ax = fig.add_axes([0.10, 0.105, 0.80, 0.135])
    ax.plot(xs2 / 1000.0, w_up2, color=BLUE, lw=1.8)
    ax.axhline(0, color=RULE, lw=0.8)
    ax.set_xlabel("distance along the flight path, km", fontsize=8.6, color=MUTED)
    ax.set_ylabel("vertical air speed  m/s", fontsize=8.6, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    for s in ax.spines.values():
        s.set_color(RULE)
    ax.set_title(f"Two cores, {SPACING:.0f} m apart, as the aircraft flies through them  "
                 f"[computed]", fontsize=9, color=INK, pad=6)
    emit(pdf, fig)

    # ------------------------------------------------------ the fig-8 result
    fig = page(pdf, "The main finding, and why it works", "The result")
    y = 0.865
    y = para(fig, y, "A second paper, from 1994, plotted many flight-recorder events as pitch change "
                     "against g-load change, and found they fell into three separate clusters "
                     "depending on what caused them [S3]: vortex encounters in one place, "
                     "thunderstorm updrafts in another, and deliberate pilot manoeuvres in a third. "
                     "The obvious question is why.")

    # Anchored to the flowing y rather than to a fixed height, so editing the
    # paragraph above cannot silently open a gap or push the page over its footer.
    CHART_H = 0.245
    ax = fig.add_axes([0.08, y - CHART_H - 0.008, 0.84, CHART_H]); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0.12, 3.60)  # cropped to the drawn content
    ax.plot([0.7, 9.3], [3.15, 3.15], color=INK, lw=1.2)
    ax.plot([0.7, 0.7], [3.05, 3.25], color=INK, lw=1.2)
    ax.plot([9.3, 9.3], [3.05, 3.25], color=INK, lw=1.2)
    ax.text(5.0, 3.40, "the aircraft's own natural pitch wobble takes 6.6 seconds",
            ha="center", fontsize=9.2, color=INK, fontweight="bold")
    # Bar length is the event duration to the same scale as the 6.6 s ruler above
    # it, so the comparison is the picture rather than the caption. The caption is
    # per bar because the three do not share a sentence shape.
    SCALE = 8.6 / 6.6  # units per second
    bars = [
        (1.550, RED, "Vortex", "weather — a sharp jolt the aircraft barely has time to react to",
         "1.6 s to cross — a fifth of one pitch period"),
        (6.609, AMBER, "Pushdown", "NOT weather — the pilot pushes the nose down and holds it",
         "6.6 s of elevator — almost exactly one pitch period"),
        (20.0, TEAL, "Updraft", "weather — slow enough that the aircraft rides up with the air",
         "20 s to cross — about three pitch periods"),
    ]
    for i, (seconds, colour, name, note, caption) in enumerate(bars):
        x, yy = 0.7, 2.45 - i * 0.72
        w = min(seconds * SCALE, 8.6)
        ax.add_patch(FancyBboxPatch((x, yy), w, 0.30,
                                    boxstyle="round,pad=0.02,rounding_size=0.05",
                                    facecolor=colour, edgecolor="none", alpha=0.9))
        if w > 6.5:  # the caption would run off the page, so it goes inside the bar
            ax.text(x + 0.18, yy + 0.15, caption, va="center", fontsize=8.4,
                    color="white", fontweight="bold")
        else:
            ax.text(x + w + 0.12, yy + 0.15, caption, va="center", fontsize=8.4,
                    color=colour, fontweight="bold")
        ax.text(x, yy - 0.16, name, fontsize=9.4, color=colour, fontweight="bold")
        ax.text(x + 1.30, yy - 0.16, note, fontsize=8.2, color=MUTED)
    ax.text(5.0, 0.28, "the two weather events differ thirteen-fold in duration — "
            "the pushdown sits between them", ha="center", fontsize=9, color=INK,
            fontweight="bold")

    y = y - CHART_H - 0.030
    y = callout(fig, y, "The finding, in two sentences",
                "A vortex is crossed in a fifth of the aircraft's natural pitch period and hits it "
                "like an impulse; an updraft takes about three of them and the aircraft has time to "
                "follow it. That thirteen-fold gap separates the two WEATHER categories on timing "
                "alone. The third separates for a quite different reason: the pilot is moving the "
                "elevator, so the pitch follows the stick rather than the air.",
                colour=BLUE, chars=92)

    rows = [
        ("Gap between gusts, Hannibal", "4.52 s", "paper says “about 5 s apart” [S2]"),
        ("Pitch change, first vortex core", "2.24 deg", "paper's figure shows about 1.4 deg [S3]"),
        ("Pitch change, updraft column", "4.37 deg", "paper states 5.2 deg [S3]"),
        ("Pitch change, elevator pushdown", "30.37 deg", "paper's figure shows about 12 deg [S3]"),
        ("Peak g-load swing, vortex", "−1.23 g", "paper's band is −1.7 to −2.0 g [S3]"),
        ("g-load swing, pushdown", "−1.90 g", "aimed AT that band, so an input not a result"),
    ]
    y = table(fig, y, rows, [0.34, 0.16, 0.34],
              header=("Measured in this model [M3]", "Value", "What the paper reports"),
              size=8.7, rowh=0.0185)

    y = para(fig, y - 0.002, "The three land in the paper's order, and no single number matches: the "
                             "pushdown overshoots by a factor of two and a half. The ordering is the "
                             "whole claim and the project makes only that one — the papers never say "
                             "what aircraft they measured, and the vortex cases were DC-10s.",
                             size=9.4)
    y = callout(fig, y, "One thing the model provably cannot do",
                "Real encounters push harder downwards than upwards, and both papers put that down "
                "to the wing stalling. This model's lift rises in a dead straight line and never "
                "stalls, so an up-gust and an equal down-gust give exactly equal and opposite "
                "loads. That would need 747 stall data no source here holds: impossible, not "
                "pending.", colour=RED, chars=92)
    emit(pdf, fig)

    # --------------------------------------------- lee wave / thrust authority
    fig = page(pdf, "A second result: when the engines are not enough", "The result")
    y = 0.865
    y = para(fig, y, "The cluster diagram asks what KIND of event an aircraft flew through. A "
                     "different and blunter question is whether it could do anything about it, "
                     "and there is a standard number for that. The F-factor [S10] measures how "
                     "fast the moving air is draining the aircraft's energy. It is positive when "
                     "the aircraft is losing, and the rule that goes with it is simple: if the "
                     "F-factor is larger than the spare thrust the engines have, the aircraft "
                     "cannot fly its way out. It can trade height for speed or speed for height, "
                     "and that is all.")
    y = callout(fig, y, "How little spare thrust an airliner has at 40,000 feet",
                "Measured from this model, not assumed: at its cruise condition the 747 has "
                "spare thrust worth 2.3% of its weight at full throttle, and 6.6% of braking "
                "if it closes them. That is the entire budget. The same source quotes about 15% "
                "for a four-engine jet at takeoff — thrust falls away with air density and "
                "weight does not.", colour=AMBER, chars=92)

    CHART_H = 0.185
    ax = fig.add_axes([0.08, y - CHART_H - 0.010, 0.84, CHART_H]); ax.set_axis_off()
    ax.set_xlim(-0.072, 0.031); ax.set_ylim(0, 3.05)
    # The band is drawn to scale, and the point of drawing it is that it is
    # wildly lopsided: the aircraft can shed energy three times faster than it
    # can add it, which is why the hazard is always the DOWNdraft.
    ax.add_patch(FancyBboxPatch(
        (-0.0657, 0.62), 0.0657 + 0.0234, 1.86,
        boxstyle="round,pad=0.0005,rounding_size=0.002",
        facecolor=WASH, edgecolor="#b8c4d0", lw=1.0))
    ax.text(-0.021, 2.72, "what the engines can cover", ha="center", fontsize=9.4,
            color=INK, fontweight="bold")
    ax.plot([0.0234, 0.0234], [0.42, 2.52], color=RED, lw=1.6, ls="--")
    ax.text(0.0234, 0.24, "full thrust\n+0.023", ha="center", va="top", fontsize=8.2,
            color=RED, fontweight="bold")
    ax.plot([-0.0657, -0.0657], [0.42, 2.52], color=MUTED, lw=1.2, ls="--")
    ax.text(-0.0657, 0.24, "idle\n−0.066", ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.plot([0.0, 0.0], [0.42, 2.52], color="0.55", lw=0.9)
    ax.text(0.0, 0.24, "no effect\n0", ha="center", va="top", fontsize=8.2, color="0.45")
    for value, colour, name, verdict, yy in (
        (0.01291, TEAL, "northern leg", "engines cover it", 1.92),
        (0.02621, RED, "southern leg", "they do not", 1.06),
    ):
        ax.plot([0.0, value], [yy, yy], color=colour, lw=6.0, solid_capstyle="butt",
                alpha=0.85)
        ax.plot(value, yy, "o", color=colour, ms=7)
        # Left-aligned into the empty negative half. The bars live in the right
        # third of the scale, so a label trailing off their ends runs off the page.
        ax.text(-0.0619, yy + 0.16, f"{name}   F = +{value:.4f}   —   {verdict}",
                ha="left", va="bottom", fontsize=8.8, color=colour, fontweight="bold")

    y = y - CHART_H - 0.030
    y = para(fig, y, "Those two bars are the same aircraft on the same day, 50 km apart: the two "
                     "legs of one research flight over the Sierra Nevada [S9]. Flown into this "
                     "model, the weaker wave is something a 747 can hold against, and the "
                     "stronger one is not. The threshold falls between them — a wave amplitude "
                     "of 5.5 m/s — which is a more useful thing to know than either “safe” or "
                     "“dangerous” would have been.", size=9.6)
    y = callout(fig, y, "What this result is NOT",
                "It is a LOWER BOUND. A real mountain wave also pushes the air along the "
                "aircraft's track, and that adds to the F-factor; this model has only the "
                "up-and-down part, because computing the other half needs atmospheric data the "
                "source paper does not contain. So the real hazard is at least this bad and "
                "probably worse. The 0.1 threshold used in airline windshear alerting is also "
                "deliberately absent here: it is calibrated for takeoff and landing, where an "
                "aircraft has no height to trade.", colour=RED, chars=92)
    emit(pdf, fig)

    # ------------------------------------------------------ the bug
    fig = page(pdf, "The mistake that hid in plain sight", "A cautionary tale")
    y = 0.865
    y = para(fig, y, "This is worth a page because it is the most instructive thing in the project, "
                     "and because the fix shaped the design of everything after it.")
    y = para(fig, y, "An aircraft has two different speeds. Its speed over the ground, and its "
                     "speed through the air. In still air they are identical. In wind they are not, "
                     "and it is the speed through the air that keeps an aeroplane flying — "
                     "which is why an aircraft can be stationary over the ground in a strong enough "
                     "headwind and still be flying perfectly well.")

    ax = fig.add_axes([0.08, 0.470, 0.84, 0.245]); ax.set_axis_off()
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.0)
    for i, (label, gs, tas, colour) in enumerate([
            ("In still air — the two are the same number", 236, 236, BLUE),
            ("In a 25 m/s headwind — they are not", 211, 236, RED)]):
        yy = 2.30 - i * 1.45
        ax.text(0.0, yy + 0.30, label, fontsize=9.6, fontweight="bold", color=colour)
        ax.add_patch(FancyArrowPatch((0.0, yy - 0.10), (gs / 44.0, yy - 0.10),
                                     arrowstyle="-|>", mutation_scale=13, color=MUTED, lw=2.6))
        ax.text(gs / 44.0 + 0.14, yy - 0.10, f"{gs} m/s over the ground", fontsize=8.6,
                color=MUTED, va="center")
        ax.add_patch(FancyArrowPatch((0.0, yy - 0.52), (tas / 44.0, yy - 0.52),
                                     arrowstyle="-|>", mutation_scale=13, color=colour, lw=2.6))
        ax.text(tas / 44.0 + 0.14, yy - 0.52, f"{tas} m/s through the air", fontsize=8.6,
                color=colour, va="center", fontweight="bold")
    ax.text(0.0, 0.10, "Only the lower number keeps the aircraft flying. "
            "The simulator was using the upper one.",
            fontsize=8.8, color=INK, style="italic")

    y = 0.450
    y = para(fig, y, "Two parts of the project asked for “airspeed” and were quietly "
                     "handed the ground speed instead: the autopilot, and the display that computes "
                     "angle of attack. In still air this is invisible, because the two numbers are "
                     "the same. Every one of the two hundred and nine tests in the project at the "
                     "time was a still-air test, so not one of them could see it.")
    y = callout(fig, y, "How large the error actually was",
                "In a vortex encounter, the angle of attack computed the wrong way differed from "
                "the correct one by up to 7 degrees — measured, not estimated. The correlation "
                "between g-load and angle of attack was 0.9990 computed correctly and 0.5572 "
                "computed wrongly. The wrong version would have destroyed the main result. [M4]",
                colour=RED, chars=92)
    y = para(fig, y, "The fix was deliberately heavy-handed. A single module now owns the question "
                     "“what is the aircraft doing?”, it cannot be called without stating "
                     "what the air is doing, and the raw ground velocity is simply not in scope for "
                     "anything that might misuse it. There is no longer a wrong number available to "
                     "reach for.")
    y = para(fig, y, "The lesson was written into the project's rules: any quantity that has both "
                     "an air-relative and a ground-relative form needs at least one test that flies "
                     "through real wind. A whole test file now exists for that and nothing else.")
    y = callout(fig, y, "And the fix was checked by breaking it again",
                "Both bugs were deliberately re-introduced to confirm that exactly the two tests "
                "written to catch them went red, and that nothing else moved. A test that has never "
                "been seen to fail is not yet evidence of anything.", colour=TEAL, chars=92)
    emit(pdf, fig)

    # ------------------------------------------------------ the panel
    fig = page(pdf, "The instrument panel", "Flying it")
    y = 0.865
    y = para(fig, y, "The panel follows the standard cockpit layout that the RAF standardised in "
                     "1937 and every aircraft has used since [S8]: attitude in the middle, airspeed "
                     "to its left, altitude to its right, heading below. A pilot trained on one "
                     "aircraft can read any other. On the right is a second set of instruments no "
                     "real cockpit has, showing the quantities this project measures.")
    if PANEL_PNG.exists():
        import matplotlib.image as mpimg
        img = mpimg.imread(str(PANEL_PNG))
        ax = fig.add_axes([0.06, 0.462, 0.88, 0.315])
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title("The 747 approaching the first vortex core — range counting down "
                     "in the status block", fontsize=8.6, color=MUTED, pad=4)

    y = 0.432
    rows = [
        ("Airspeed, altitude", "Sliding scales, as in a modern airliner. The number stays\nstill and the scale moves past it."),
        ("Attitude", "Artificial horizon, with the aircraft symbol fixed and the\nworld moving behind it."),
        ("Slip ball", "Shows sideways force, not sideslip angle — a real ball is a\npendulum, and the two are different quantities."),
        ("Vertical speed", "Rate of climb or descent. Without it you cannot hold a\nheight by hand; you end up chasing the altimeter."),
        ("Load factor", f"The g-load. This is the vertical axis of the comparison on\npage {pageno('result')}, and it holds the peak of any excursion."),
        ("Angle of attack", "With a coloured band marking where this model's straight-\nline aerodynamics stop being trustworthy."),
        ("Wind and gust rate", "The gust rotation is labelled SIM TRUTH, because no real\ninstrument can measure it."),
    ]
    y = table(fig, y, rows, [0.20, 0.68], header=("Instrument", "What it shows"),
              size=8.6, rowh=0.036)
    y = callout(fig, y + 0.014, "Flying it",
                "Arrow keys are the control stick, comma and full stop the rudder, minus and equals "
                "the throttle, square brackets the trim, 't' trims out whatever you are holding, "
                "and 'a' hands over to the autopilot.", colour=BLUE, chars=92)
    emit(pdf, fig)

    # ------------------------------------------- what is proven / not
    fig = page(pdf, "What is proven, and what is not", "Honesty")
    y = 0.865
    y = para(fig, y, "The project keeps a ledger. Every claim in it is a measured number with the "
                     "tolerance its automated test enforces, and no row is ever deleted — if a "
                     "figure changes, the old one stays on the record, superseded.")

    rows = [
        ("Angular momentum held over 10 minutes", "0.00000000000057", "1 part in 10^11"),
        ("Orientation stays valid over 100,000 steps", "holds", "1 part in 10^12"),
        ("Coordinated turn against the textbook formula", "1.12% off", "under 3%"),
        ("Zero-strength wind against no wind at all", "bit-for-bit identical", "exactly equal"),
        ("Autopilot holds airspeed, not ground speed", "yes", "within 1.5 m/s"),
        ("Angle-of-attack change from a 10 m/s updraft", "2.43 deg", "within 2%"),
    ]
    y = table(fig, y, rows, [0.44, 0.24, 0.20],
              header=("Check [M1, M5]", "Measured", "Test allows"), size=8.7, rowh=0.0195)

    y = callout(fig, y, "What that fourth row is for",
                "Running the simulator with a wind model that returns exactly zero wind gives "
                "results identical to the last decimal place to running it with no wind model at "
                "all. That proves the turbulence machinery adds nothing and disturbs nothing when "
                "it is switched off — so any difference seen with it switched on is real.",
                colour=TEAL, chars=92)

    y = para(fig, y, "Set against that, the project is equally explicit about its limits:", size=10)
    limits = [
        ("Cannot reproduce the up/down asymmetry", "The wing never stalls in this model, and no "
         "source held by the project has stall data for the 747. Recorded as impossible, not pending."),
        ("Cannot match the papers' g-loads exactly", "Neither paper says what aircraft it measured, "
         "and the vortex cases were DC-10s. Only orderings and magnitudes are ever claimed."),
        ("The vortex numbers carry the source's own uncertainty", "The paper derived them through "
         "an assumed aerodynamic model, so a 1-degree error becomes 4.12 m/s of wind. Treat them "
         "as accurate to roughly a quarter."),
        ("Any encounter past about 10-12 degrees of angle of attack", "reports lift the sources say "
         "is not there — in either direction, since the model's lift is a straight line and a "
         "pushdown leaves it as surely as a pull-up. The panel marks the band green, amber and red "
         f"so a run that leaves the valid range says so. The pushdown on page {pageno('result')} reaches 10.3 "
         "degrees: inside the amber, and reported as such rather than quietly quoted."),
        ("The Cessna 172 is out of scope", "Its rudder data is missing from the source, so its "
         "turns are wrong. It flies and passes its tests, but no result may be quoted from it."),
    ]
    for head, text in limits:
        lines = wrap(text, 78)
        fig.text(0.095, y, "—", color=RED, fontsize=9.5, va="top")
        fig.text(0.125, y, head, color=INK, fontsize=9.2, fontweight="bold", va="top")
        fig.text(0.125, y - 0.0175, lines, color=MUTED, fontsize=8.8, va="top", linespacing=1.5)
        y -= 0.0175 * (lines.count("\n") + 1) + 0.0215
    emit(pdf, fig)

    # ------------------------------------------------------ running it
    fig = page(pdf, "Running it, and how fast it goes", "Practicalities")
    y = 0.865
    rows = [
        ("python -m pytest atisim/tests -q", "All 260 tests. The only complete statement of\nwhat works."),
        ("python scripts/fly.py", "Fly it yourself, in still air."),
        ("python scripts/fly.py --wind hannibal", "Fly into the 1985 vortex array."),
        ("python scripts/vortex.py", "Run the turbulence analysis and draw the figure."),
        ("python scripts/checkpoint.py", "Check the 747 against CR-2144."),
        ("python scripts/analyse.py run.npz", "Re-plot a saved flight without re-flying it."),
    ]
    y = table(fig, y, rows, [0.42, 0.46], header=("Command", "What it does"),
              size=8.4, rowh=0.032, mono_cols=(0,))

    y = para(fig, y + 0.008, "On the question of speed: the simulator deliberately does not use much "
                             "of a modern computer. Two separate reasons, and only one of them is "
                             "worth fixing.", size=10)
    y = callout(fig, y, "Why the processor looks idle",
                "The whole loop runs on a single processor core, and paces itself to the wall clock "
                "— it does exactly fifty physics steps per second and then waits. On a "
                "twelve-core machine one saturated core is about eight percent of the total, which "
                "is what the task manager shows. More cores cannot help: drawing is inherently "
                "sequential, and each physics calculation is far too small to split up.",
                colour=BLUE, chars=92)

    y = para(fig, y, "The frame rate was another matter. Profiling found the physics was never the "
                     "problem — it is a rounding error in the budget — and that most of "
                     "the time was going somewhere unexpected. [M6]", size=10)

    rows = [
        ("One step of the physics", "0.11 ms", "0.7% of one core, at 50 steps a second"),
        ("Reading the instruments  BEFORE", "6.56 ms", "called once per physics step"),
        ("Reading the accelerometers  BEFORE", "10.70 ms", "called once per frame"),
        ("Drawing the panel", "30 ms", "the genuine cost"),
        ("Reading the instruments  AFTER", "0.03 ms", "234x faster"),
    ]
    y = table(fig, y, rows, [0.36, 0.16, 0.36],
              header=("Where the time went", "Time", "Note"), size=8.7, rowh=0.0195)

    y = para(fig, y - 0.004, "The instrument reads were doing about twenty tiny calculations one at "
                             "a time, and the overhead of launching each one dwarfed the arithmetic. "
                             "Compiling them into a single operation — a two-line change — "
                             "cut the frame from 73 ms to 38 ms and took the frame rate from 13.7 "
                             "to 26.4 per second. Drawing the panel is now four-fifths of what "
                             "remains, and the three sixty-second history strips, at 1,201 points "
                             "each redrawn every frame, are the obvious next target.", size=9.4)
    y = callout(fig, y, "Where the computer's full power would actually be used",
                "Not in flying one aircraft. The project is built to run many simulated flights at "
                "once — the same turbulence encounter flown by hundreds of copies of the "
                "aircraft in parallel, to get error bars on the result. That is what the underlying "
                "library is for, and that is the step this work was building towards.",
                colour=TEAL, chars=92)
    emit(pdf, fig)

    # ------------------------------------------------------ sources
    fig = page(pdf, "Sources", "Citations")
    y = 0.865
    y = para(fig, y, "Every number in this document traces to one of the following. Source tags "
                     "[S] are published documents; measurement tags [M] are figures produced by "
                     "the project's own automated tests and recorded in its evidence ledger.",
                     size=9.6)

    srcs = [
        ("[S1]", "Heffley, R. K. and Jewell, W. F. (1972). Aircraft Handling Qualities Data. "
                 "NASA CR-2144, section IX. Supplies the Boeing 747 geometry, mass, inertia and "
                 f"aerodynamic coefficients, and the reference oscillation table on page {pageno('data')}. "
                 f"Contains no stall data, which is why page {pageno('result')}'s asymmetry is out of reach."),
        ("[S2]", "Parks, E. K., Wingrove, R. C., Bach, R. E. and Mehta, R. S. (1985). "
                 "Identification of Vortex-Induced Clear Air Turbulence Using Airline Flight "
                 "Records. Journal of Aircraft, volume 22, number 2, pages 124-129. Supplies the "
                 "vortex model, both identified cases, and the spacing check against Scorer."),
        ("[S3]", "Wingrove, R. C. and Bach, R. E. (1994). Severe Turbulence and Maneuvering from "
                 "Airline Flight Records. Journal of Aircraft, volume 31, number 4, pages 753-760. "
                 "Supplies the updraft magnitudes and duration, the g-load statistics, and the "
                 f"cluster diagram discussed on page {pageno('result')}."),
        ("[S4]", "McCormick, B. W., via a worked example. Supplies the Piper PA-28-180 Cherokee "
                 "coefficients. No second source was found for its lateral set."),
        ("[S5]", "Roskam, J. / USAF DATCOM, via the PyFME project. Supplies the Cessna 172 "
                 "coefficients. Its rudder data is omitted and inconsistent, so the entire rudder "
                 "set is zeroed and the aircraft is out of scope."),
        ("[S6]", "Nelson, R. C. / Etkin, B. / McRuer, D. Supply the Navion coefficients used in "
                 "testing. No published oscillation table was found, so its tests assert ranges "
                 "rather than values."),
        ("[S7]", "MIL-F-8785C. The military specification for the Dryden turbulence spectra. Not "
                 "yet used: its intensity figure at altitude is a chart to be read off, not a "
                 "formula, and has still to be digitised."),
        ("[S8]", "The standard six-instrument “basic T” arrangement, standardised by the "
                 "Royal Air Force in 1937 and used on essentially every aircraft since; and the "
                 "modern primary flight display layout that preserves it. General aviation "
                 "practice rather than a single document."),
        ("[S9]", "Doyle, J. D., Jiang, Q., Smith, R. B. and Grubisic, V. (2011). "
                 "Three-Dimensional Characteristics of Stratospheric Mountain Waves during "
                 "T-REX. Monthly Weather Review, volume 139, pages 3-23. Supplies the lee-wave "
                 f"amplitudes on page {pageno('leewave')}. Chosen over a textbook because its research aircraft "
                 "flew at 11.3 and 13.1 km and this project's 747 cruises between them. Its "
                 "wavelength figure is tropospheric, so the wavelength used here is declared."),
        ("[S10]", "Proctor, F. H., Hinton, D. A. and Bowles, R. L. (2000). A Windshear Hazard "
                  "Index. 9th Conference on Aviation, Range and Aerospace Meteorology, paper "
                  "7.7, pages 482-487. Supplies the F-factor and the rule that a shear beating "
                  "an aircraft's spare thrust cannot be flown out of. Its 0.1 alerting "
                  "threshold is for takeoff and landing and is deliberately not used here."),
    ]
    for tag, text in srcs:
        lines = wrap(text, 88)
        fig.text(0.08, y, tag, color=BLUE, fontsize=8.8, fontweight="bold", va="top",
                 family=MONO)
        fig.text(0.145, y, lines, color=INK, fontsize=8.5, va="top", linespacing=1.5)
        y -= 0.0155 * (lines.count("\n") + 1) + 0.011

    # Measurements get their own page: the source list outgrew one page when the
    # lee-wave sources landed, and cramming both would have meant shrinking the
    # citations, which are the part a reader is most likely to need to read.
    emit(pdf, fig)

    fig = page(pdf, "Measurements", "Citations")
    y = 0.865
    y = para(fig, y, "Each tag below is a figure this project measured itself. They are "
                     "reproducible: the script or test named against each one produces it, and "
                     "the evidence ledger in PROJECT.md section 4 records it with the tolerance "
                     "its test enforces.", size=9.6)
    meas = [
        ("[M1]", "Integrator and rigid-body conservation checks. PROJECT.md section 4, "
                 "“Integrator and rigid body”; tests in test_conservation.py."),
        ("[M2]", "Boeing 747 oscillations against CR-2144. PROJECT.md section 4, “747 modes "
                 "vs CR-2144”; tests in test_cr2144_modes.py."),
        ("[M3]", "Vortex and updraft encounter results. PROJECT.md section 4, “Vortex and "
                 "updraft encounters”; produced by scripts/vortex.py."),
        ("[M4]", "Air-relative sensing, including the 7-degree error and the two correlations. "
                 "PROJECT.md sections 4 and 6; tests in test_sensors.py."),
        ("[M5]", "Air-relative sensing tolerances. PROJECT.md section 4, “Air-relative "
                 "sensing”."),
        ("[M6]", "Live-loop timing, 7 August 2026, 12-core machine, medians of 120 runs."),
        ("[M7]", "Lee wave, F-factor and thrust authority. PROJECT.md section 4, “Mountain lee "
                 "wave and the F-factor”; produced by scripts/leewave.py, tests in "
                 "test_lee_wave.py."),
    ]
    for tag, text in meas:
        lines = wrap(text, 88)
        fig.text(0.08, y, tag, color=TEAL, fontsize=8.8, fontweight="bold", va="top",
                 family=MONO)
        fig.text(0.145, y, lines, color=INK, fontsize=8.5, va="top", linespacing=1.5)
        y -= 0.0155 * (lines.count("\n") + 1) + 0.011

    fig.text(0.08, y - 0.012, "The project's own standing record is docs/PROJECT.md, which carries "
                              "the full evidence\nledger, the list of known gaps, and the "
                              "session-by-session history.",
             color=MUTED, fontsize=8.5, va="top", linespacing=1.5)
    emit(pdf, fig)

    check_pagination()

    info = pdf.infodict()
    info["Title"] = "AtiSim - project summary"
    info["Subject"] = "Plain-English summary with cited sources"

print(f"wrote {OUT}  ({len(PAGES)} pages)")
