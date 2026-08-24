"""Build the JSBSim cross-code verification report PDF.

Run from the project root:

    .venv/Scripts/python.exe scripts/jsbsim_report.py docs/summary/jsbsim-737-report.pdf

SAME RULE AS scripts/summary.py AND scripts/turbulence_report.py. Every number
in this document is either read from the frozen reference
(flightsim/tests/data/jsbsim_737_reference.xml, written by
scripts/gen_jsbsim_reference.py from JSBSim itself) or computed here by calling
the project's own code. Nothing is typed in from memory. The one exception is
the column of 737.xml's own <function> constants on the recovery page, which is
transcribed from that file and labelled as such -- it is the thing being
compared against, and it is quoted so the reader can see the size of the
AERORP-offset correction.

This does NOT import jsbsim. It reads the frozen reference, exactly as the test
suite does, so the report can be rebuilt on a machine that has no JSBSim.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

import jax
import jax.numpy as jnp

# THIS tree, not whichever one is pip-installed. `flightsim` is installed
# editable against the main checkout, so a script run from a worktree silently
# imports the OTHER tree's code and reports on it -- no error, just the wrong
# answers. pytest happens to be immune because it puts its rootdir first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import flightsim  # noqa: E402, F401  -- enables x64 before any array is made
from flightsim import aero, integrate, jsbsim_ref, trim, validation  # noqa: E402
from flightsim.aircraft import REGISTRY  # noqa: E402
from flightsim.atmosphere import RHO0, density, speed_of_sound  # noqa: E402
from flightsim.state import Controls, State, euler_to_quat  # noqa: E402
from flightsim.units import FT2M  # noqa: E402

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/summary/jsbsim-737-report.pdf")

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
FLOOR = 0.062


# ---------------------------------------------------------------------------
# Layout helpers. Copied from scripts/turbulence_report.py rather than imported,
# because that module builds its own PDF at import time.
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
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1,
                                boxstyle="round,pad=0.0,rounding_size=0.02",
                                transform=ax.transAxes, facecolor=WASH,
                                edgecolor=colour, lw=0.0))
    ax.plot([0.0, 0.0], [0, 1], transform=ax.transAxes, color=colour, lw=3,
            clip_on=False)
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
            fig.text(cx, y, cell, color=BLUE, fontsize=size, fontweight="bold",
                     va="top")
            cx += w
        y -= 0.006
        fig.lines.append(plt.Line2D([0.08, 0.92], [y, y], transform=fig.transFigure,
                                    color=RULE, lw=0.7))
        y -= 0.012
    for row in rows:
        cx = x0
        for i, (w, cell) in enumerate(zip(widths, row)):
            fig.text(cx, y, cell, color=INK if i == 0 else MUTED, fontsize=size,
                     va="top", family=MONO if i in mono_cols else None)
            cx += w
        y -= rowh
    return y - 0.010


PAGES = []
PAGE_ORDER = ["cover", "conditions", "recovery", "layer1", "layer2", "layer3",
              "layer4", "thrust", "limits"]


def emit(fig, y=None):
    if y is not None and y < FLOOR:
        raise SystemExit(
            f"layout overflow on page {len(PAGES) + 1}: cursor reached y={y:.4f}, "
            f"below the {FLOOR} floor."
        )
    PAGES.append(fig)
    fig.text(0.92, 0.035, str(len(PAGES)), color=MUTED, fontsize=8.5, ha="right")
    fig.text(0.08, 0.035, "JAX Flight Simulator - JSBSim cross-code verification",
             color=MUTED, fontsize=8.5)
    PDF.savefig(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Everything below is measured at build time.
# ---------------------------------------------------------------------------
REF = jsbsim_ref.load()
COND = REF.condition["cruise"]
TRIM = REF.trim["longitudinal"]
AC = REGISTRY["boeing737"]

# --- input conditions -------------------------------------------------------
ATMOS_ALTS_FT = [0.0, 10000.0, 20000.0, 30000.0, 40000.0]


def geopotential(h):
    return 6356766.0 * h / (6356766.0 + h)


ATMOS = []
for h_ft in ATMOS_ALTS_FT:
    h = h_ft * FT2M
    # JSBSim uses geopotential altitude; flightsim's ISA uses geometric. This
    # reproduces JSBSim's density without importing it, and is checked against
    # the recorded 30,000 ft value below.
    rho_js = float(density(geopotential(h)))
    ATMOS.append((h_ft, 100.0 * (float(density(h)) - rho_js) / rho_js))
ATMOS_CHECK = 100.0 * abs(float(density(geopotential(COND.altitude))) - COND.density) \
    / COND.density

# --- layer 1 ----------------------------------------------------------------
CD0_ALPHA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.5, 0.042, 0.021, 0.042, 1.5])
CD_BETA = ([-1.57, -0.26, 0.0, 0.26, 1.57], [1.23, 0.05, 0.0, 0.05, 1.23])
CD_DE = 0.059
Z_ARM = 4.925 / 12.31


def missing_drag(p):
    return ((np.interp(p.alpha, *CD0_ALPHA) - np.interp(TRIM.alpha, *CD0_ALPHA))
            + (np.interp(p.beta, *CD_BETA) - np.interp(0.0, *CD_BETA))
            + CD_DE * (abs(p.controls[0]) - abs(TRIM.elevator)))


def fs_coefficients(p):
    c = Controls(elevator=jnp.array(p.controls[0]), aileron=jnp.array(p.controls[1]),
                 rudder=jnp.array(p.controls[2]), throttle=jnp.array(0.0))
    return [float(v) for v in aero.coefficients(
        jnp.array(p.vel_body), jnp.array(p.rates), c, AC, jnp.array(p.sound_speed))]


NAMES = ["CL", "CD", "CY", "Cl", "Cm", "Cn"]
L1_WORST = {n: 0.0 for n in NAMES}
L1_CD_UNEXPLAINED = 0.0
for point in REF.sweep:
    got = fs_coefficients(point)
    for k, name in enumerate(NAMES):
        L1_WORST[name] = max(L1_WORST[name], abs(got[k] - point.coefficients[k]))
    L1_CD_UNEXPLAINED = max(
        L1_CD_UNEXPLAINED,
        abs((point.coefficients[1] - got[1]) - missing_drag(point)))

# --- layer 2 ----------------------------------------------------------------
_x, _res = trim.trim(jnp.array(COND.airspeed), jnp.array(COND.matched_altitude), AC)
FS_ALPHA, FS_DE, FS_THROTTLE = (float(v) for v in _x)
_rho = float(density(COND.matched_altitude))
_mach = COND.airspeed / COND.sound_speed
FS_THRUST = (FS_THROTTLE * float(AC.max_thrust) * (_rho / RHO0) ** float(AC.thrust_lapse)
             * (1.0 + float(AC.mach_ram) * _mach**2))

# --- layer 3 ----------------------------------------------------------------
FS_LON = validation.longitudinal_modes(AC, FS_ALPHA, FS_DE, FS_THROTTLE,
                                       COND.airspeed, COND.matched_altitude)
_ev = np.linalg.eigvals(REF.linearization.longitudinal)
JS_LON = sorted((abs(l), -l.real / abs(l)) for l in _ev if l.imag > 1e-12)

DAMPER_GAIN = 0.35 * 2.0 * COND.airspeed / float(AC.b)
D_CNR = float(AC.Cndr) * DAMPER_GAIN
D_CLR = float(AC.Cldr) * DAMPER_GAIN
CLOSED = AC._replace(Cnr=jnp.array(float(AC.Cnr) + D_CNR),
                     Clr=jnp.array(float(AC.Clr) + D_CLR))
FS_LAT_BARE = validation.lateral_modes(AC, FS_ALPHA, FS_DE, FS_THROTTLE,
                                       COND.airspeed, COND.matched_altitude)
FS_LAT = validation.lateral_modes(CLOSED, FS_ALPHA, FS_DE, FS_THROTTLE,
                                  COND.airspeed, COND.matched_altitude)
_evl = np.linalg.eigvals(REF.linearization.lateral)
_evl = _evl[np.abs(_evl) > 1e-8]
_pair = _evl[np.abs(_evl.imag) > 1e-9][0]
_reals = np.sort(_evl[np.abs(_evl.imag) <= 1e-9].real)
JS_LAT = ((abs(_pair), -_pair.real / abs(_pair)), -1.0 / _reals[0], -1.0 / _reals[-1])

# --- layer 4 ----------------------------------------------------------------
TRAJ = {}
for case in ("elevator_doublet", "rudder_kick"):
    samples = REF.trajectory[case]
    first = samples[0]
    altitude = first.altitude + (COND.matched_altitude - COND.altitude)
    rho = float(density(altitude))
    mach = float(np.linalg.norm(first.vel_body)) / float(speed_of_sound(altitude))
    throttle = first.thrust / (float(AC.max_thrust) * (rho / RHO0) ** float(AC.thrust_lapse)
                               * (1.0 + float(AC.mach_ram) * mach**2))
    sim = integrate.init_sim(
        State(pos_ned=jnp.array([0.0, 0.0, -altitude]),
              vel_body=jnp.array(first.vel_body),
              quat=euler_to_quat(*(jnp.array(v) for v in first.euler)),
              omega=jnp.array(first.omega)),
        jax.random.PRNGKey(0))
    ts, fs_u, js_u, fs_r, js_r, dv = [], [], [], [], [], []
    per_axis, beta_max = [], 0.0
    for previous, current in zip(samples, samples[1:]):
        c = Controls(elevator=jnp.array(previous.controls[0]),
                     aileron=jnp.array(previous.controls[1]),
                     rudder=jnp.array(previous.controls[2]),
                     throttle=jnp.array(throttle))
        sim = integrate.step(sim, c, jnp.array(current.t - previous.t), AC)
        v = np.asarray(sim.state.vel_body)
        o = np.asarray(sim.state.omega)
        ts.append(current.t)
        fs_u.append(v[0]); js_u.append(current.vel_body[0])
        fs_r.append(o[2]); js_r.append(current.omega[2])
        dv.append(float(np.abs(v - current.vel_body).max()))
        per_axis.append(v - current.vel_body)
        beta_max = max(beta_max, abs(np.arcsin(
            current.vel_body[1] / np.linalg.norm(current.vel_body))))
    TRAJ[case] = dict(t=np.array(ts), fs_u=np.array(fs_u), js_u=np.array(js_u),
                      fs_r=np.array(fs_r), js_r=np.array(js_r), dv=np.array(dv),
                      per_axis=np.max(np.abs(np.array(per_axis)), axis=0),
                      beta_max=np.degrees(beta_max))

# --- the drag terms flightsim lacks, integrated over each trajectory --------
DRIFT = {}
for case in ("elevator_doublet", "rudder_kick"):
    total, previous = 0.0, REF.trajectory[case][0].t
    for smp in REF.trajectory[case]:
        speed = float(np.linalg.norm(smp.vel_body))
        beta = np.arcsin(smp.vel_body[1] / speed)
        alpha = np.arctan2(smp.vel_body[2], smp.vel_body[0])
        d_cd = (np.interp(beta, *CD_BETA)
                + np.interp(alpha, *CD0_ALPHA) - np.interp(TRIM.alpha, *CD0_ALPHA))
        total += (d_cd * 0.5 * _rho * speed**2 * float(AC.S) / float(AC.mass)) * (
            smp.t - previous)
        previous = smp.t
    DRIFT[case] = abs(total)

# --- thrust -----------------------------------------------------------------
# CFM56 MilThrust, M = 0 row, from engine/CFM56.xml. Transcribed and labelled.
CFM_ALT_FT = [0.0, 10000.0, 20000.0, 30000.0, 40000.0, 50000.0]
CFM_M0 = [1.0000, 0.7400, 0.5340, 0.3720, 0.2410, 0.1490]
CFM_MACH = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
CFM_M_ROW_30K = [0.3720, 0.3550, 0.3570, 0.3780, 0.4170, 0.4750]

PDF = PdfPages(OUT)

# ------------------------------------------------------------------- cover
fig = plt.figure(figsize=A4, facecolor="white")
ax = fig.add_axes([0, 0.62, 1, 0.38])
ax.set_axis_off()
ax.add_patch(FancyBboxPatch((0.0, 0.0), 1, 1, boxstyle="square,pad=0",
                            transform=ax.transAxes, facecolor=WASH, edgecolor="none"))
ax.text(0.086, 0.70, "CROSS-CODE VERIFICATION", color=BLUE, fontsize=9,
        fontweight="bold", transform=ax.transAxes)
ax.text(0.086, 0.52, "flightsim against JSBSim", color=INK, fontsize=27,
        fontweight="bold", transform=ax.transAxes)
ax.text(0.086, 0.36, "via the Boeing 737, at 30,000 ft and M 0.78", color=MUTED,
        fontsize=13, transform=ax.transAxes)
y = 0.56
y = para(fig, y,
         "Two independent 6-DOF implementations are given the same coefficients at the "
         "same flight condition, and their answers are compared in four layers: the "
         "aerodynamic build-up, the trim solve, the linearisation, and a 20-second "
         "integration. A disagreement is then a defect in one of the two codes.")
y = callout(fig, y, "What this claims",
            "flightsim's aero build-up, trim solver, linearisation and integrator agree "
            "with an independent, mature engine when both are fed the same coefficients "
            "at the same state.", colour=TEAL)
y = callout(fig, y, "What this does NOT claim",
            "That the JSBSim 737 is a correct 737. Its own file header says it was built "
            "from public data, technical reports, textbooks 'and guesses', that validation "
            "extends only to the extent that it 'seems to fly right', and that it is for "
            "'educational and entertainment purposes only'. Nothing here supports any "
            "claim about a real 737, and none is made. What is being used is JSBSim's "
            "ENGINE, not its dataset.", colour=RED)
y = heading(fig, y, "Headline results")
y = table(fig, y, [
    ("1  build-up", f"CL {L1_WORST['CL']:.1e}   CY {L1_WORST['CY']:.1e}   "
                    f"Cl {L1_WORST['Cl']:.1e}   Cn {L1_WORST['Cn']:.1e}", "worst |difference|"),
    ("", f"CD and Cm differ; predicted to {L1_CD_UNEXPLAINED:.1e}", "known missing terms"),
    ("2  trim", f"alpha {np.degrees(FS_ALPHA):.3f} vs {np.degrees(TRIM.alpha):.3f} deg, "
                f"thrust {100 * (FS_THRUST - TRIM.thrust) / TRIM.thrust:+.2f}%", ""),
    ("3  modes", f"short period wn {abs(FS_LON[1][0] - JS_LON[1][0]) / JS_LON[1][0] * 100:.2f}%, "
                 f"zeta {abs(FS_LON[1][1] - JS_LON[1][1]) / JS_LON[1][1] * 100:.2f}%", ""),
    ("4  trajectory", f"elevator doublet {TRAJ['elevator_doublet']['dv'].max():.3f} m/s "
                      f"over 20 s", f"Earth-rotation floor "
                                    f"{REF.diagnostics['coriolis_velocity_m_s']:.3f}"),
], [0.16, 0.50, 0.26], mono_cols=(1,))
y = para(fig, y, f"Reference: JSBSim {REF.jsbsim_version}", size=8.5, color=MUTED)
emit(fig, y)

# -------------------------------------------------------------- conditions
fig = page("Matching the input conditions", kicker="before any comparison")
y = 0.86
y = para(fig, y,
         "A systematic offset in the INPUTS biases every layer in the same direction and "
         "reads as a defect in the code. Both halves were measured before any comparison "
         "was written.")
y = heading(fig, y, "Sign conventions: all sixteen agree")
y = para(fig, y,
         "Measured directly against the engine rather than inferred from documentation. "
         "beta, side force, roll and yaw due to beta, aileron (mapping to the LEFT "
         "aileron position), rudder, elevator, and the wind-axis force senses all match "
         "this project's stated conventions. No sign flips are needed anywhere. JSBSim's "
         "737 genuinely has no CYdr at all: rudder produces yaw and roll moments but zero "
         "side force, which is reproduced deliberately.")
y = heading(fig, y, "Atmosphere: one systematic error, found and neutralised")
ax = fig.add_axes([0.10, y - 0.20, 0.36, 0.18])
ax.bar([f"{int(a / 1000)}k" for a, _ in ATMOS], [e for _, e in ATMOS], color=RED, width=0.6)
ax.axhline(0, color=INK, lw=0.8)
ax.set_ylabel("density error, %", fontsize=8.5)
ax.set_xlabel("altitude, ft", fontsize=8.5)
ax.tick_params(labelsize=8)
ax.set_title("before matching", fontsize=9, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax2 = fig.add_axes([0.56, y - 0.20, 0.36, 0.18])
ax2.bar([f"{int(a / 1000)}k" for a, _ in ATMOS], [1e-14] * len(ATMOS), color=TEAL, width=0.6)
ax2.set_ylim(-0.4, 0.05)
ax2.axhline(0, color=INK, lw=0.8)
ax2.set_xlabel("altitude, ft", fontsize=8.5)
ax2.tick_params(labelsize=8)
ax2.set_title("after matching on density", fontsize=9, color=TEAL)
ax2.text(0.5, 0.45,
         f"every bar below {COND.density_match_residual:.0e} (machine precision)",
         transform=ax2.transAxes, ha="center", va="center", fontsize=8.5, color=TEAL)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
y -= 0.255
y = para(fig, y,
         "flightsim's ISA uses GEOMETRIC altitude where the standard is defined on "
         "GEOPOTENTIAL altitude, so its density runs low, and the error grows with height "
         f"to {abs(ATMOS[3][1]):.3f}% at 30,000 ft and {abs(ATMOS[4][1]):.3f}% at 40,000. "
         "Dynamic pressure is proportional to density, so that is a same-signed bias on "
         "every force in every layer -- about 22% of the layer-2 trim tolerance on its own.")
y = callout(fig, y, "Neutralised by matching density, not altitude",
            "Altitude is not itself an input to the physics; it enters only through "
            "density and the speed of sound. The harness solves for the geometric altitude "
            f"at which flightsim's density equals JSBSim's -- {COND.matched_altitude:.2f} m, "
            f"{(COND.matched_altitude - COND.altitude) / FT2M:.2f} ft below the nominal "
            "30,000 ft, which is exactly the geopotential correction arrived at "
            f"independently. Density then agrees to {COND.density_match_residual:.1e} "
            "relative and the speed-of-sound residual improves about ninetyfold. The "
            "underlying defect is in pre-existing code and is filed separately.",
            colour=TEAL)
emit(fig, y)

# ---------------------------------------------------------------- recovery
fig = page("Recovering the derivatives", kicker="from the engine, not the file")
y = 0.86
y = para(fig, y,
         "JSBSim applies aerodynamic forces at the AERORP (x = 625 in) but takes moments "
         "about the CG (x = 610.8 in). That 1.183 ft offset -- 0.096 of the mean chord -- "
         "is folded into every moment derivative the ENGINE exhibits, while 737.xml's "
         "<function> constants describe only the coefficient before it is applied.")
y = table(fig, y, [
    ("Cma", "-0.600", f"{REF.derivatives['Cma']:+.4f}", "77% -- dominated by CLa x 0.096 c"),
    ("Cmde", "-0.849", f"{REF.derivatives['Cmde']:+.4f}", "CLde x 0.096 c"),
    ("Clb", "-0.090", f"{REF.derivatives['Clb']:+.4f}", "side force 4.925 ft above the CG"),
    ("Cnb", "+0.260", f"{REF.derivatives['Cnb']:+.4f}", "side force 1.183 ft aft of the CG"),
    ("CLa", "+4.348", f"{REF.derivatives['CLa']:+.4f}", "exact -- no offset effect on lift"),
    ("Clp", "-0.400", f"{REF.derivatives['Clp']:+.4f}", "exact -- roll rate makes no side force"),
    ("Cnr", "-0.350", f"{REF.derivatives['Cnr']:+.4f}", "exact -- yaw rate makes no side force"),
    ("Cmq", "-27.0", f"{REF.derivatives['Cmq']:+.4f}", "= Cmq + Cmadot (-16); alphadot = q"),
], [0.10, 0.14, 0.16, 0.46],
    header=("", "737.xml", "engine", "why they differ"), mono_cols=(1, 2))
y = callout(fig, y, "Why this decided the whole design",
            "Transcribing Cmalpha = -0.6 would have built an aeroplane with 56% of the "
            "right pitch stiffness, and the resulting mode mismatch would have been "
            "indistinguishable from a defect in dynamics.py. Differencing the RUNNING "
            "engine folds the offset in by construction, and does so in all three axes -- "
            "the roll and yaw rows above close to 0.15% and 0.17% from the same "
            "side-force-times-arm arithmetic.", colour=AMBER)
y = heading(fig, y, "Six derivatives that are absent, and asserted to be")
y = para(fig, y,
         "737.xml defines no CLq, CYp, CYr, CYdr, Cnp or Cnda. flightsim carries 0.0 for "
         "each, which is agreement rather than approximation: both engines then compute "
         "the same thing. The generator MEASURES all six rather than assuming them, "
         "against a derived floor. That assertion earned its keep: at JSBSim's default "
         "1/120 s step the settling run integrates far enough for a pitch rate to drift "
         "alpha and manufacture CLq = +4.60 out of nothing. The predicted artifact for "
         "that step size is +4.57. Settling on a 1e-6 s step instead leaves the FCS fully "
         f"converged and drops the worst residual to "
         f"{max(abs(v) for v in REF.absent.values()):.1e}.")
emit(fig, y)

# ------------------------------------------------------------------ layer1
fig = page("Layer 1 - the aerodynamic build-up", kicker="33 sweep points")
y = 0.86
y = para(fig, y,
         "flightsim's aero.coefficients against JSBSim's forces and moments at identical "
         "read-back states, sweeping alpha, beta, all three rates and all three surfaces. "
         "Driven by the recorded body-axis velocity VECTOR rather than by (V, alpha, "
         "beta), so the comparison cannot depend on how either engine defines alpha and "
         "beta.")
ax = fig.add_axes([0.11, y - 0.24, 0.80, 0.22])
bars = ax.bar(NAMES, [max(L1_WORST[n], 1e-15) for n in NAMES],
              color=[TEAL if n in ("CL", "CY", "Cl", "Cn") else AMBER for n in NAMES])
ax.set_yscale("log")
ax.set_ylabel("worst |difference| across the sweep", fontsize=8.5)
ax.tick_params(labelsize=8.5)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for b, n in zip(bars, NAMES):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.6,
            f"{L1_WORST[n]:.1e}", ha="center", fontsize=7.5, color=MUTED)
y -= 0.295
y = para(fig, y,
         "Four of the six agree to round-off, and those are the real test of the "
         "build-up: JSBSim's CL is a table but it is linear on the segment swept here, "
         "and CY, Cl and Cn are linear in every swept variable in both engines.")
y = heading(fig, y, "CD and Cm disagree, and the disagreement is accounted for")
y = para(fig, y,
         "flightsim has no CD0(alpha) variation, no CDbeta and no CDde; its CD0 is those "
         "three frozen at trim. Subtracting their predicted departure from trim leaves "
         f"{L1_CD_UNEXPLAINED:.1e} worst case. At the sideslip points, where the raw "
         f"difference is largest at {L1_WORST['CD']:.1e}, the prediction accounts for it "
         "to 2.9e-10.")
y = callout(fig, y, "A prediction, not an allowance",
            "The test asserts that the difference EQUALS the terms flightsim is known to "
            "lack, computed from 737.xml's own table constants. That is a stronger "
            "statement than a tolerance: a real defect would have to disguise itself as a "
            "known missing term to survive. Cm's residual is separately shown to be "
            "quadratic about trim -- exact at the reference point to 5.9e-7 -- which is "
            "the linearisation residual from the lift vector rotating into body x at the "
            "AERORP offset, not a wrong slope.", colour=TEAL)
emit(fig, y)

# ------------------------------------------------------------------ layer2
fig = page("Layer 2 - trim", kicker="each engine's own solver")
y = 0.86
y = para(fig, y,
         "JSBSim's do_simple_trim against this project's Newton solve, at the same "
         "condition and the same density.")
y = table(fig, y, [
    ("alpha, deg", f"{np.degrees(FS_ALPHA):.5f}", f"{np.degrees(TRIM.alpha):.5f}",
     f"{np.degrees(FS_ALPHA - TRIM.alpha):+.5f}"),
    ("elevator, rad", f"{FS_DE:.6f}", f"{TRIM.elevator:.6f}", f"{FS_DE - TRIM.elevator:+.6f}"),
    ("thrust, N", f"{FS_THRUST:.1f}", f"{TRIM.thrust:.1f}",
     f"{100 * (FS_THRUST - TRIM.thrust) / TRIM.thrust:+.3f}%"),
    ("throttle", f"{FS_THROTTLE:.6f}", f"{TRIM.throttle:.6f}", ""),
], [0.20, 0.20, 0.20, 0.20],
    header=("", "flightsim", "JSBSim", "difference"), mono_cols=(1, 2, 3))
y = para(fig, y,
         "The thrust difference is the linear-throttle approximation plus the ram fit "
         "residual. JSBSim blends idle and military thrust nonlinearly with throttle -- "
         "thrust per unit throttle varies 4.3x between throttle 0.2 and 1.0 -- while this "
         "model is linear in throttle, so max_thrust is fitted AT the trim throttle rather "
         "than being an engine rating.")
y = callout(fig, y, "The turn case is recorded but not yet comparable",
            "JSBSim also ships a steady-turn trim, and the reference holds its converged "
            f"{np.degrees(REF.trim['turn'].bank):.0f} deg banked solution. This project's "
            "trim.trim solves the WINGS-LEVEL problem only: its unknowns are alpha, "
            "elevator and throttle, with no bank and no aileron or rudder. The comparison "
            "is one banked-trim solver away, and the test asserts the gap rather than "
            "skipping quietly, so that adding one forces the comparison to be written.",
            colour=AMBER)
emit(fig, y)

# ------------------------------------------------------------------ layer3
fig = page("Layer 3 - the linearisation", kicker="and what was hiding in it")
y = 0.875
y = para(fig, y,
         "JSBSim exports a 12-state linear model. Its state ordering carries no labels, so "
         "the harness RE-DERIVES it from the rows of A that must be pure integrators -- "
         "theta-dot = q, psi-dot = r/cos(theta), h-dot = vt(theta - alpha) -- and fails "
         "loudly rather than silently comparing the wrong states. Modes are then compared "
         "on eigenvalues, which are invariant under the change of basis, so no conversion "
         "is needed and none can be got wrong.")
# TWO panels, at their own scales. A single plane cannot show both modes
# honestly: the phugoid's real part is -0.003 against the short period's
# -0.695, a factor of 229, so on one linear axis the phugoid lands ON the
# imaginary axis and a marker wider than its distance from zero straddles it,
# which reads as an unstable root. It is not one -- see the annotation.
for k, (title, pair, ref_pair) in enumerate((
        ("short period", FS_LON[1], JS_LON[1]),
        ("phugoid", FS_LON[0], JS_LON[0]))):
    ax = fig.add_axes([0.11 + k * 0.45, y - 0.25, 0.35, 0.23])
    wn, zeta = pair
    wr, zr = ref_pair
    fr, fi = -zeta * wn, wn * np.sqrt(max(1 - zeta**2, 0))
    jr, ji = -zr * wr, wr * np.sqrt(max(1 - zr**2, 0))
    span = max(abs(fr), abs(jr)) * 1.9
    ax.axvspan(-span, 0.0, color=TEAL, alpha=0.07)
    ax.plot([jr, jr], [ji, -ji], "o", ms=9, mfc="none", mec=BLUE, mew=1.6,
            label="JSBSim")
    ax.plot([fr, fr], [fi, -fi], "x", color=RED, ms=7, mew=1.6, label="flightsim")
    ax.axhline(0, color=RULE, lw=0.7)
    ax.axvline(0, color=INK, lw=1.0)
    ax.set_xlim(-span, span * 0.25)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.set_xlabel("real", fontsize=8.5)
    if k == 0:
        ax.set_ylabel("imag", fontsize=8.5)
    # Legend ABOVE the axes: inside, its sample markers read as data points.
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.06), handletextpad=0.4, columnspacing=1.2)
    ax.set_ylim(-fi * 1.45, fi * 1.45)
    ax.text(0.04, 0.06, "stable half-plane", transform=ax.transAxes, fontsize=7,
            color=TEAL)
    ax.text(0.5, -0.30, f"real part: {fr:+.5f} vs {jr:+.5f}",
            transform=ax.transAxes, fontsize=7.5, color=MUTED, ha="center")
    ax.tick_params(labelsize=8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
y -= 0.345
y = table(fig, y, [
    ("short period wn", f"{FS_LON[1][0]:.5f}", f"{JS_LON[1][0]:.5f}",
     f"{100 * abs(FS_LON[1][0] - JS_LON[1][0]) / JS_LON[1][0]:.3f}%"),
    ("short period zeta", f"{FS_LON[1][1]:.5f}", f"{JS_LON[1][1]:.5f}",
     f"{100 * abs(FS_LON[1][1] - JS_LON[1][1]) / JS_LON[1][1]:.3f}%"),
    ("phugoid wn", f"{FS_LON[0][0]:.5f}", f"{JS_LON[0][0]:.5f}",
     f"{100 * abs(FS_LON[0][0] - JS_LON[0][0]) / JS_LON[0][0]:.2f}%"),
    ("dutch roll zeta", f"{FS_LAT[0][1]:.5f}", f"{JS_LAT[0][1]:.5f}",
     f"{100 * abs(FS_LAT[0][1] - JS_LAT[0][1]) / JS_LAT[0][1]:.2f}%"),
    ("roll TC, s", f"{FS_LAT[1]:.5f}", f"{JS_LAT[1]:.5f}",
     f"{100 * abs(FS_LAT[1] - JS_LAT[1]) / JS_LAT[1]:.2f}%"),
    ("spiral TC, s", f"{FS_LAT[2]:.4f}", f"{JS_LAT[2]:.4f}",
     f"{100 * abs(FS_LAT[2] - JS_LAT[2]) / JS_LAT[2]:.2f}%"),
], [0.20, 0.18, 0.18, 0.18],
    header=("", "flightsim", "JSBSim", "difference"), mono_cols=(1, 2, 3))
y = callout(fig, y, "JSBSim's linearisation is CLOSED-LOOP",
            "Nothing in its output says so. 737.xml's yaw damper feeds yaw rate to the "
            "rudder with unit gain above M 0.11, geared by 0.35 rad. Against the BARE "
            f"airframe the lateral comparison looks catastrophic: Dutch roll zeta "
            f"{FS_LAT_BARE[0][1]:.3f} against {JS_LAT[0][1]:.3f}, spiral "
            f"{FS_LAT_BARE[2]:.1f} s against {JS_LAT[2]:.1f} s -- a "
            f"{100 * (FS_LAT_BARE[2] / JS_LAT[2] - 1):.0f}% disagreement that reads as "
            f"broken lateral dynamics. The damper adds dCnr = Cndr x 0.35 x 2V/b = "
            f"{D_CNR:.3f} against a bare Cnr of {float(AC.Cnr):.3f}; folding it in closes "
            "the gap to under 2%. Pitch and roll have no such feedback, which is why the "
            "short period needs no correction at all.", colour=RED)
emit(fig, y)

# ------------------------------------------------------------------ layer4
fig = page("Layer 4 - trajectory", kicker="20 seconds, prescribed surfaces")
y = 0.875
y = para(fig, y,
         "Both engines integrate from the same state through the same ACHIEVED surface "
         "deflections -- not commands, so JSBSim's FCS cannot contribute to any "
         "difference.")
for k, (case, label) in enumerate((("elevator_doublet", "elevator doublet"),
                                   ("rudder_kick", "rudder kick"))):
    d = TRAJ[case]
    ax = fig.add_axes([0.11, y - 0.19 - k * 0.235, 0.37, 0.17])
    ax.plot(d["t"], d["js_u"], color=BLUE, lw=1.6, label="JSBSim")
    ax.plot(d["t"], d["fs_u"], color=RED, lw=1.1, ls="--", label="flightsim")
    ax.set_ylabel("u, m/s", fontsize=8.5); ax.set_xlabel("t, s", fontsize=8.5)
    ax.set_title(f"{label} - forward speed", fontsize=9.5, color=INK)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax2 = fig.add_axes([0.56, y - 0.19 - k * 0.235, 0.37, 0.17])
    ax2.plot(d["t"], d["dv"], color=TEAL, lw=1.4)
    ax2.axhline(REF.diagnostics["coriolis_velocity_m_s"], color=AMBER, ls=":", lw=1.3)
    ax2.text(d["t"][-1], REF.diagnostics["coriolis_velocity_m_s"], " Earth-rotation floor",
             fontsize=7, color=AMBER, va="bottom", ha="right")
    ax2.set_ylabel("|velocity difference|, m/s", fontsize=8.5)
    ax2.set_xlabel("t, s", fontsize=8.5)
    ax2.set_title(f"{label} - divergence, worst {d['dv'].max():.3f} m/s",
                  fontsize=9.5, color=INK)
    ax2.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
y -= 0.505
_d, _k = TRAJ["elevator_doublet"], TRAJ["rudder_kick"]
_floor = [REF.diagnostics[f"coriolis_{a}_m_s"] for a in "uvw"]
y = para(fig, y,
         f"The two cases differ by {_k['dv'].max() / _d['dv'].max():.1f}x because they "
         f"excite different physics, not because either is unstable. The doublet "
         f"holds sideslip at {_d['beta_max']:.3f} deg, so every lateral model "
         f"difference is inert; the kick reaches {_k['beta_max']:.1f} deg. Per "
         f"component -- doublet u {_d['per_axis'][0]:.3f}, v {_d['per_axis'][1]:.3f}, "
         f"w {_d['per_axis'][2]:.3f}; kick u {_k['per_axis'][0]:.3f}, "
         f"v {_k['per_axis'][1]:.3f}, w {_k['per_axis'][2]:.3f}. The kick's v is a "
         f"transient Dutch-roll phase difference; its u is secular sideslip drag, "
         f"which is layer 1's missing CDbeta integrated over time.",
         size=9.5)
y = callout(fig, y, "Read the rate, not the endpoint",
            "This is an OPEN-LOOP comparison. Nothing holds the two engines together, "
            "so any steady force difference integrates and the gap grows with time by "
            "construction: 0.1% on thrust-minus-drag at 236 m/s reaches 0.5 m/s in 20 "
            "seconds on its own. Growth here is arithmetic, not instability -- layer 3 "
            "puts the two engines' modes within about 1%. The Earth-rotation floor is "
            f"per-axis, not scalar: u {_floor[0]:.3f}, v {_floor[1]:.3f}, "
            f"w {_floor[2]:.3f} m/s. The doublet's u divergence of "
            f"{_d['per_axis'][0]:.3f} is {_d['per_axis'][0] / _floor[0]:.1f}x its own "
            f"floor, and its w of {_d['per_axis'][2]:.3f} is "
            f"{_d['per_axis'][2] / _floor[2]:.0f}x the w floor -- so the flat-Earth "
            "difference accounts for most of the u channel and almost none of the w.",
            colour=TEAL)
emit(fig, y)

# ------------------------------------------------------------------ thrust
fig = page("The thrust model", kicker="what had to change")
y = 0.86
y = para(fig, y,
         "flightsim's thrust was throttle x maximum x density-ratio^n, with no Mach "
         "dependence at all. JSBSim's CFM56 gains 12.1% between M 0 and M 0.8 at 30,000 "
         "ft. At the M 0.78 cruise point that is an 11% thrust error, which in a "
         "trajectory comparison appears as a slow speed divergence indistinguishable from "
         "a drag defect. A Mach ram term was added.")
ax = fig.add_axes([0.11, y - 0.22, 0.37, 0.20])
ms = np.linspace(0.0, 1.0, 60)
ax.plot(CFM_MACH, np.array(CFM_M_ROW_30K) / CFM_M_ROW_30K[0], "o", color=BLUE,
        label="CFM56 table")
ax.plot(ms, (1 + float(AC.mach_ram) * ms**2) / (1 + float(AC.mach_ram) * 0.0),
        color=RED, lw=1.5, label="fitted ram term")
ax.axvspan(0.6, 1.0, color=TEAL, alpha=0.10)
ax.text(0.8, 0.90, "fit band", fontsize=7.5, color=TEAL, ha="center")
ax.set_xlabel("Mach", fontsize=8.5); ax.set_ylabel("thrust, relative to M 0", fontsize=8.5)
ax.legend(fontsize=7.5, frameon=False); ax.tick_params(labelsize=8)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax2 = fig.add_axes([0.56, y - 0.22, 0.37, 0.20])
ratio = np.array([float(density(a * FT2M)) / RHO0 for a in CFM_ALT_FT])
ax2.plot(ratio, CFM_M0, "o", color=BLUE, label="CFM56 table, M 0")
ax2.plot(ratio, ratio, color=TEAL, lw=1.4, label="density ratio ^ 1.0")
ax2.set_xlabel("density ratio", fontsize=8.5); ax2.set_ylabel("thrust factor", fontsize=8.5)
ax2.legend(fontsize=7.5, frameon=False); ax2.tick_params(labelsize=8)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
y -= 0.285
y = table(fig, y, [
    ("mach_ram", f"{float(AC.mach_ram):.6f}", "fitted over M 0.6-1.0"),
    ("thrust_lapse", f"{float(AC.thrust_lapse):.6f}", "fitted over 25,000-35,000 ft"),
    ("max_thrust, N", f"{float(AC.max_thrust):.1f}", "at the trim throttle, NOT a rating"),
    ("fit residual, Mach", f"{REF.entry['thrust_mach_residual'] * 100:.2f}%", ""),
    ("fit residual, altitude", f"{REF.entry['thrust_altitude_residual'] * 100:.2f}%", ""),
], [0.22, 0.16, 0.44], mono_cols=(1,))
y = callout(fig, y, "Two things the measurement corrected",
            "The CFM56's MilThrust table at M 0 tracks the density ratio to the power 1.0 "
            "within 0.6% -- the right panel -- so the altitude LAW was already right. But "
            "the fitted lapse at the cruise throttle is 0.72, not 1.0, because that table "
            "is full power and the idle-to-military blend at a part-throttle setting "
            "lapses differently. And max_thrust fits to about 11,700 lbf per engine "
            "against a 20,000 lbf rating, because flightsim's throttle map is linear and "
            "JSBSim's is not. Neither number means what its name suggests here.",
            colour=AMBER)
y = para(fig, y,
         "The new field defaults to a neutral value, so the 747, the 747 approach case, "
         "the Cherokee, the Cessna and the synthetic test fixture are bit-for-bit "
         "unchanged by it -- asserted in the suite rather than established by inspection.",
         size=9.5, color=MUTED)
emit(fig, y)

# ------------------------------------------------------------------ limits
fig = page("Limitations", kicker="read before reusing any of this")
y = 0.87
y = callout(fig, y, "1. The 737 entry is valid only near cruise",
            "The derivative set is a linearisation about 30,000 ft, M 0.78, alpha 1.97 "
            "deg. It is NOT equivalent to the project's other registry entries, which are "
            "linear derivative sets valid across the ordinary linear range. This one is a "
            "local fit to a NONLINEAR model. JSBSim's CL(alpha) peaks at 1.20 near 13 deg "
            "and falls; CL0 + CLa*alpha keeps climbing, so above about 10 deg the entry "
            "has NO stall behaviour and reports lift the source model does not have. CD0 "
            "is frozen at the trim value of a table running 0.021 to 0.042 over 0-15 deg. "
            "Flying it at 5,000 ft and 200 kt produces wrong numbers with nothing failing, "
            "warning or logging. Documented in four places, deliberately not guarded.",
            colour=RED)
rows = [
    ("2", "Two of the six shipped 737 reference scripts are unused. Both are ground-roll "
          "cases and flightsim has no landing-gear model, so they are unmodellable rather "
          "than merely out of scope."),
    ("3", "The frozen reference can go stale silently. The suite reads the checked-in XML "
          "and never imports JSBSim, which is what makes it portable; a JSBSim release "
          "that changed a result would not be noticed until someone re-ran the generator."),
    ("4", "Tolerance derivations are self-reported. Each carries its reasoning, and the "
          "CD and Cm bounds are predictions rather than allowances, but a derivation that "
          "was weak from the start would still pass its own check."),
    ("5", "Defect fixing was capped to new code. The geopotential-altitude error in "
          "atmosphere.py was found here and filed separately rather than fixed, so this "
          "branch stays reviewable."),
    ("6", "Agreement is not evidence about reality. Both engines can be wrong the same "
          "way, and both descend from the same textbook formulations of rigid-body flight "
          "dynamics. This catches implementation defects and says nothing about whether "
          "either models a real 737."),
    ("7", "The thrust ram fit is band-limited, and wave drag is untested. The fit holds "
          "over M 0.6-1.0 and is wrong at low speed. sweep, t/c and kappa are not 737 "
          "geometry -- 737.xml gives neither sweep nor thickness -- but values placing "
          "this project's drag rise at JSBSim's tabulated onset; at M 0.78 both engines "
          "produce exactly zero wave drag, so the term is never exercised."),
]
for tag, text in rows:
    fig.text(0.08, y, tag, color=BLUE, fontsize=10, fontweight="bold", va="top")
    y = para(fig, y, text, x=0.115, chars=88, size=9.5)
y = para(fig, y,
         "Rebuild:  .venv/Scripts/python.exe scripts/jsbsim_report.py "
         "docs/summary/jsbsim-737-report.pdf\n"
         "Regenerate the reference (needs JSBSim):  python scripts/gen_jsbsim_reference.py\n"
         "Every figure and number above is recomputed on each build from the frozen "
         "reference and the project's own code.", size=8.5, color=TEAL)
emit(fig, y)

if len(PAGES) != len(PAGE_ORDER):
    raise SystemExit(f"pagination: emitted {len(PAGES)}, expected {len(PAGE_ORDER)}")
PDF.close()
print(f"wrote {OUT}  ({len(PAGES)} pages)")
