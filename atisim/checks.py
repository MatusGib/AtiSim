"""Run checks: does this particular flight hold together?

A third tier alongside `verification.py` and `validation.py`, and the distinction
matters. Those two ask whether the MODEL is right -- the arithmetic, and the
agreement with published data -- and they answer it once, in the test suite.
This module asks whether ONE RUN is sensible, and it answers it every time a run
is flown.

**Everything the analysis UI displays is computed here and written to the run
artifact. The UI renders and never computes.** That is not a preference: it is
the protocol `docs/ASSUMPTIONS.md` states for the notebook -- add the
computation to a module, assert it in a test, then display it -- and session 13
exists entirely because session 12 did only the middle step, leaving the notebook
telling a reader something the code had already disproved. A figure is more
persuasive than a print, so the same drift in a UI would be worse.

FOUR OF THESE CHECKS HAVE NEVER FIRED, and that is recorded rather than hidden.
The quaternion norm sits at 2.2e-16, every field the project holds is solenoidal
to round-off, and the Parks vortex produces exactly zero lateral response. Those
are `tripwire` checks: they exist for the field or the integrator change that has
not happened yet. Rendering them as a green tick would claim evidence they do not
provide, so `Check.kind` carries the distinction through to the UI, and
`test_checks.py` gives each one a negative control -- an input that must make it
fail -- because otherwise nothing distinguishes a tripwire from a function that
returns True.

Checks needing a SECOND RUN -- Galilean invariance, strip-versus-point, a halved
step size -- are not here. They belong to a comparison driver that writes its own
artifact, because a check cannot be computed by the run it is about. The UI must
never launch a simulation to fill a panel: `n_steps` is a `static_argname`, so
every distinct dt pays a fresh 0.6-0.9 s JAX compile, and a panel whose contents
depend on the machine is not a check.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from atisim import dynamics
from atisim.aero import air_data
from atisim.aircraft import Aircraft
from atisim.atmosphere import G0, speed_of_sound
from atisim.state import Controls, State, quat_to_dcm
from atisim.units import RAD2DEG

# The declared linear-aero band. Imported rather than restated: `panel.py` owns
# these and PROJECT.md section 7 owns the reasoning. A second copy is a number
# that will eventually disagree with itself.
from atisim.panel import ALPHA_INVALID_DEG, ALPHA_LINEAR_DEG

KINDS = ("gate", "tripwire", "report")


class Check(NamedTuple):
    """One check's result.

    `kind` is what stops the UI from lying:

        gate      it can fail, it has failed, and a failure means the run is bad
        tripwire  it has never fired on anything the project holds. Render the
                  NUMBER and the word, never a bare tick
        report    a measured quantity with no threshold. `passed` is None, and a
                  UI that colours it green has invented a verdict

    `worst_index` is the sample where the check is worst, so a badge can put the
    time cursor on it. None where the check is not per-sample.
    """

    name: str
    value: float
    tolerance: float | None
    kind: str
    passed: bool | None
    detail: str
    worst_index: int | None = None

    def as_dict(self) -> dict:
        """JSON-ready. `numpy` scalars do not survive `json.dumps`."""
        return {
            "name": self.name,
            "value": float(self.value),
            "tolerance": None if self.tolerance is None else float(self.tolerance),
            "kind": self.kind,
            "passed": self.passed,
            "detail": self.detail,
            "worst_index": None if self.worst_index is None else int(self.worst_index),
        }


def _verdict(value, tolerance, kind, name, detail, worst_index=None) -> Check:
    passed = None if kind == "report" else bool(value <= tolerance)
    return Check(name, float(value), tolerance, kind, passed, detail, worst_index)


# ---------------------------------------------------------------------------
# C1 -- quaternion norm
# ---------------------------------------------------------------------------


def quaternion_norm(traj) -> Check:
    """Largest departure from unit norm over the run.

    `integrate.step` re-normalises every step, so this measures whether the
    projection is doing real work. ASSUMPTIONS.md F2: it should be removing
    essentially nothing, and a value that grows means the integrator is drifting
    off the unit sphere while the projection hides it.
    """
    drift = np.abs(1.0 - np.linalg.norm(np.asarray(traj.quat), axis=1))
    return _verdict(
        drift.max(), 1e-12, "tripwire", "quaternion norm",
        "max |1 - ||q|| |. Renormalised every step, so this should sit at "
        "machine epsilon; ASSUMPTIONS.md F2 measured 1.1e-16 over 3000 steps.",
        int(drift.argmax()),
    )


# ---------------------------------------------------------------------------
# C2 -- field divergence
# ---------------------------------------------------------------------------


def field_divergence(field, positions, n_samples: int = 200) -> Check:
    """Largest |div w| along the flown path.

    Every field the project holds is incompressible by construction -- the
    Rankine vortex, the axisymmetric updraft, the purely vertical lee wave, and
    the Oseguera & Bowles microburst, whose whole selling point is that it
    satisfies continuity exactly. So this is zero to round-off today and its
    entire value is the NEXT field somebody writes.

    Sampled along the trajectory rather than on a grid: the aircraft only ever
    experiences the field where it flew, and a grid would spend its budget in
    regions no result depends on.
    """
    positions = np.asarray(positions)
    if len(positions) > n_samples:
        positions = positions[np.linspace(0, len(positions) - 1, n_samples).astype(int)]

    div = jax.jit(jax.vmap(lambda p: jnp.trace(jax.jacfwd(field)(p))))
    values = np.abs(np.asarray(div(jnp.asarray(positions))))
    return _verdict(
        values.max(), 1e-12, "tripwire", "field divergence",
        "max |div w| along the path, 1/s. All four current fields are "
        "solenoidal by construction, so this fires only on a new field.",
        int(values.argmax()),
    )


# ---------------------------------------------------------------------------
# C3, C4 -- energy budget
# ---------------------------------------------------------------------------


def _energy_and_power(traj, ac: Aircraft, field):
    """Mechanical energy and the power of the non-gravitational forces.

    Power comes through `dynamics.specific_force`, which INVERTS the derivative
    sum rather than recomputing it -- so if a force term is ever added to
    `derivatives`, this follows it instead of silently disagreeing.

    The wind is taken from the RECORDED columns, not re-evaluated. Re-evaluating
    is exact for a deterministic field and wrong for a stochastic one, which
    would split the key again and give a different realisation from the one
    flown. `field` is accepted only for the gust rate, which is not recorded at
    the precision this needs; see `recorded_wind_matches_field` for why the
    recorded translational wind is the one to trust.
    """
    del field  # gust rates come from the recorded column too
    pos = np.asarray(traj.pos_ned)
    vel_b = jnp.asarray(traj.vel_body)
    dcm = np.asarray(jax.vmap(quat_to_dcm)(jnp.asarray(traj.quat)))
    vel_ned = np.einsum("nij,nj->ni", dcm, np.asarray(traj.vel_body))

    mass = float(ac.mass)
    energy = 0.5 * mass * (vel_ned**2).sum(axis=1) + mass * float(G0) * (-pos[:, 2])

    controls_hist = Controls(*[jnp.asarray(traj.controls[:, i]) for i in range(4)])

    def power(p, v, q, w, wn, og, el, ai, ru, th):
        s = State(pos_ned=p, vel_body=v, quat=q, omega=w)
        c = Controls(elevator=el, aileron=ai, rudder=ru, throttle=th)
        f_body = dynamics.specific_force(s, c, ac, wn, og) * G0 * ac.mass
        return jnp.dot(f_body, v)

    p = np.asarray(
        jax.vmap(power)(
            jnp.asarray(traj.pos_ned), vel_b, jnp.asarray(traj.quat),
            jnp.asarray(traj.omega), jnp.asarray(traj.wind_ned),
            jnp.asarray(traj.omega_gust), *controls_hist,
        )
    )
    return energy, p


def energy_closure(traj, ac: Aircraft, field) -> Check:
    """|dE - integral(P dt)| over |dE|max.

    THE EXPECTED ORDER IS 1, NOT 4, WHENEVER A WIND MODEL IS PRESENT. The
    trajectory itself is first order in a spatially varying field, because
    `integrate.step` holds the wind across the four RK4 stages -- measured
    1.05 against 3.99 in still air, and 4.05 with the hold removed
    (`test_verification.py`, PROJECT.md section 4). So this residual halving as
    dt halves is the CORRECT behaviour and a residual falling like dt^4 would
    mean the closure was wrong.
    """
    t = np.asarray(traj.t)
    energy, power = _energy_and_power(traj, ac, field)
    work = np.concatenate(
        [[0.0], np.cumsum(np.diff(t) * 0.5 * (power[1:] + power[:-1]))]
    )
    residual = np.abs((energy - energy[0]) - work)
    scale = max(np.abs(energy - energy[0]).max(), 1e-30)
    return _verdict(
        residual.max() / scale, 2e-3, "gate", "energy closure",
        "max |dE - integral(P dt)| / |dE|max. Expected order in dt is 1 with a "
        "wind model and 4 without -- the wind is held across the RK4 stages.",
        int(residual.argmax()),
    )


class EnergyProfile(NamedTuple):
    """Per-step closure residual against position. C4, a `report`.

    This is the sharpest positional diagnostic the project has: the Rankine core
    boundary, where ASSUMPTIONS.md E2 records the one-sided derivatives differ by
    2*V0/r0 with opposite signs, shows up here at 20-350x the run median.
    """

    north: np.ndarray  # (n-1,) m
    per_step: np.ndarray  # (n-1,) J
    median: float


def energy_residual_profile(traj, ac: Aircraft, field) -> EnergyProfile:
    t = np.asarray(traj.t)
    energy, power = _energy_and_power(traj, ac, field)
    work = np.concatenate(
        [[0.0], np.cumsum(np.diff(t) * 0.5 * (power[1:] + power[:-1]))]
    )
    residual = (energy - energy[0]) - work
    per_step = np.abs(np.diff(residual))
    return EnergyProfile(
        north=np.asarray(traj.pos_ned)[1:, 0],
        per_step=per_step,
        median=float(np.median(per_step)),
    )


# ---------------------------------------------------------------------------
# C5 -- trimmed start
# ---------------------------------------------------------------------------


def load_factor_series(traj, ac: Aircraft) -> np.ndarray:
    """n_z at every sample, from the RECORDED wind and gust.

    Recorded rather than re-evaluated, for the reason `_energy_and_power` gives:
    re-evaluating is exact for a deterministic field and returns a different
    realisation for a stochastic one.
    """
    controls_hist = Controls(*[jnp.asarray(traj.controls[:, i]) for i in range(4)])

    def one(p, v, q, w, wn, og, el, ai, ru, th):
        s = State(pos_ned=p, vel_body=v, quat=q, omega=w)
        c = Controls(elevator=el, aileron=ai, rudder=ru, throttle=th)
        return dynamics.load_factor(s, c, ac, wn, og)

    return np.asarray(
        jax.vmap(one)(
            jnp.asarray(traj.pos_ned), jnp.asarray(traj.vel_body),
            jnp.asarray(traj.quat), jnp.asarray(traj.omega),
            jnp.asarray(traj.wind_ned), jnp.asarray(traj.omega_gust), *controls_hist,
        )
    )


def trimmed_start(traj, ac: Aircraft, controls: Controls, field) -> Check:
    """How far out of equilibrium the run begins, in g -- and what that is worth.

    The load factor in trimmed level flight is cos(theta), not 1: the body-normal
    accelerometer reads g*cos(theta). PROJECT.md section 9 session 3 is why this
    exists -- a -6 r0 lead-in starts the aircraft 0.20 g out and understated
    first-core d(theta) by 15%, and nothing caught it at the time.

    **IN A WIND FIELD THERE IS NO TRIMMED START, AND THIS IS A `report`.** The
    Parks far field is 1/r and never dies away: at scripts/vortex.py's own 40
    core radii it is still V0*r0/r = 0.648 m/s, which is 0.157 deg of alpha and
    a MEASURED 0.0384 g of offset. Demanding cos(theta0) there would fail the
    project's canonical run, and picking a tolerance loose enough to pass it
    would be choosing a number to make a check succeed -- which PROJECT.md
    forbids in as many words.

    So the reported quantity is the offset AS A FRACTION OF THE RUN'S OWN PEAK
    EXCURSION, which is the thing a start error actually threatens and is
    dimensionless. For the canonical vortex run that is 0.0384 / 1.235 = 3.1%;
    for the -6 r0 run section 9 describes it is 0.20 / 1.235 = 16%.

    **In STILL AIR it is a `gate`**, because there the right answer exists and is
    exactly cos(theta0). That is the manoeuvring case, whose n_z[0] = 0.9967
    PROJECT.md section 4 records as "the trimmed value, i.e. the lead-in worked".
    """
    del field, controls
    n_z = load_factor_series(traj, ac)
    dcm = np.asarray(quat_to_dcm(jnp.asarray(traj.quat[0])))
    theta0 = float(np.arcsin(np.clip(-dcm[2, 0], -1.0, 1.0)))
    offset = abs(float(n_z[0]) - np.cos(theta0))

    still_air = float(np.abs(np.asarray(traj.wind_ned)).max()) == 0.0
    if still_air:
        return _verdict(
            offset, 1e-3, "gate", "trimmed start",
            f"|n_z[0] - cos(theta0)| = {offset:.3e} g in STILL AIR, where the "
            f"trimmed value is exactly cos(theta0) = {np.cos(theta0):.6f}.",
            0,
        )

    excursion = abs(float(n_z.min() - n_z[0]))
    fraction = offset / max(excursion, 1e-30)
    return Check(
        name="trimmed start",
        value=fraction,
        tolerance=None,
        kind="report",
        passed=None,
        detail=(
            f"starts {offset:.4f} g off cos(theta0) = {np.cos(theta0):.6f}, which "
            f"is {fraction * 100:.1f}% of this run's own peak excursion "
            f"({excursion:.3f} g). In a 1/r field there is no trimmed start -- "
            f"40 core radii still leaves 0.0384 g -- so this is reported, not "
            f"gated. Section 9 session 3's -6 r0 run was 16%."
        ),
        worst_index=0,
    )


# ---------------------------------------------------------------------------
# C8 -- the declared linear-aero band
# ---------------------------------------------------------------------------


class AlphaBand(NamedTuple):
    """C8, reported for the window AND the whole run, because they differ.

    Measured on scripts/vortex.py's canonical run: 8.27 deg inside the first
    core, which is green, against 11.39 deg over the whole run, which is amber,
    with 1.34% of the run outside the linear band. A whole-run-only gate would
    flag a run whose reported result is entirely valid -- the same trap the Fig. 8
    panel draws with its hollow whole-run marker.
    """

    window_deg: float
    whole_run_deg: float
    window_band: str
    whole_run_band: str
    fraction_outside: float
    value: float  # = window_deg, so this can also be read as a Check
    worst_index: int

    def as_check(self) -> Check:
        """THE GATE CONDEMNS ONLY `INVALID`, NOT `MARGINAL`.

        Three bands do not fit in a boolean, and picking the wrong collapse makes
        the check useless in one direction or the other. `scripts/vortex.py`
        prints "INVALID -- this run proves nothing" for the red band alone, and
        the manoeuvring Fig. 8 point is a PUBLISHED result at |alpha| 10.31 deg,
        which is amber. A gate that failed it would be condemning PROJECT.md
        section 4's own table.

        So `passed` answers "does this run prove anything?" and the band word
        travels in `window_band` for the UI to colour amber. A caller wanting the
        stricter reading asks for `window_band == "linear"`.
        """
        return Check(
            name="alpha band",
            value=self.window_deg,
            tolerance=ALPHA_INVALID_DEG,
            kind="gate",
            passed=self.window_band != "INVALID",
            detail=(
                f"|alpha| max: window {self.window_deg:.2f} deg "
                f"({self.window_band}), whole run {self.whole_run_deg:.2f} deg "
                f"({self.whole_run_band}); {self.fraction_outside * 100:.2f}% of "
                f"the run outside the {ALPHA_LINEAR_DEG:g} deg linear band. "
                "Declared band, PROJECT.md section 7 -- not a stall table."
            ),
            worst_index=self.worst_index,
        )


def _band(peak_deg: float) -> str:
    if peak_deg < ALPHA_LINEAR_DEG:
        return "linear"
    if peak_deg < ALPHA_INVALID_DEG:
        return "MARGINAL"
    return "INVALID"


def alpha_band(traj, field, window) -> AlphaBand:
    """Air-relative |alpha| against the declared linear range.

    Air-relative, not inertial: in a Parks encounter the two differ by up to
    7 deg, so the inertial one would report the wrong band entirely.

    The band is symmetric because `aero.py` is odd-symmetric in Δalpha -- a
    pushdown leaves the model exactly as far out of range as an equal pull-up.
    Latent bug (e) was a gauge that could not see half of its own invalid range.
    """
    del field  # alpha comes from the recorded wind, not a re-evaluated field
    vel_rel = jax.vmap(dynamics.relative_velocity)(
        jnp.asarray(traj.vel_body), jnp.asarray(traj.quat), jnp.asarray(traj.wind_ned)
    )
    _, alpha, _ = jax.vmap(air_data)(vel_rel)
    alpha_deg = np.abs(np.asarray(alpha)) * RAD2DEG

    window = np.asarray(window, dtype=bool)
    windowed = alpha_deg[window] if window.any() else alpha_deg
    return AlphaBand(
        window_deg=float(windowed.max()),
        whole_run_deg=float(alpha_deg.max()),
        window_band=_band(float(windowed.max())),
        whole_run_band=_band(float(alpha_deg.max())),
        fraction_outside=float((alpha_deg > ALPHA_LINEAR_DEG).mean()),
        value=float(windowed.max()),
        worst_index=int(alpha_deg.argmax()),
    )


# ---------------------------------------------------------------------------
# C10 -- the recovery band
# ---------------------------------------------------------------------------
def _excursion(x, lo, hi):
    """How far outside [lo, hi], in band widths. Zero inside."""
    return np.maximum(0.0, np.maximum(lo - x, x - hi)) / (hi - lo)


def recovery_band(traj, ac: Aircraft) -> Check:
    """Was this run flown where the aircraft's derivatives were recovered?

    A DIFFERENT QUESTION FROM `alpha_band`, and both are needed. That one asks
    whether alpha left the linear-aero range, which is a property of `aero.py`
    and applies to every aircraft equally. This one asks whether the run left
    the conditions one PARTICULAR derivative set was fitted at, which is a
    property of the data and differs per aircraft. A 737 flown at 5,000 ft and
    200 kt sits comfortably inside the alpha band and is still nonsense.

    The 737 docstring says of exactly that case: "produces numbers that are
    wrong without anything failing, warning or logging". The design that
    recovered it declined a runtime guard and said so -- "Mitigation taken:
    documentation only, deliberately". This is that guard, added after a
    documented-but-unasserted Ixz convention turned out to be wrong for a whole
    comparison. Documentation is not a check.

    Measured in BAND WIDTHS rather than in Mach or metres, so the two axes are
    one number and the number keeps meaning something for an aircraft whose band
    is a different size. 0.0 is inside; 1.0 is one full band width outside.

    Aircraft declaring no band get `report`, never a pass. See `Aircraft`.
    """
    mach_lo, mach_hi = (float(v) for v in ac.valid_mach)
    alt_lo, alt_hi = (float(v) for v in ac.valid_altitude)
    altitude = -np.asarray(traj.pos_ned)[:, 2]

    vel_rel = jax.vmap(dynamics.relative_velocity)(
        jnp.asarray(traj.vel_body), jnp.asarray(traj.quat), jnp.asarray(traj.wind_ned)
    )
    V, _, _ = jax.vmap(air_data)(vel_rel)
    mach = np.asarray(V) / np.asarray(speed_of_sound(jnp.asarray(altitude)))

    axes, worst = [], np.zeros(len(altitude))
    if mach_hi > mach_lo:
        out = _excursion(mach, mach_lo, mach_hi)
        worst = np.maximum(worst, out)
        axes.append(f"Mach {mach.min():.3f}-{mach.max():.3f} against "
                    f"[{mach_lo:g}, {mach_hi:g}]")
    if alt_hi > alt_lo:
        out = _excursion(altitude, alt_lo, alt_hi)
        worst = np.maximum(worst, out)
        axes.append(f"altitude {altitude.min():.0f}-{altitude.max():.0f} m against "
                    f"[{alt_lo:.0f}, {alt_hi:.0f}]")

    if not axes:
        return Check(
            name="recovery band",
            value=0.0,
            tolerance=None,
            kind="report",
            passed=None,
            detail=(
                "this aircraft declares no recovery band, so nothing was checked. "
                "Its derivatives are a linear set valid across the ordinary "
                "linear range rather than a local fit, and its source states no "
                "bound to check against."
            ),
            worst_index=None,
        )

    peak = int(worst.argmax())
    inside = bool(worst[peak] <= 0.0)
    where = (
        "inside the recovery condition: " if inside
        else f"{worst[peak]:.2f} band widths OUTSIDE the recovery condition: "
    )
    consequence = (
        "" if inside
        else " Out there the entry is extrapolating: no layer of the JSBSim "
        "comparison measured it, and nothing else will notice."
    )
    unchecked = (
        "" if len(axes) == 2
        else " The other axis declares no band and was NOT checked."
    )
    return Check(
        name="recovery band",
        value=float(worst[peak]),
        tolerance=0.0,
        kind="gate",
        passed=inside,
        detail=where + "; ".join(axes) + "." + consequence + unchecked,
        worst_index=peak,
    )


# ---------------------------------------------------------------------------
# C9 -- lateral symmetry
# ---------------------------------------------------------------------------


def lateral_symmetry(traj) -> Check:
    """max(|v|, |p|, |r|) for a field that cannot produce any of them.

    The Parks vortex has no east variation and the updraft column is
    axisymmetric about an axis the aircraft flies straight through, so every
    strip on the span sees the same vertical gust and the antisymmetric roll
    integral cancels exactly. That is why flying the strip path moves the vortex
    result by 0.000000 m (PROJECT.md section 4). The same fact makes ANY lateral
    response on these fields a defect rather than a small number.

    Applies only to a symmetric encounter, so the caller decides whether to run
    it. It is exactly zero when it holds, not merely small, which is what lets
    the tolerance be this tight.
    """
    lateral = np.concatenate([
        np.abs(np.asarray(traj.vel_body)[:, 1]),
        np.abs(np.asarray(traj.omega)[:, 0]),
        np.abs(np.asarray(traj.omega)[:, 2]),
    ])
    return _verdict(
        lateral.max(), 1e-10, "tripwire", "lateral symmetry",
        "max(|v|, |p|, |r|). Exactly zero for a field with no spanwise "
        "structure; anything else is a defect, not a small number.",
        int(np.asarray(traj.vel_body)[:, 1].argmax()),
    )


# ---------------------------------------------------------------------------
# C11 -- recorded wind against the analytic field
# ---------------------------------------------------------------------------


class WindMatch(NamedTuple):
    """C11. Both numbers, because the unshifted one is not a defect."""

    shifted: float
    unshifted: float
    peak_gust: float
    value: float  # = shifted
    worst_index: int

    def as_check(self) -> Check:
        return Check(
            name="recorded wind",
            value=self.shifted,
            tolerance=1e-12,
            kind="gate",
            passed=bool(self.shifted <= 1e-12),
            detail=(
                f"max |recorded - field(pos)| shifted one step: {self.shifted:.3e} "
                f"m/s. UNSHIFTED it is {self.unshifted:.4f} m/s, "
                f"{self.unshifted / max(self.peak_gust, 1e-30) * 100:.2f}% of the "
                "peak gust -- that lag is SimState caching the previous step's "
                "wind and is correct behaviour, not a defect."
            ),
            worst_index=self.worst_index,
        )


def recorded_wind_matches_field(traj, field) -> WindMatch:
    """Is the recorded wind the field the aircraft actually flew through?

    `SimState.wind_ned` caches what the PREVIOUS step applied, so the recorded
    column lags the field at the recorded position by exactly one step. A panel
    that overlays them unshifted shows a systematic error that is correct
    behaviour, which is why both numbers come out of here and the detail string
    says which is which.

    This is a `gate` rather than a tripwire because it stops being trivially true
    the moment a stochastic field lands: re-evaluating a Dryden model at a logged
    state splits the key again and returns a different realisation.
    """
    analytic = np.asarray(jax.jit(jax.vmap(field))(jnp.asarray(traj.pos_ned)))
    recorded = np.asarray(traj.wind_ned)
    unshifted = np.abs(recorded - analytic).max(axis=1)
    shifted = np.abs(recorded[1:] - analytic[:-1]).max(axis=1)
    return WindMatch(
        shifted=float(shifted.max()),
        unshifted=float(unshifted.max()),
        peak_gust=float(np.abs(analytic).max()),
        value=float(shifted.max()),
        worst_index=int(shifted.argmax()) + 1,
    )


# ---------------------------------------------------------------------------
# The collected report
# ---------------------------------------------------------------------------


def run_checks(traj, ac: Aircraft, controls: Controls, field, window,
               symmetric: bool = True) -> list[Check]:
    """Every check this run supports, in the order a badge row should read them.

    Gates first, then reports, then tripwires -- so the thing that can condemn
    the run is what the eye lands on, and the four checks that have never fired
    do not occupy the top of the panel.

    `symmetric` says whether the encounter is one for which `lateral_symmetry`
    is a claim at all. It is the caller's statement about the FIELD, and getting
    it wrong would turn a real defect into a skipped check, so it has no default
    guess -- `True` is right for all four fields the project currently holds.
    """
    gates = [
        trimmed_start(traj, ac, controls, field),
        energy_closure(traj, ac, field),
        alpha_band(traj, field, window).as_check(),
        # Listed among the gates even though it degrades to `report` for an
        # aircraft that declares no band. It belongs at the top either way: a
        # reader needs to see "this run was outside the fit" and "nobody checked
        # whether it was" in the same place, and burying the second among the
        # reports is how the second reads as the first.
        recovery_band(traj, ac),
        recorded_wind_matches_field(traj, field).as_check(),
    ]
    profile = energy_residual_profile(traj, ac, field)
    peak = int(profile.per_step.argmax())
    reports = [
        Check(
            name="energy residual peak",
            value=float(profile.per_step[peak] / max(profile.median, 1e-30)),
            tolerance=None,
            kind="report",
            passed=None,
            detail=(
                f"per-step closure residual peaks at {profile.per_step[peak]:.3e} J, "
                f"{profile.per_step[peak] / max(profile.median, 1e-30):.0f}x the run "
                f"median, at north = {profile.north[peak]:+.1f} m. On a Rankine "
                "field this locates the core boundary."
            ),
            worst_index=peak + 1,
        ),
    ]
    tripwires = [quaternion_norm(traj), field_divergence(field, traj.pos_ned)]
    if symmetric:
        tripwires.append(lateral_symmetry(traj))
    return gates + reports + tripwires
