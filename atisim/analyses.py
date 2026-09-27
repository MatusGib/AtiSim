"""Every simulation that is not one open-loop flight: ensembles, the closed
loop, the linear analyses, the verification experiments.

`atisim.run` flies one trimmed aircraft through one field. The engine does much
more than that, and until phase 2 each of the rest lived only in a script: the
seed ensembles (`scripts/cat_ensemble.py`), the autopilot step responses
(`scripts/tune.py`), the modes against CR-2144 (`scripts/checkpoint.py`), the
AD sensitivity screen (`scripts/sensitivity_screen.py`), the gust transfer
sweep (`scripts/gust_transfer_sweep.py`), the discrete gust against Pratt &
Walker (`scripts/discrete_gust.py`), the convergence and conservation
experiments (`atisim.verification`). This module makes each one an ANALYSIS: a
spec (`AnalysisSpec`, JSON like a `RunSpec`), a validation, and a function
that runs the engine's own code and returns a `report.Report` of figures and
tables. The CLI (`atisim analyse`) and the app run them the same way.

The analyses call the engine and the scripts' own functions; they do not
re-derive what those compute. Where a script's number is the reference (the
CR-2144 modes, the 747 gains), it is cited in the report.

No Dash at module level. Plotly is imported when a figure is drawn, so the
`ui` extra is needed to run an analysis, as it is to save a run.
"""

import json
import math
import shlex
import shutil
import time
from pathlib import Path
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import run, studies
from atisim.aircraft import CRUISE, REGISTRY
from atisim.analysis import artifact
from atisim.analysis.report import Report, Section, table, write_report
from atisim.run import Issue, Param
from atisim.units import RAD2DEG

# ---------------------------------------------------------------------------
# The spec
# ---------------------------------------------------------------------------


class AnalysisSpec(NamedTuple):
    """One analysis: which, its parameters, and the run it analyses if any.

    `base` is the `RunSpec` an ensemble, a convergence study, a closed-loop
    flight or a load sensitivity starts from; None for the analyses of an
    aircraft at a flight condition.
    """

    name: str
    analysis: str
    params: dict
    base: run.RunSpec | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "analysis": self.analysis, "params": dict(self.params),
                "base": None if self.base is None else self.base.to_dict()}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text) -> "AnalysisSpec":
        data = json.loads(text) if isinstance(text, (str, bytes)) else dict(text)
        unknown = set(data) - set(cls._fields)
        if unknown:
            raise ValueError(f"unknown AnalysisSpec fields: {', '.join(sorted(unknown))}")
        base = data.get("base")
        data["base"] = None if base is None else run.RunSpec.from_json(base)
        data["params"] = dict(data.get("params", {}))
        return cls(**data)


class Analysis(NamedTuple):
    key: str
    label: str
    family: str
    description: str
    params: tuple
    needs_base: bool
    execute: Callable  # (AnalysisSpec, directory, stage) -> Report
    source: str = ""


FAMILIES = (
    ("Ensembles", "the same run over many turbulence realisations"),
    ("Closed loop", "the autopilot flying"),
    ("Linear analysis", "trim, modes and their sensitivities, without flying"),
    ("Gust response", "the transfer from gust to load, flown against its exact answer"),
    ("Verification", "the integrator and the checks, against closed forms"),
    ("Studies", "a research script from scripts/, run as it is"),
)

# ---------------------------------------------------------------------------
# Shared parameter rows
# ---------------------------------------------------------------------------

AIRCRAFT = Param("aircraft", "Aircraft", "", "choice", tuple(sorted(REGISTRY)), "boeing747")
# Airspeed and altitude default to None: "the aircraft's CRUISE entry", which
# is the only condition each entry is valid near.
AIRSPEED = Param("airspeed_mps", "Airspeed", "m/s", "optional", (), None)
ALTITUDE = Param("altitude_m", "Altitude", "m", "optional", (), None)
CONDITION = (AIRCRAFT, AIRSPEED, ALTITUDE)


def _value(aspec: AnalysisSpec, param: Param):
    return aspec.params.get(param.name, param.default)


def condition(aspec: AnalysisSpec) -> tuple[str, float, float]:
    """(aircraft, V, H), with CRUISE filling what the spec leaves empty."""
    key = _value(aspec, AIRCRAFT)
    cruise = CRUISE[key]
    V = aspec.params.get("airspeed_mps")
    H = aspec.params.get("altitude_m")
    return key, float(cruise["airspeed"] if V is None else V), \
        float(cruise["altitude"] if H is None else H)


def _trimmed(key: str, V: float, H: float):
    """Trim, refusing a solution that is not a flight condition, as `run` does."""
    from atisim import trim

    spec = run.BLANK._replace(aircraft=key, airspeed_mps=V, altitude_m=H)
    x, residual = run._trim(spec)
    return x, residual, trim


# ---------------------------------------------------------------------------
# Figures (plotly, imported when drawn)
# ---------------------------------------------------------------------------


def _fig(title: str, subtitle: str, height: int = 360, legend: bool = True):
    import plotly.graph_objects as go

    from atisim.analysis import figures

    fig = go.Figure()
    fig._atisim = (figures, height, title, subtitle, legend)
    return fig, go, figures


def _finish(fig):
    figures, height, title, subtitle, legend = fig._atisim
    return figures._base(fig, height, title=title, subtitle=subtitle, legend=legend)


def _line(go, x, y, name, color, dash=None, width=1.6, **kw):
    return go.Scatter(x=np.asarray(x), y=np.asarray(y), name=name, mode="lines",
                      line=dict(color=color, width=width, dash=dash), **kw)


# ---------------------------------------------------------------------------
# Ensembles
# ---------------------------------------------------------------------------

MEMBERS = Param("members", "Members (seeds)", "", "integer", (), 8)
# The TPAWS reduction's encounter length (`scripts/phase2_common.py`).
ENCOUNTER_SECONDS = 30.0


def stochastic_part(spec: run.RunSpec) -> str | None:
    """Which part of a spec carries the seed: "wind", "overlay", or None."""
    if spec.wind.kind in run.STOCHASTIC_KINDS:
        return "wind"
    if spec.overlay is not None:
        return "overlay"
    return None


def with_seed_offset(spec: run.RunSpec, offset: int) -> run.RunSpec:
    """The spec with its stochastic component's seed moved on by `offset`.

    Only the random part changes. `wind.field_model` returns the PRNG key
    untouched, so every member meets any deterministic structure (a vortex) at
    the same place -- the experiment design an error bar on a deterministic
    encounter needs.
    """
    part = stochastic_part(spec)
    if part == "wind":
        seed = run.param_value(spec.wind, run._param(spec.wind.kind, "seed"))
        return spec._replace(wind=spec.wind._replace(
            params={**spec.wind.params, "seed": int(seed) + offset}))
    if part == "overlay":
        seed = run.param_value(spec.overlay, run._param(spec.overlay.kind, "seed"))
        return spec._replace(overlay=spec.overlay._replace(
            params={**spec.overlay.params, "seed": int(seed) + offset}))
    raise ValueError("this run has no stochastic component to vary")


def ensemble(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import insitu
    from atisim.analysis import runs as runs_mod

    base, n = aspec.base, int(_value(aspec, MEMBERS))
    t = None
    loads, rows, members = [], [], []
    for i in range(n):
        spec = with_seed_offset(base, i)
        stage(f"flying member {i + 1} of {n}")
        flown = run.fly(spec)
        rel = f"members/m{i:03d}"
        run.write(flown, directory / rel)
        members.append(rel)
        enc = flown.encounter
        t = enc.t if t is None else t
        n_z = np.asarray(enc.n_z)
        loads.append(n_z)
        d = n_z - n_z.mean()
        seed = (spec.wind if stochastic_part(spec) == "wind" else spec.overlay).params["seed"]
        rows.append([i, seed, float(n_z.max()), float(n_z.min()), float(d.std()),
                     runs_mod.verdict([c.as_dict() for c in flown.report])])
    stage("reducing")
    loads = np.vstack([x[: min(len(v) for v in loads)] for x in loads])
    t = np.asarray(t)[: loads.shape[1]]
    dt = float(base.dt)
    seconds = min(ENCOUNTER_SECONDS, float(t[-1] - t[0]))
    factors = np.concatenate([
        insitu.encounter_peak_factors(x - x.mean(), dt, seconds) for x in loads])

    fig, go, figures = _fig("Load factor across the ensemble",
                            "Median and 5-95% band over the members; member 0 drawn "
                            "in full. Every member flies the same deterministic "
                            "structure; only the seed differs.")
    lo, med, hi = np.percentile(loads, [5, 50, 95], axis=0)
    fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[hi, lo[::-1]], fill="toself",
                             fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
                             name="5-95%", hoverinfo="skip"))
    fig.add_trace(_line(go, t, med, "median", figures.SERIES[0]))
    fig.add_trace(_line(go, t, loads[0], "member 0", figures.REFERENCE, width=1.0))
    fig.update_xaxes(title_text="time  s")
    fig.update_yaxes(title_text="n_z  g")
    envelope = _finish(fig)

    fig, go, figures = _fig("Peak factors, TPAWS' reduction",
                            f"Peak over the {seconds:.0f} s encounter divided by the "
                            "rms inside a 5 s window (insitu.encounter_peak_factors), "
                            "every encounter of every member.")
    fig.add_trace(go.Histogram(x=factors, marker_color=figures.SERIES[0], nbinsx=20,
                               name="peak factor"))
    fig.update_xaxes(title_text="peak factor")
    fig.update_yaxes(title_text="encounters")
    histogram = _finish(fig)

    stage("spectra and exceedance")
    from atisim import response

    spectra = [response.spectrum(x, dt) for x in loads]
    f_hz, psd = spectra[0][0], np.mean([p for _, p in spectra], axis=0)
    keep = f_hz > 0
    fig, go, figures = _fig("Spectrum of the load factor",
                            "One-sided PSD of n_z, mean over the members "
                            "(response.spectrum: Hann window, mean removed).",
                            legend=False)
    fig.add_trace(_line(go, f_hz[keep], psd[keep], "PSD", figures.SERIES[0]))
    fig.update_xaxes(type="log", title_text="frequency  Hz")
    fig.update_yaxes(type="log", title_text="PSD  g^2/Hz")
    psd_fig = _finish(fig)

    dev = loads - loads.mean(axis=1, keepdims=True)
    levels = np.linspace(0.0, float(np.abs(dev).max()), 40)[1:]
    up = np.mean([response.exceedance(d, dt, levels) for d in dev], axis=0)
    down = np.mean([response.exceedance(-d, dt, levels) for d in dev], axis=0)
    fig, go, figures = _fig("Exceedance",
                            "Upcrossings per second of each level of dn_z about the "
                            "member's mean, mean over the members "
                            "(response.exceedance). Per second of record, not per "
                            "flight hour.")
    fig.add_trace(_line(go, levels, np.where(up > 0, up, np.nan), "positive",
                        figures.SERIES[0]))
    fig.add_trace(_line(go, levels, np.where(down > 0, down, np.nan), "negative",
                        figures.SERIES[1], dash="dash"))
    fig.update_xaxes(title_text="|dn_z|  g")
    fig.update_yaxes(type="log", title_text="upcrossings per second")
    exceed_fig = _finish(fig)

    peaks = loads.max(axis=1) - loads.mean(axis=1)
    summary = table(
        ["Quantity", "Mean", "Std", "Min", "Max"],
        [["Peak n_z above the mean, g", peaks.mean(), peaks.std(), peaks.min(), peaks.max()],
         ["Peak factor", factors.mean(), factors.std(), factors.min(), factors.max()]],
    )
    return Report(
        analysis="ensemble", title=f"Ensemble of {n}: {base.name}", spec=aspec.to_dict(),
        aircraft=base.aircraft, members=tuple(members),
        summary=f"{n} members; mean peak factor {factors.mean():.3f} "
                f"(sd {factors.std():.3f}, {len(factors)} encounters)",
        caveats=tuple(base.caveats),
        sections=[
            Section("Spread", "", envelope),
            Section("Peak factors", "", histogram, summary),
            Section("Spectrum", "", psd_fig),
            Section("Exceedance", "", exceed_fig),
            Section("Members", "Open a member in Results for its deep dive.", None,
                    table(["Member", "Seed", "Max n_z", "Min n_z", "rms dn_z", "Checks"],
                          rows)),
        ],
    )


# ---------------------------------------------------------------------------
# Trim and modes
# ---------------------------------------------------------------------------

# CR-2144 Table IX-5, flight condition 9, as `scripts/checkpoint.py` holds it.
_FC9 = {"phugoid": (0.0673, 0.0489), "short period": (0.964, 0.387)}
_FC9_SOURCE = "CR-2144 Table IX-5, FC 9 denominator (scripts/checkpoint.py)"


def _mode_references(key: str, V: float, H: float) -> dict:
    """Published (wn, zeta) for this aircraft at this condition, with the source."""
    from atisim import validation

    cruise = CRUISE[key]
    at_cruise = math.isclose(V, cruise["airspeed"], rel_tol=1e-9) and \
        math.isclose(H, cruise["altitude"], rel_tol=1e-9, abs_tol=1e-6)
    if not at_cruise:
        return {}
    if key == "boeing747":
        return {m: (wn, z, _FC9_SOURCE) for m, (wn, z) in _FC9.items()}
    if key == "boeing747_approach":
        r = validation.REFERENCES
        return {
            "phugoid": (r["747pa_phugoid_wn"].value, r["747pa_phugoid_zeta"].value,
                        r["747pa_phugoid_wn"].source),
            "short period": (r["747pa_short_period_wn"].value,
                             r["747pa_short_period_zeta"].value,
                             r["747pa_short_period_wn"].source),
        }
    return {}


def modes(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import validation

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    stage("trimming")
    x, residual, _ = _trimmed(key, V, H)
    alpha, elevator, throttle = (float(v) for v in x)
    stage("linearising")
    lon = validation.longitudinal_modes(ac, alpha, elevator, throttle, V, H)
    dutch, roll_tau, spiral_tau = validation.lateral_modes(ac, alpha, elevator,
                                                           throttle, V, H)
    A_lon = np.asarray(validation.longitudinal_matrix(ac, alpha, elevator, throttle, V, H))
    refs = _mode_references(key, V, H)
    rows = []
    for label, (wn, zeta) in zip(("phugoid", "short period"), lon):
        ref = refs.get(label)
        rows.append([label, wn, zeta, ref[0] if ref else None, ref[1] if ref else None,
                     ref[2] if ref else "no published value at this condition"])
    rows.append(["dutch roll", dutch[0], dutch[1], None, None, ""])
    rows.append(["roll subsidence tau, s", roll_tau, None, None, None, ""])
    rows.append(["spiral tau, s", spiral_tau, None, None, None, ""])

    eig = np.linalg.eigvals(A_lon)
    fig, go, figures = _fig("Longitudinal roots",
                            "Eigenvalues of the body-axis [u, w, q, theta] plant, "
                            "linearised by jacfwd of the real dynamics.")
    fig.add_trace(go.Scatter(x=eig.real, y=eig.imag, mode="markers", name="model",
                             marker=dict(symbol="x", size=11, color=figures.SERIES[0])))
    for label, (wn, zeta, _src) in refs.items():
        re, im = -zeta * wn, wn * math.sqrt(max(0.0, 1 - zeta ** 2))
        fig.add_trace(go.Scatter(x=[re, re], y=[im, -im], mode="markers",
                                 name=f"{label}, published",
                                 marker=dict(symbol="circle-open", size=12,
                                             color=figures.REFERENCE)))
    fig.update_xaxes(title_text="real  1/s")
    fig.update_yaxes(title_text="imaginary  rad/s")
    return Report(
        analysis="modes", title=f"Modes: {key}", spec=aspec.to_dict(), aircraft=key,
        summary=f"short period wn {lon[-1][0]:.4f} rad/s, zeta {lon[-1][1]:.4f}",
        sections=[
            Section("Trim", f"{key} at {V:.2f} m/s, {H:.0f} m. Residual {residual:.1e}.",
                    None, table(["alpha, deg", "elevator, deg", "throttle"],
                                [[alpha * RAD2DEG, elevator * RAD2DEG, throttle]])),
            Section("Modes", "Longitudinal from the 4-state [u, w, q, theta] plant; "
                             "lateral from the 4-state [v, p, r, phi] reduction "
                             "(validation.lateral_modes).", None,
                    table(["Mode", "wn, rad/s", "zeta", "Published wn", "Published zeta",
                           "Source"], rows)),
            Section("Root locus", "", _finish(fig)),
        ],
    )


def trim_analysis(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import trim, verification
    from atisim.aircraft import B747_BUFFET_CL_UNCERTAINTY, buffet_cl
    from atisim.atmosphere import density, speed_of_sound

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    stage("trimming")
    x, residual, _ = _trimmed(key, V, H)
    alpha, elevator, throttle = (float(v) for v in x)
    history = np.asarray(verification.newton_residual_history(
        jnp.array(V), jnp.array(H), ac, iterations=8))
    v_md = float(trim.minimum_drag_speed(ac, jnp.array(H)))
    rho = float(density(jnp.array(H)))
    mach = V / float(speed_of_sound(jnp.array(H)))
    CL = float(ac.mass) * 9.80665 / (0.5 * rho * V ** 2 * float(ac.S))
    rows = [["alpha", alpha * RAD2DEG, "deg"], ["elevator", elevator * RAD2DEG, "deg"],
            ["throttle", throttle, "-"], ["residual norm", residual, "-"],
            ["minimum-drag speed V_md", v_md, "m/s"], ["margin V - V_md", V - v_md, "m/s"],
            ["Mach", mach, "-"], ["CL (weight / q S)", CL, "-"]]
    text = ""
    if key.startswith("boeing747"):
        limit = float(buffet_cl(mach))
        rows.append(["initial-buffet CL at this Mach", limit, "-"])
        text = (f"Buffet boundary: Boeing D6-30643 / CR-114494 p. 2.0-38, read to "
                f"+-{B747_BUFFET_CL_UNCERTAINTY}. Margin {limit - CL:+.3f} in CL.")
    fig, go, figures = _fig("Newton convergence",
                            "Residual norm after each Newton iteration from trim.py's "
                            "own start (verification.newton_residual_history). "
                            "Quadratic convergence halves the exponent's gap each step.",
                            legend=False)
    fig.add_trace(go.Scatter(x=np.arange(len(history)), y=np.maximum(history, 1e-17),
                             mode="lines+markers", line=dict(color=figures.SERIES[0])))
    fig.update_yaxes(type="log", title_text="residual norm")
    fig.update_xaxes(title_text="iteration")
    return Report(
        analysis="trim", title=f"Trim: {key}", spec=aspec.to_dict(), aircraft=key,
        summary=f"alpha {alpha * RAD2DEG:.3f} deg, residual {residual:.1e}",
        sections=[Section("Trim solution", f"{key} at {V:.2f} m/s, {H:.0f} m. " + text,
                          None, table(["Quantity", "Value", "Unit"], rows)),
                  Section("Convergence", "", _finish(fig))],
    )


def mode_sensitivity(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import sensitivity

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    stage("differentiating the longitudinal plant")
    lon, _, _, sep_lon, x, res = sensitivity.mode_sensitivity(
        ac, V, H, axis="longitudinal", fields=sensitivity.LONGITUDINAL_FIELDS)
    stage("differentiating the lateral plant")
    lat, _, _, sep_lat, _, _ = sensitivity.mode_sensitivity(
        ac, V, H, axis="lateral", fields=sensitivity.LATERAL_FIELDS)
    entries = []  # scripts/sensitivity_screen.py section D, as data
    for field in sensitivity.LONGITUDINAL_FIELDS:
        ms = sensitivity.oscillatory_modes(lon[field])
        if len(ms) != 2:
            continue
        p = float(getattr(ac, field))
        for label, m in (("phugoid", ms[0]), ("short period", ms[-1])):
            entries.append((field, f"{label} wn", sensitivity.elasticity(m.dwn, p, m.wn)))
            entries.append((field, f"{label} zeta",
                            sensitivity.elasticity(m.dzeta, p, m.zeta)))
    for field in sensitivity.LATERAL_FIELDS:
        osc = sensitivity.oscillatory_modes(lat[field])
        rls = sensitivity.real_modes(lat[field])
        p = float(getattr(ac, field))
        if len(osc) == 1:
            entries.append((field, "dutch roll wn",
                            sensitivity.elasticity(osc[0].dwn, p, osc[0].wn)))
            entries.append((field, "dutch roll zeta",
                            sensitivity.elasticity(osc[0].dzeta, p, osc[0].zeta)))
        if len(rls) >= 2:
            entries.append((field, "roll tau", sensitivity.elasticity(rls[0].dtau, p, rls[0].tau)))
            entries.append((field, "spiral tau",
                            sensitivity.elasticity(rls[1].dtau, p, rls[1].tau)))
    sections = [Section(
        "What this is",
        f"Elasticity (dQ/dp)(p/Q): per cent of the mode quantity per per cent of the "
        f"input, one at a time, no interaction term. {key} at {V:.2f} m/s, {H:.0f} m; "
        f"eigenvalue separation longitudinal {sep_lon:.4f}, lateral {sep_lat:.4f}. "
        "An elasticity of exactly zero means the input is not used at this condition.")]
    for mode in ("short period wn", "short period zeta", "phugoid wn", "phugoid zeta",
                 "dutch roll wn", "dutch roll zeta", "roll tau", "spiral tau"):
        ranked = sorted([(f, e) for f, m, e in entries if m == mode and np.isfinite(e)
                         and abs(e) > 1e-10], key=lambda r: -abs(r[1]))[:8]
        if not ranked:
            continue
        fig, go, figures = _fig(mode, "The eight largest elasticities.", 300, legend=False)
        fig.add_trace(go.Bar(x=[e for _, e in ranked][::-1], y=[f for f, _ in ranked][::-1],
                             orientation="h", marker_color=figures.SERIES[0]))
        fig.update_xaxes(title_text="elasticity")
        sections.append(Section(mode, "", _finish(fig)))
    sections.append(Section("Every elasticity", "", None,
                            table(["Input", "Mode quantity", "Elasticity"], entries)))
    return Report(analysis="mode-sensitivity", title=f"Mode sensitivity: {key}",
                  spec=aspec.to_dict(), aircraft=key,
                  summary=f"{len(entries)} elasticities", sections=sections)


_SWEPT = ('mass', 'c', 'CD0', 'e', 'CL0', 'CLa', 'CLq', 'CLde', 'Cm0', 'Cma', 'Cmq',
          'Cmde', 'max_thrust', 'thrust_lapse', 'CYb', 'CYp', 'CYr', 'CYdr', 'Clb', 'Clp',
          'Clr', 'Clda', 'Cldr', 'Cnb', 'Cnp', 'Cnr', 'Cnda', 'Cndr')
FIELD = Param("field", "Coefficient", "", "choice", _SWEPT, "Cmq")
SPAN = Param("span", "Range either side", "fraction", "positive", (), 0.3)
SWEEP_POINTS = Param("points", "Points", "", "integer", (), 7)
_QUANTITIES = ("phugoid wn", "phugoid zeta", "short period wn", "short period zeta",
               "dutch roll wn", "dutch roll zeta", "roll tau", "spiral tau")


def coefficient_sweep(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    """validation.sweep: the finite version of what mode-sensitivity differentiates.

    The aircraft is re-trimmed at every value, so each point is a flight
    condition and not a perturbation of one.
    """
    from atisim import validation

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    field = _value(aspec, FIELD)
    span, n = float(_value(aspec, SPAN)), int(_value(aspec, SWEEP_POINTS))
    p0 = float(getattr(ac, field))
    scale = abs(p0) if p0 != 0 else 1.0
    values = p0 + np.linspace(-span, span, n) * scale

    def quantity(swept, alpha, elevator, throttle):
        (ph, sp) = validation.longitudinal_modes(swept, alpha, elevator, throttle, V, H)
        dutch, roll_tau, spiral_tau = validation.lateral_modes(swept, alpha, elevator,
                                                               throttle, V, H)
        return (ph[0], ph[1], sp[0], sp[1], dutch[0], dutch[1], roll_tau, spiral_tau)

    stage(f"re-trimming and linearising at {n} values of {field}")
    out = np.asarray(validation.sweep(ac, field, values, quantity, V, H), dtype=float)
    rows, sections = [], []
    for j, name in enumerate(_QUANTITIES):
        y = out[:, j]
        if not np.all(np.isfinite(y)):
            rows.append([name, None, None, None])
            continue
        slope, intercept, worst = validation.affine_fit(values, y)
        mid = y[n // 2]
        rows.append([name, float(slope), float(slope * p0 / mid) if mid else None,
                     float(worst)])
    for group, members in (("Longitudinal", _QUANTITIES[:4]), ("Lateral", _QUANTITIES[4:])):
        fig, go, figures = _fig(f"{group} modes against {field}",
                                "Each quantity divided by its value at the aircraft's "
                                f"own {field} = {p0:.6g}. Re-trimmed at every point.")
        for i, name in enumerate(members):
            j = _QUANTITIES.index(name)
            y = out[:, j]
            ref = y[n // 2]
            if np.all(np.isfinite(y)) and ref:
                fig.add_trace(go.Scatter(x=values, y=y / ref, mode="lines+markers",
                                         name=name, line=dict(color=figures.SERIES[i % 3],
                                                              dash=None if i < 3 else "dot")))
        fig.update_xaxes(title_text=field)
        fig.update_yaxes(title_text="relative to the aircraft's value")
        sections.append(Section("", "", _finish(fig)))
    return Report(
        analysis="coefficient-sweep", title=f"Sweep of {field}: {key}",
        spec=aspec.to_dict(), aircraft=key,
        summary=f"{field} from {values[0]:.4g} to {values[-1]:.4g}, {n} points",
        sections=[Section("Fit", "Slope of an affine fit through the swept points, the "
                                 "elasticity at the aircraft's value, and the worst relative "
                                 "residual of the fit (validation.affine_fit). An affine "
                                 "relation fits to round-off.", None,
                          table(["Quantity", "Slope", "Elasticity", "Worst residual"], rows)),
                  *sections,
                  Section("Points", "", None, table(["Value", *_QUANTITIES],
                                                    [[v, *r] for v, r in zip(values, out)]))],
    )


def load_sensitivity(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import sensitivity, wind

    base = aspec.base
    ac = REGISTRY[base.aircraft]
    geo = run.geometry(base)
    field = run.build_field(base)
    model = run.wind_model(base, field) or wind.field_model(field)
    stage("flying the run and its tangents (one forward pass)")
    Q, grads, elas, extra = sensitivity.load_elasticities(
        ac, model, float(base.airspeed_mps), float(base.altitude_m),
        start_north=geo.start_north, n_steps=int(round(geo.seconds / base.dt)),
        dt=base.dt, window=geo.window if geo.window else (-math.inf, math.inf),
        moving_air=base.wind.kind == "MehtaHannibal")
    ranked = sorted(elas.items(), key=lambda kv: -abs(kv[1]) if np.isfinite(kv[1]) else 0)
    fig, go, figures = _fig("Load elasticities",
                            "Elasticity of the peak-to-peak n_z in the window with "
                            "respect to each aircraft field, all in one forward-mode "
                            "pass (sensitivity.load_elasticities).", 420, legend=False)
    top = [r for r in ranked if np.isfinite(r[1])][:12]
    fig.add_trace(go.Bar(x=[e for _, e in top][::-1], y=[f for f, _ in top][::-1],
                         orientation="h", marker_color=figures.SERIES[0]))
    fig.update_xaxes(title_text="elasticity")
    return Report(
        analysis="load-sensitivity", title=f"Load sensitivity: {base.name}",
        spec=aspec.to_dict(), aircraft=base.aircraft,
        summary=f"peak-to-peak n_z {Q:.4f} g",
        caveats=tuple(base.caveats),
        sections=[Section("Headline", f"Peak-to-peak n_z in the window: {Q:.6f} g. "
                                      "One at a time; no interaction term.",
                          _finish(fig)),
                  Section("Every field", "", None,
                          table(["Field", "Value", "dQ/dp", "Elasticity"],
                                [[f, extra["values"][f], grads[f], e] for f, e in ranked]))],
    )


# ---------------------------------------------------------------------------
# Gust response
# ---------------------------------------------------------------------------

POINTS = Param("points", "Wavelengths", "", "integer", (), 6)
WL_MIN = Param("wavelength_min", "Shortest wavelength", "m", "positive", (), 200.0)
WL_MAX = Param("wavelength_max", "Longest wavelength", "m", "positive", (), 20000.0)
GUST_LAG = Param("gust_lag", "Kussner gust lag", "", "choice", ("off", "on"), "off")
WING_TAIL = Param("wing_tail", "Wing-tail gust delay", "", "choice", ("off", "on"), "off")


def gust_transfer(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import airframe, gust

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    lag = _value(aspec, GUST_LAG) == "on"
    arm = None
    if _value(aspec, WING_TAIL) == "on":
        arm = run.tail_arm(key)
        if arm is None:
            arm = float(airframe.effective_tail_arm(ac) * ac.c)
    n = int(_value(aspec, POINTS))
    wavelengths = np.geomspace(float(_value(aspec, WL_MIN)), float(_value(aspec, WL_MAX)), n)
    measured = []
    for i, wl in enumerate(wavelengths):
        stage(f"flying wavelength {i + 1} of {n} ({wl:.0f} m)")
        measured.append(gust.measure_gust_transfer(ac, V, H, float(wl), gust_lag=lag,
                                                   tail_arm=arm))
    stage("comparing with the exact answer")
    Omega = np.array([m["spatial_frequency"] for m in measured])
    freq = np.array([m["frequency_hz"] for m in measured])
    H_meas = np.array([m["H"] for m in measured])
    # Each point against the exact answer at ITS OWN mean ground speed, as
    # scripts/gust_transfer_sweep.py compares them; the drawn line uses the mean.
    H_at = np.array([complex(gust.gust_transfer(ac, V, H, m["spatial_frequency"],
                                                ground_speed=m["ground_speed"],
                                                gust_lag=lag, tail_arm=arm))
                     for m in measured])
    ground = float(np.mean([m["ground_speed"] for m in measured]))
    dense = np.geomspace(Omega.min(), Omega.max(), 200)
    H_exact = np.asarray(gust.gust_transfer(ac, V, H, dense, ground_speed=ground,
                                            gust_lag=lag, tail_arm=arm))
    f_dense = dense * ground / (2 * np.pi)
    fig, go, figures = _fig("Gust to load: magnitude",
                            "|H|, g per m/s of upward gust. Line: gust.gust_transfer, "
                            "the linearised answer. Markers: flown through a frozen "
                            "sinusoid and fitted (gust.measure_gust_transfer).")
    fig.add_trace(_line(go, f_dense, np.abs(H_exact), "exact", figures.REFERENCE))
    fig.add_trace(go.Scatter(x=freq, y=np.abs(H_meas), mode="markers", name="flown",
                             marker=dict(color=figures.SERIES[0], size=9)))
    fig.update_xaxes(type="log", title_text="encounter frequency  Hz")
    fig.update_yaxes(type="log", title_text="|H|  g/(m/s)")
    mag = _finish(fig)
    fig, go, figures = _fig("Gust to load: phase", "Degrees; same line and markers.")
    fig.add_trace(_line(go, f_dense, np.degrees(np.angle(H_exact)), "exact", figures.REFERENCE))
    fig.add_trace(go.Scatter(x=freq, y=np.degrees(np.angle(H_meas)), mode="markers",
                             name="flown", marker=dict(color=figures.SERIES[0], size=9)))
    fig.update_xaxes(type="log", title_text="encounter frequency  Hz")
    fig.update_yaxes(title_text="phase  deg")
    phase = _finish(fig)
    rows = [[wl, f, abs(hm), abs(he), 100 * (abs(hm) / abs(he) - 1),
             np.degrees(np.angle(hm / he)), m["dt"]]
            for wl, f, hm, he, m in zip(wavelengths, freq, H_meas, H_at, measured)]
    worst = max(abs(r[4]) for r in rows)
    return Report(
        analysis="gust-transfer", title=f"Gust transfer: {key}", spec=aspec.to_dict(),
        aircraft=key, summary=f"worst magnitude error {worst:.3f}%",
        caveats=((run.GUST_LAG_CAVEAT,) if lag else ()) + ((run.WING_TAIL_CAVEAT,) if arm else ()),
        sections=[Section("Magnitude", "", mag), Section("Phase", "", phase),
                  Section("Points", "", None, table(
                      ["Wavelength, m", "f, Hz", "|H| flown", "|H| exact",
                       "Magnitude error, %", "Phase error, deg", "dt, s"], rows))],
    )


GRADIENTS = Param("gradients", "Gradient distances", "chords", "text", (), "6, 12.5, 25, 50")
PEAK = Param("peak", "Derived gust U_de", "m/s", "positive", (), 1.0)


def _floats(text) -> list[float]:
    if isinstance(text, (int, float)):
        return [float(text)]
    return [float(x) for x in str(text).replace(";", ",").split(",") if x.strip()]


def pratt_walker(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import gust, vortex_viz, wind
    from atisim.atmosphere import G0, density

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    c = float(ac.c)
    peak = float(_value(aspec, PEAK))
    rho = float(density(jnp.array(H)))
    formula = float(gust.pratt_walker(ac, rho, V, peak, float(G0)))
    rows, runs_ = [], []
    chords = _floats(_value(aspec, GRADIENTS))
    for i, ch in enumerate(chords):
        stage(f"flying gradient distance {i + 1} of {len(chords)} ({ch:g} chords)")
        gd = ch * c
        lead = 40.0 * c  # scripts/discrete_gust.py's fly_gust
        field = wind.one_minus_cosine_gust(peak, gd, start_north=lead)
        seconds = (lead + 2.0 * gd + 60.0 * c) / V + 40.0
        enc = vortex_viz.fly(ac, field, V, H, label=f"H={gd:.0f}", start_north=0.0,
                             seconds=seconds, dt=0.01, window=(-np.inf, np.inf),
                             window_name="whole run", stage_sampled=True)
        n0 = float(enc.n_z[0])
        dn = float(np.max(enc.n_z) - n0)
        rows.append([ch, gd, dn, formula if ch == 12.5 else None,
                     100 * (dn / formula - 1) if ch == 12.5 else None])
        runs_.append((ch, enc))
    fig, go, figures = _fig("Load through a 1 - cos gust",
                            "n_z increment against time from the gust's start, one "
                            "line per gradient distance.")
    for i, (ch, enc) in enumerate(runs_):
        fig.add_trace(_line(go, enc.t, np.asarray(enc.n_z) - float(enc.n_z[0]),
                            f"H = {ch:g} chords", figures.SERIES[i % 3] if i < 3
                            else figures.REFERENCE))
    fig.update_xaxes(title_text="time  s")
    fig.update_yaxes(title_text="dn_z  g")
    return Report(
        analysis="pratt-walker", title=f"Discrete gust vs Pratt & Walker: {key}",
        spec=aspec.to_dict(), aircraft=key,
        summary=f"Pratt & Walker {formula:.4f} g at U_de {peak:g} m/s",
        sections=[Section("Against the formula",
                          "Pratt & Walker, NACA Report 1206, Eq. (5): dn = K_g U_de V "
                          "CL_a rho S / (2 W). K_g was fitted at H = 12.5 chords, so that "
                          "row is the only strict comparison; the others are context. "
                          "The model overshoots the formula there by 17.57% at the "
                          "validation condition. This is a known difference, not a fault.",
                          None, table(["H, chords", "H, m", "Flown dn, g",
                                       "Formula dn, g", "Difference, %"], rows)),
                  Section("Responses", "", _finish(fig))],
    )


# ---------------------------------------------------------------------------
# Closed loop
# ---------------------------------------------------------------------------

# scripts/tune.py's steps. Empty sizes take its rule: 300 m and 15 m/s above
# 100 m/s, 150 m and 8 m/s below, because a step gentle for a 747 is violent for
# a Cherokee.
ALT_STEP = Param("altitude_step", "Altitude step", "m", "optional", (), None)
SPEED_STEP = Param("speed_step", "Airspeed step", "m/s", "optional", (), None)
HEADING_STEP = Param("heading_step", "Heading step", "deg", "positive", (), 30.0)
SECONDS = Param("seconds", "Duration of each step", "s", "positive", (), 300.0)
AP_DT = 0.02  # scripts/tune.py's step


def _closed_loop(key, V, H, targets, seconds, wind_model=None, start_north=0.0):
    from atisim import autopilot as ap_mod
    from atisim import integrate, trim, wind
    from atisim.sensors import sense

    ac = REGISTRY[key]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    state = state._replace(pos_ned=jnp.array([start_north, 0.0, -H]))
    controls = trim.trimmed_controls(x[1], x[2])
    gains = ap_mod.GAINS[key]
    ap = ap_mod.engage(sense(state), controls, targets, gains, ac)
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    (_, _), (hist, ctrl) = ap_mod.closed_loop_rollout(
        sim, ap, targets, gains, jnp.array(AP_DT), ac, int(round(seconds / AP_DT)),
        wind_model=wind.zero_wind if wind_model is None else wind_model)
    return hist, ctrl


def _loop_figures(t, hist, ctrl, targets_row):
    from atisim.state import quat_to_euler

    alt = -np.asarray(hist.pos_ned)[:, 2]
    spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
    euler = np.asarray(jax.vmap(quat_to_euler)(hist.quat))
    out = []
    for title, y, unit, target in (
            ("Altitude", alt, "m", targets_row.get("altitude")),
            ("Airspeed (body velocity)", spd, "m/s", targets_row.get("airspeed")),
            ("Heading", np.degrees(euler[:, 2]), "deg", targets_row.get("heading")),
            ("Bank", np.degrees(euler[:, 0]), "deg", None),
            ("Pitch", np.degrees(euler[:, 1]), "deg", None)):
        fig, go, figures = _fig(title, "", 240)
        fig.add_trace(_line(go, t, y, title.lower(), figures.SERIES[0]))
        if target is not None:
            fig.add_trace(_line(go, [t[0], t[-1]], [target, target], "target",
                                figures.REFERENCE, dash="dot"))
        fig.update_yaxes(title_text=unit)
        out.append(_finish(fig))
    fig, go, figures = _fig("Controls", "Elevator, aileron and rudder in degrees; "
                                        "throttle on the right scale is 0-1.", 260)
    for i, name in enumerate(("elevator", "aileron", "rudder")):
        fig.add_trace(_line(go, t, np.degrees(np.asarray(getattr(ctrl, name))), name,
                            figures.SERIES[i]))
    fig.add_trace(_line(go, t, np.asarray(ctrl.throttle) * 10, "throttle x 10",
                        figures.REFERENCE, dash="dot"))
    fig.update_yaxes(title_text="deg")
    out.append(_finish(fig))
    return out, alt, spd, np.degrees(euler[:, 2])


def _settle(t, signal, target, band):
    err = np.abs(signal - target)
    outside = np.where(err > band)[0]
    return float(t[outside[-1]]) if len(outside) else 0.0


def _tune_summary(signal, target, tol, band):
    """scripts/tune.py's `summarise`: final error, settle time, overshoot, OK.

    Settle time is the last time outside the band, on tune.py's time base (the
    first sample at t = 0), so the numbers here are the script's to the digit.
    """
    t = np.arange(len(signal)) * AP_DT
    err = np.abs(signal - target)
    outside = np.where(err > band)[0]
    settle = float(t[outside[-1]]) if len(outside) else 0.0
    over = (signal.max() - target) if signal[0] < target else (target - signal.min())
    return float(signal[-1] - target), settle, float(over), bool(err[-1] < tol)


def _signals(hist):
    from atisim.state import quat_to_euler

    euler = np.asarray(jax.vmap(quat_to_euler)(hist.quat))
    return {"altitude": -np.asarray(hist.pos_ned)[:, 2],
            "airspeed": np.linalg.norm(np.asarray(hist.vel_body), axis=1),
            "heading": np.degrees(euler[:, 2])}


def step_response(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    """scripts/tune.py as an analysis: the same four steps, the same grading."""
    from atisim import autopilot as ap_mod
    from atisim import trim

    key, V, H = condition(aspec)
    ac = REGISTRY[key]
    big = V > 100.0
    d_alt = _value(aspec, ALT_STEP) or (300.0 if big else 150.0)
    d_spd = _value(aspec, SPEED_STEP) or (15.0 if big else 8.0)
    d_hdg = float(_value(aspec, HEADING_STEP))
    seconds = float(_value(aspec, SECONDS))
    v_md = float(trim.minimum_drag_speed(ac, jnp.array(H)))

    def targets(alt, hdg_deg, spd):
        return ap_mod.Targets(altitude=jnp.array(alt), heading=jnp.array(math.radians(hdg_deg)),
                              airspeed=jnp.array(spd))

    # (title, targets, seconds, (loop, target, unit, tol, band), held (...), graded)
    cases = [
        (f"Altitude step +{d_alt:g} m", targets(H + d_alt, 0.0, V), seconds,
         ("altitude", H + d_alt, "m", 10.0, 15.0), ("airspeed", V, "m/s", 3.0, 3.0), True),
        (f"Heading step +{d_hdg:g} deg", targets(H, d_hdg, V), seconds,
         ("heading", d_hdg, "deg", 1.0, 2.0), ("altitude", H, "m", 30.0, 40.0), True),
    ]
    for sign in (-1.0, 1.0):
        target = V + sign * d_spd
        cases.append((f"Airspeed step {sign * d_spd:+g} m/s", targets(H, 0.0, target),
                      seconds * 4.0 / 3.0, ("airspeed", target, "m/s", 1.0, 1.5),
                      ("altitude", H, "m", 30.0, 40.0), target > v_md))
    rows, sections, all_ok = [], [], True
    for i, (title, tg, secs, stepped, held, graded) in enumerate(cases):
        stage(f"flying {title.lower()} ({i + 1} of {len(cases)})")
        hist, _ = _closed_loop(key, V, H, tg, secs)
        sig = _signals(hist)
        t = np.arange(len(sig["altitude"])) * AP_DT
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        from atisim.analysis import figures

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        for row, (loop, target, unit, tol, band) in enumerate((stepped, held), start=1):
            err, settle, over, ok = _tune_summary(sig[loop], target, tol, band)
            verdict = ("OK" if ok else "FAIL") + ("" if graded else ", not graded")
            if graded:
                all_ok &= ok
            rows.append([title, loop + (" (hold)" if row == 2 else ""), unit, err, settle,
                         over, verdict])
            fig.add_trace(_line(go, t,
                                sig[loop], loop, figures.SERIES[row - 1]), row=row, col=1)
            fig.add_hline(y=target, line=dict(color=figures.REFERENCE, dash="dot", width=1),
                          row=row, col=1)
            fig.update_yaxes(title_text=f"{loop}  {unit}", row=row, col=1)
        fig.update_xaxes(title_text="time  s", row=2, col=1)
        sub = "" if graded else (f"Below V_md {v_md:.1f} m/s the loop pairing inverts: "
                                 "reported, not graded (scripts/tune.py).")
        sections.append(Section("", "", figures._base(fig, 380, title=title, subtitle=sub,
                                                       legend=False)))
    stage("bumpless engagement")
    hist, ctrl = _closed_loop(key, V, H, targets(H, 0.0, V), 20.0)
    excursion = float(np.abs(_signals(hist)["altitude"] - H).max())
    trim_elevator = float(_trimmed(key, V, H)[0][1])
    first = float(np.asarray(ctrl.elevator)[0])
    bumpless = excursion < 1.0
    all_ok &= bumpless
    rows.append(["Bumpless engagement, 20 s", "altitude excursion", "m", excursion, None,
                 None, "OK" if bumpless else "FAIL"])
    rows.append(["Bumpless engagement, 20 s", "first elevator minus trim", "deg",
                 math.degrees(first - trim_elevator), None, None, ""])
    return Report(
        analysis="step-response", title=f"Autopilot steps: {key}", spec=aspec.to_dict(),
        aircraft=key, summary=("all graded steps OK" if all_ok else "SOME STEPS FAILED")
        + f"; V_md {v_md:.1f} m/s",
        caveats=(f"Gains: autopilot.GAINS['{key}'], dt {AP_DT} s as scripts/tune.py.",),
        sections=[Section("Steps", "scripts/tune.py's four steps and its grading: final "
                                   "error within the tolerance; settle time is the last time "
                                   "outside the band. A step below the minimum-drag speed is "
                                   "reported and not graded.", None,
                          table(["Step", "Loop", "Unit", "Final error", "Settle, s",
                                 "Overshoot", "Verdict"], rows))] + sections,
    )


def closed_loop(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import autopilot as ap_mod
    from atisim import wind

    base = aspec.base
    geo = run.geometry(base)
    field = run.build_field(base)
    model = run.wind_model(base, field) or wind.field_model(field)
    V, H = float(base.airspeed_mps), float(base.altitude_m)
    targets = ap_mod.Targets(altitude=jnp.array(H), heading=jnp.array(0.0),
                             airspeed=jnp.array(V))
    stage("flying the closed loop through the field")
    hist, ctrl = _closed_loop(base.aircraft, V, H, targets, geo.seconds, model,
                              geo.start_north)
    t = np.arange(1, len(np.asarray(hist.pos_ned)) + 1) * AP_DT
    figs, alt, spd, _ = _loop_figures(t, hist, ctrl, {"altitude": H, "airspeed": V,
                                                      "heading": 0.0})
    return Report(
        analysis="closed-loop", title=f"Autopilot through {base.name}",
        spec=aspec.to_dict(), aircraft=base.aircraft,
        summary=f"largest altitude excursion {np.abs(alt - H).max():.1f} m",
        caveats=tuple(base.caveats) + (
            "Autopilot engaged: the Fig. 8 discriminator is defined on OPEN-LOOP "
            "flight (vortex_viz.fly), so these runs are not comparable with it.",
            f"Step {AP_DT} s (scripts/tune.py), not the run's own dt.",),
        sections=[Section("Hold", f"The autopilot holds {H:.0f} m, heading 0 and "
                                  f"{V:.1f} m/s through the field. Largest altitude "
                                  f"excursion {np.abs(alt - H).max():.2f} m, airspeed "
                                  f"{np.abs(spd - V).max():.2f} m/s.")]
        + [Section("", "", f) for f in figs],
    )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

REFINEMENTS = Param("refinements", "Refinements", "", "integer", (), 3)


def convergence(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import verification

    base = aspec.base
    k = int(_value(aspec, REFINEMENTS))
    dts = [base.dt / 2 ** i for i in range(k + 1)]
    flights = []
    for i, dt in enumerate(dts):
        stage(f"flying at dt {dt:g} s ({i + 1} of {len(dts)})")
        spec = base._replace(dt=dt)
        started = time.time()
        flights.append((dt, run.fly(spec).encounter, time.time() - started))
    stage("comparing with the finest step")
    fine = flights[-1][1]
    rows, errors = [], []
    for dt, enc, took in flights[:-1]:
        step = int(round(dt / dts[-1]))
        n = min(len(enc.t), len(fine.t) // step)
        err_nz = float(np.max(np.abs(np.asarray(enc.n_z)[:n]
                                     - np.asarray(fine.n_z)[step - 1::step][:n])))
        err_th = float(np.max(np.abs(np.asarray(enc.theta)[:n]
                                     - np.asarray(fine.theta)[step - 1::step][:n])))
        errors.append(err_nz)
        rows.append([dt, err_nz, err_th * RAD2DEG, took])
    rows.append([dts[-1], None, None, flights[-1][2]])
    order = float(verification.fitted_order(dts[:-1], errors)) if len(errors) >= 2 else None
    fig, go, figures = _fig("Error against step",
                            "Largest |n_z - n_z(finest)| over the run, on the common "
                            "time grid. The fitted slope is the observed order "
                            "(verification.fitted_order).", legend=False)
    fig.add_trace(go.Scatter(x=dts[:-1], y=errors, mode="lines+markers",
                             line=dict(color=figures.SERIES[0])))
    fig.update_xaxes(type="log", title_text="dt  s")
    fig.update_yaxes(type="log", title_text="max |dn_z|  g")
    text = ("Observed order " + (f"{order:.2f}" if order is not None else "not fitted "
            "(needs three or more steps)") + ". RK4 is fourth order in still air and "
            "through a stage-sampled smooth field; a Rankine core's kink makes the "
            "order non-monotone (ASSUMPTIONS E9), and a held wind makes it first order "
            "(E4).")
    return Report(
        analysis="convergence", title=f"Convergence: {base.name}", spec=aspec.to_dict(),
        aircraft=base.aircraft,
        summary="observed order " + (f"{order:.2f}" if order is not None else "n/a"),
        caveats=tuple(base.caveats),
        sections=[Section("Refinement", text, _finish(fig),
                          table(["dt, s", "max |dn_z|, g", "max |dtheta|, deg",
                                 "wall time, s"], rows))],
    )


def verification_suite(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import integrate, trim, verification
    from atisim.aircraft import inertia_tensor
    from atisim.state import State, euler_to_quat
    from atisim.tests.conftest import make_test_aircraft

    sections = []
    stage("harmonic oscillator through the real RK4")
    dts = [0.2, 0.1, 0.05, 0.025]
    errs, order = verification.oscillator_refinement(dts)
    sections.append(Section(
        "RK4 on a harmonic oscillator",
        f"integrate.rk4_step driven directly; exact solution a rotation. Observed "
        f"order {float(order):.3f}; expected 4.", None,
        table(["dt", "error"], list(zip(dts, np.asarray(errs).tolist())))))

    stage("the 6-DOF rollout against a fine step")
    ac747 = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    six_dts = np.array([1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32])  # test_verification's window
    six_errs, six_order = verification.fixed_control_refinement(ac747, V, H, six_dts,
                                                                dt_ref=1.0 / 1024.0)
    sections.append(Section(
        "The 6-DOF rollout, fixed controls",
        "The real 747 at CRUISE after an elevator step, still air, against a 1/1024 s "
        f"reference (verification.fixed_control_refinement). Observed order "
        f"{float(six_order):.3f}; expected 4. Through a wind held across the stages "
        "it falls to 1 (ASSUMPTIONS E4).", None,
        table(["dt", "final position error, m"],
              list(zip(six_dts.tolist(), np.asarray(six_errs).tolist())))))

    stage("free fall through a swinging wind")
    ac = verification.without_aerodynamics(REGISTRY["boeing747"])
    got = verification.free_fall_through_a_swinging_wind(ac, dt=0.02, n=300)
    sections.append(Section(
        "Free fall through a swinging wind",
        "An accelerating air mass must not push the aeroplane: the wind enters "
        "through the relative velocity only. Aerodynamics removed; closed-form "
        "reference.", None,
        table(["Quantity", "Value"], [
            ["max position error, m", got.max_position_error],
            ["peak wind flown through, m/s", got.peak_wind],
            ["peak dW/dt, m/s^2", got.peak_dwdt],
            ["elapsed, s", got.elapsed]])))

    stage("torque-free rotation")
    I1, I2, I3 = 1420.0, 4070.0, 4780.0  # test_verification.py's body
    omega0 = np.array([0.6, 0.0, 0.9])
    inertia = inertia_tensor(I1, I2, I3, 0.0)
    zeroed = {k: jnp.array(0.0) for k in (
        "CL0", "CLa", "CLq", "CLde", "Cm0", "Cma", "Cmq", "Cmde", "CD0", "CYb", "CYp",
        "CYr", "CYdr", "Clb", "Clp", "Clr", "Clda", "Cldr", "Cnb", "Cnp", "Cnr", "Cnda",
        "Cndr", "max_thrust")}
    body = make_test_aircraft()._replace(inertia=inertia, inertia_inv=jnp.linalg.inv(inertia),
                                         **zeroed)
    state = State(pos_ned=jnp.array([0.0, 0.0, -3000.0]), vel_body=jnp.array([60.0, 0.0, 0.0]),
                  quat=euler_to_quat(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)),
                  omega=jnp.array(omega0))
    dt, n = 0.002, 1500
    _, traj = integrate.rollout(integrate.init_sim(state, jax.random.PRNGKey(0)),
                                trim.trimmed_controls(jnp.array(0.0), jnp.array(0.0)),
                                jnp.array(dt), body, n)
    t = np.arange(1, n + 1) * dt
    exact = verification.torque_free_omega(I1, I2, I3, omega0, t)
    got_w = np.asarray(traj.omega).T
    fig, go, figures = _fig("Torque-free rotation", "Body rates, integrated (solid) "
                            "against Landau & Lifshitz's closed form (dotted).")
    for i, name in enumerate("pqr"):
        fig.add_trace(_line(go, t, got_w[i], name, figures.SERIES[i]))
        fig.add_trace(_line(go, t, exact[i], f"{name} exact", figures.REFERENCE, dash="dot"))
    fig.update_xaxes(title_text="time  s")
    fig.update_yaxes(title_text="rad/s")
    sections.append(Section(
        "Torque-free rotation",
        f"Conservation is not correctness: the trajectory is checked, not the "
        f"invariant. Max |omega - exact| {np.abs(got_w - exact).max():.2e} rad/s.",
        _finish(fig)))
    return Report(analysis="verification", title="Verification suite", spec=aspec.to_dict(),
                  summary=f"RK4 order {float(order):.2f}; free-fall error "
                          f"{got.max_position_error:.1e} m",
                  sections=sections)


# The frozen JSBSim reference's encounters, one per case (tests/data/
# jsbsim_vortex_reference.xml; test_jsbsim_vortex.EXPECTED_KEYS).
CASE = Param("case", "Encounter", "", "choice", ("cimarron", "hannibal", "morton", "mehta"),
             "hannibal")
# scripts/vortex_compare.py's arms: which gust-gradient terms AtiSim carries.
# JSBSim samples the wind at one point and has no writable gust-rate input, so
# "translational" is the only like-for-like arm.
_ARMS = {"translational": (False, False), "+ alphadot": (False, True),
         "+ omega_gust": (True, False), "+ both": (True, True)}
ARM = Param("arm", "AtiSim gradient terms", "", "choice", tuple(_ARMS), "+ both")


def _partial_field_model(field, omega_gust: bool, alphadot: bool):
    """`wind.field_model` with either gradient term off (scripts/vortex_compare.py)."""
    from atisim import wind

    def model(wind_state, state, key, dt):
        del dt
        w = field(state.pos_ned)
        og = wind.gust_rates(state.pos_ned, state.quat, field) if omega_gust \
            else jnp.zeros(3)
        ad = wind.gust_alphadot(state.pos_ned, state.quat, state.vel_body, field) \
            if alphadot else jnp.array(0.0)
        return w, og, wind_state, key, ad
    return model


def _fly_like_jsbsim(enc, arm: str):
    """AtiSim from JSBSim's recorded state, through the same field.

    One field, one starting state, one density: the three conditions
    scripts/vortex_compare.py sets out. The cores hang from the density-matched
    altitude so the relative geometry is the one JSBSim flew.
    """
    from atisim import dynamics, vortex_viz, wind
    from atisim.state import Controls, State, euler_to_quat

    ac = REGISTRY[enc.aircraft]
    v = enc.values
    altitude = v["matched_altitude"]
    if enc.cores is None:
        north, down = jnp.array([v["core_north"]]), jnp.array([-altitude])
    else:
        z = v["altitude"] + enc.cores[1]
        north, down = jnp.array(enc.cores[0]), jnp.array(-(altitude - z))
    array = wind.VortexArray(north=north, down=down, r0=jnp.array(v["r0"]),
                             v0=jnp.array(v["v0"]),
                             cos_dpsi=jnp.array(v.get("cos_dpsi", 1.0)))
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731
    state = State(pos_ned=jnp.array([0.0, 0.0, -altitude]),
                  vel_body=jnp.array(enc.initial.vel_body),
                  quat=euler_to_quat(*(jnp.array(x) for x in enc.initial.euler)),
                  omega=jnp.array(enc.initial.omega))
    controls = Controls(elevator=jnp.array(enc.initial.controls[0]),
                        aileron=jnp.array(enc.initial.controls[1]),
                        rudder=jnp.array(enc.initial.controls[2]),
                        throttle=jnp.array(enc.initial.throttle))
    flown = vortex_viz.fly_from_state(
        ac, field, state, controls, label=f"{enc.case} {arm}", seconds=v["duration"],
        dt=0.01, window=enc.window_bounds(),
        window_name="core" if enc.cores is None else "array",
        wind_model=_partial_field_model(field, *_ARMS[arm]))
    datum = float(dynamics.load_factor(state, controls, ac, jnp.zeros(3), jnp.zeros(3)))
    return flown, datum


def cross_code(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    from atisim import jsbsim_vortex_ref

    case, arm = _value(aspec, CASE), _value(aspec, ARM)
    ref = jsbsim_vortex_ref.load()
    enc = next(e for (c, _), e in ref.encounters.items() if c == case)
    v = enc.values
    arms = ["translational"] + ([arm] if arm != "translational" else [])
    flown = {}
    for i, name in enumerate(arms):
        stage(f"flying AtiSim, {name} ({i + 1} of {len(arms)})")
        flown[name] = _fly_like_jsbsim(enc, name)
    stage("comparing")
    single = enc.cores is None
    js_north = np.array([s.north for s in enc.samples])
    x_js = (js_north - v["core_north"]) / v["r0"] if single else js_north
    xlabel = "north from the core  core radii" if single else "north  m"
    js_nz = np.array([s.Nz for s in enc.samples]) - v["trim_Nz"]
    js_th = np.degrees(np.array([s.theta for s in enc.samples]) - enc.initial.euler[1])
    lo, hi = enc.window_bounds()
    c = enc.core_response()
    js_span = (abs(c[0] - c[1]), abs(c[2] - c[3]))
    figs = []
    for title, js_y, getter, unit in (
            ("Load factor increment", js_nz, lambda r, d: np.asarray(r.n_z) - d, "g"),
            ("Pitch increment", js_th,
             lambda r, d: np.degrees(np.asarray(r.theta) - enc.initial.euler[1]), "deg")):
        fig, go, figures = _fig(title, "Each engine from its own still-air datum. The "
                                       "shaded band is the comparison window.", 300)
        wx = [(lo - v["core_north"]) / v["r0"], (hi - v["core_north"]) / v["r0"]] \
            if single else [lo, hi]
        fig.add_vrect(x0=wx[0], x1=wx[1], fillcolor="rgba(120,120,120,0.10)", line_width=0)
        fig.add_trace(_line(go, x_js, js_y, "JSBSim (frozen)", figures.REFERENCE, width=2.0))
        for i, (name, (r, datum)) in enumerate(flown.items()):
            x = (np.asarray(r.north) - v["core_north"]) / v["r0"] if single \
                else np.asarray(r.north)
            fig.add_trace(_line(go, x, getter(r, datum), f"AtiSim, {name}",
                                figures.SERIES[i], dash=None if i == 0 else "dash"))
        fig.update_xaxes(title_text=xlabel)
        fig.update_yaxes(title_text=unit)
        figs.append(_finish(fig))
    rows = [["JSBSim", js_span[0], js_span[1], None, None]]
    for name, (r, _) in flown.items():
        w = np.asarray(r.window)
        span_n = float(np.ptp(np.asarray(r.n_z)[w]))
        span_t = float(np.ptp(np.degrees(np.asarray(r.theta)[w])))
        rows.append([f"AtiSim, {name}", span_n, span_t,
                     100 * (span_n / js_span[0] - 1), 100 * (span_t / js_span[1] - 1)])
    return Report(
        analysis="cross-code", title=f"Cross-code: {case} against JSBSim",
        spec=aspec.to_dict(), aircraft=enc.aircraft,
        summary=f"like-for-like n_z span error {rows[1][3]:+.1f}%",
        caveats=("JSBSim samples the wind at one point and cannot take a gust rate, so "
                 "only the translational arm is like-for-like; the other arm shows what "
                 "the gradient terms add.",),
        sections=[
            Section("Setup", f"{enc.aircraft} at {v['altitude']:.0f} m (AtiSim at the "
                             f"density-matched {v['matched_altitude']:.1f} m), started from "
                             f"JSBSim's recorded state. {ref.jsbsim_version.split('[')[0].strip()}"
                             f", frozen in tests/data/jsbsim_vortex_reference.xml. "
                             f"r0 {v['r0']:.1f} m, V0 {v['v0']:.2f} m/s "
                             f"({enc.radius_source}).", None,
                    table(["Engine", "n_z span, g", "theta span, deg",
                           "n_z error, %", "theta error, %"], rows)),
            *[Section("", "", f) for f in figs],
        ],
    )


# ---------------------------------------------------------------------------
# Studies: a script from scripts/, run as it is (plan section 3.5)
# ---------------------------------------------------------------------------

SCRIPT = Param("script", "Script", "", "choice", studies.names(), "sanity")
ARGUMENTS = Param("arguments", "Arguments", "", "arguments", (), "")
MINUTES = Param("minutes", "Time limit", "min", "positive", (), 30.0)
LOG_LINES = 400  # of the log shown on the page; log.txt holds all of it
CSV_ROWS = 30


def _csv_preview(path: Path):
    import csv

    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None, 0

    def cell(value):
        try:
            return float(value) if value.strip() else value
        except ValueError:
            return value

    body = [[cell(v) for v in r] for r in rows[1:]]
    return table(rows[0], body[:CSV_ROWS]), len(body)


def study(aspec: AnalysisSpec, directory: Path, stage) -> Report:
    script = studies.find(_value(aspec, SCRIPT))
    args = shlex.split(_value(aspec, ARGUMENTS) or "")
    minutes = float(_value(aspec, MINUTES))
    shown = " ".join(["scripts/" + script.path.name, *args])
    stage(f"running {shown}")
    out = studies.run_script(script, directory, on_line=getattr(stage, "log", None),
                             minutes=minutes, args=args)
    (directory / "log.txt").write_text(out.log + "\n")
    last = next((ln.strip() for ln in reversed(out.log.splitlines()) if ln.strip()),
                "It printed nothing.")
    if out.returncode is None:
        verdict = f"Stopped at the time limit, {minutes:g} min."
    elif out.returncode != 0:
        verdict = f"Exited with status {out.returncode} after {out.elapsed_s:.1f} s."
    else:
        verdict = f"Ran for {out.elapsed_s:.1f} s."
    caveats = []
    if out.returncode != 0:
        caveats.append("The script did not finish cleanly. Its own message is at the end of "
                       "the log. A script that needs data the public tree does not hold "
                       "fails this way.")
    sections = []
    images = [(rel, n) for rel, n in out.outputs if rel.lower().endswith(studies.IMAGES)]
    tables = [(rel, n) for rel, n in out.outputs if rel.lower().endswith(".csv")]
    other = [(rel, n) for rel, n in out.outputs if (rel, n) not in images + tables]
    wrote = (f"It wrote {len(out.outputs)} file{'s' if len(out.outputs) != 1 else ''} "
             "that a study keeps (images, CSV, JSON, Markdown, text). Each is a copy in "
             "this study's files/ folder; the script's own copy stays where it wrote it."
             if out.outputs else "It wrote no image, CSV, JSON, Markdown or text file.")
    for rel, reason in out.skipped:
        wrote += f" Not kept: {rel} ({reason})."
    sections.append(Section("The script", f"{script.description} {verdict} {wrote}", None,
                            table(["Command", "Working directory", "Exit status"],
                                  [[f"python {shown}", "the repository root",
                                    "stopped" if out.returncode is None
                                    else out.returncode]])))
    for rel, _ in images:
        sections.append(Section(rel, image=f"files/{rel}"))
    for rel, _ in tables:
        preview, n = _csv_preview(directory / "files" / rel)
        text = (f"The first {CSV_ROWS} of {n} rows." if n > CSV_ROWS
                else f"{n} row{'s' if n != 1 else ''}.")
        sections.append(Section(rel, text, None, preview, files=(f"files/{rel}",)))
    if other:
        sections.append(Section("Other files it wrote", "",
                                files=tuple(f"files/{rel}" for rel, _ in other)))
    lines = out.log.splitlines()
    sections.append(Section(
        "What it printed",
        f"The last {LOG_LINES} of {len(lines)} lines; log.txt holds them all."
        if len(lines) > LOG_LINES else "", log="\n".join(lines[-LOG_LINES:]),
        files=("log.txt",)))
    return Report("study", f"Study: {script.path.name}", aspec.to_dict(), sections,
                  tuple(caveats), summary=f"{verdict} {last}")


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------

ANALYSES: dict[str, Analysis] = {a.key: a for a in (
    Analysis("ensemble", "Seed ensemble", "Ensembles",
             "Fly a run with a stochastic field over many seeds; spread and TPAWS "
             "peak factors.", (MEMBERS,), True, ensemble,
             "scripts/cat_ensemble.py, scripts/phase2_baseline_records.py"),
    Analysis("step-response", "Autopilot step response", "Closed loop",
             "scripts/tune.py's altitude, heading and airspeed steps through the "
             "autopilot, graded as it grades them.",
             CONDITION + (ALT_STEP, SPEED_STEP, HEADING_STEP, SECONDS), False,
             step_response, "scripts/tune.py"),
    Analysis("closed-loop", "Autopilot through a field", "Closed loop",
             "Fly a run with the autopilot holding altitude, heading and airspeed.",
             (), True, closed_loop, "autopilot.closed_loop_rollout"),
    Analysis("trim", "Trim", "Linear analysis",
             "The trim solution, its Newton convergence, V_md and the buffet margin.",
             CONDITION, False, trim_analysis, "scripts/checkpoint.py"),
    Analysis("modes", "Modes", "Linear analysis",
             "Longitudinal and lateral modes, against the published values where held.",
             CONDITION, False, modes, "scripts/checkpoint.py; validation.REFERENCES"),
    Analysis("mode-sensitivity", "Mode sensitivity", "Linear analysis",
             "AD elasticities of every mode to every aircraft field.",
             CONDITION, False, mode_sensitivity, "scripts/sensitivity_screen.py"),
    Analysis("coefficient-sweep", "Coefficient sweep", "Linear analysis",
             "Vary one aircraft coefficient, re-trim at every value, and follow the modes.",
             CONDITION + (FIELD, SPAN, SWEEP_POINTS), False, coefficient_sweep,
             "validation.sweep; scripts/checkpoint.py"),
    Analysis("load-sensitivity", "Load sensitivity", "Linear analysis",
             "AD elasticities of a run's peak-to-peak load to every aircraft field.",
             (), True, load_sensitivity, "scripts/sensitivity_load.py"),
    Analysis("gust-transfer", "Gust transfer function", "Gust response",
             "Fly frozen sinusoids and compare the fitted transfer with the exact one.",
             CONDITION + (POINTS, WL_MIN, WL_MAX, GUST_LAG, WING_TAIL), False,
             gust_transfer, "scripts/gust_transfer_sweep.py"),
    Analysis("pratt-walker", "Discrete gust vs Pratt & Walker", "Gust response",
             "1 - cos gusts at several gradient distances against NACA Report 1206.",
             CONDITION + (PEAK, GRADIENTS), False, pratt_walker, "scripts/discrete_gust.py"),
    Analysis("convergence", "Step-size convergence", "Verification",
             "Re-fly a run at halved steps and fit the observed order.",
             (REFINEMENTS,), True, convergence, "atisim.verification.fitted_order"),
    Analysis("cross-code", "Cross-code: JSBSim", "Verification",
             "Fly AtiSim from JSBSim's recorded state through the vortex JSBSim flew, "
             "and overlay the frozen reference.", (CASE, ARM), False, cross_code,
             "scripts/vortex_compare.py; atisim.jsbsim_vortex_ref"),
    Analysis("verification", "Verification suite", "Verification",
             "RK4 order on an oscillator and on the 6-DOF rollout, free fall through a "
             "swinging wind, torque-free rotation.",
             (), False, verification_suite, "atisim.verification"),
    Analysis("study", "Script study", "Studies",
             "Run one of the research scripts unchanged and keep what it printed and "
             "the figures and tables it wrote.",
             (SCRIPT, ARGUMENTS, MINUTES), False, study, "atisim.studies; scripts/"),
)}


# ---------------------------------------------------------------------------
# Specs, validation, running
# ---------------------------------------------------------------------------


def default(key: str, base: run.RunSpec | None = None) -> AnalysisSpec:
    a = ANALYSES[key]
    params = {p.name: p.default for p in a.params}
    name = key if base is None else f"{key}-{base.name}"
    return AnalysisSpec(name=name, analysis=key, params=params,
                        base=base if a.needs_base else None)


def apply_set(aspec: AnalysisSpec, assignment: str) -> AnalysisSpec:
    """One `key=value` from `atisim analyse --set`.

    `name=...` renames, `base.<key>=...` is `run.apply_set` on the base run, and
    any other key is one of the analysis's own parameters.
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    key, raw = (s.strip() for s in assignment.split("=", 1))
    if key == "name":
        return aspec._replace(name=raw)
    if key.startswith("base."):
        if aspec.base is None:
            raise ValueError(f"{aspec.analysis} has no base run to set {key!r} on")
        return aspec._replace(base=run.apply_set(aspec.base, key[5:] + "=" + raw))
    params = {p.name: p for p in ANALYSES[aspec.analysis].params}
    names = list(params)
    if key not in names:
        raise ValueError(f"cannot --set {key!r} on {aspec.analysis}; settable: "
                         + ", ".join(["name"] + names + (["base.<key>"] if aspec.base
                                                         is not None else [])))
    value = raw if params[key].check == "arguments" else run._parse(raw)
    return aspec._replace(params={**aspec.params, key: value})


def validate(aspec: AnalysisSpec) -> list[Issue]:
    issues: list[Issue] = []
    if aspec.analysis not in ANALYSES:
        return [Issue("error", "analysis", f"Unknown analysis {aspec.analysis!r}. "
                                           f"Choose one of: {', '.join(ANALYSES)}.")]
    a = ANALYSES[aspec.analysis]
    name = aspec.name
    if not isinstance(name, str) or not name.strip() or any(
            ch in name for ch in '/\\:*?"<>|') or name.startswith("."):
        issues.append(Issue("error", "name", "Name must be a valid directory name: "
                                             "letters, digits, - and _."))
    for p in a.params:
        value = aspec.params.get(p.name, p.default)
        if p.check == "optional":
            if value is not None and (not run._number(value) or value <= 0):
                issues.append(Issue("error", p.name, f"{p.label} must be a positive "
                                                     f"number of {p.unit}, or empty for "
                                                     "the aircraft's CRUISE value."))
        elif p.check == "text":
            try:
                if not _floats(value):
                    raise ValueError
            except ValueError:
                issues.append(Issue("error", p.name, f"{p.label} must be numbers "
                                                     "separated by commas."))
        elif p.check == "choice":
            if value not in p.choices:
                issues.append(Issue("error", p.name,
                                    f"{p.label} must be one of: {', '.join(p.choices)}."))
        elif p.check == "integer":
            if not (isinstance(value, int) and not isinstance(value, bool)) or value < 1:
                issues.append(Issue("error", p.name, f"{p.label} must be a whole number, "
                                                     "one or more."))
        elif p.check == "arguments":
            try:
                shlex.split(value if isinstance(value, str) else "")
                if not isinstance(value, str):
                    raise ValueError
            except ValueError:
                issues.append(Issue("error", p.name, f"{p.label} must be text as on a "
                                                     "command line, quotes closed."))
        elif p.check == "nonnegative":
            if not run._number(value) or value < 0:
                issues.append(Issue("error", p.name, f"{p.label} must be zero or more."))
        elif not run._number(value) or value <= 0:
            issues.append(Issue("error", p.name, f"{p.label} must be a positive number."))
    if a.needs_base:
        if aspec.base is None:
            issues.append(Issue("error", "base", "This analysis runs on a run: set one "
                                                 "up in Setup first."))
        else:
            issues.extend(Issue(i.level, "base", f"The run: {i.message}")
                          for i in run.validate(aspec.base))
            if aspec.analysis == "ensemble" and stochastic_part(aspec.base) is None:
                issues.append(Issue("error", "base", "An ensemble varies a seed, and this "
                                                     "run has none. Choose a turbulence "
                                                     "field, or add a turbulence overlay."))
            if aspec.analysis == "closed-loop" and aspec.base.wind.kind == "manoeuvre":
                issues.append(Issue("error", "base", "The manoeuvre is an elevator pulse "
                                                     "and cannot be flown by the "
                                                     "autopilot."))
    elif not a.needs_base and "aircraft" in [p.name for p in a.params]:
        key = aspec.params.get("aircraft", AIRCRAFT.default)
        if key in REGISTRY and not issues:
            _, V, H = condition(aspec)
            spec = run.BLANK._replace(aircraft=key, airspeed_mps=V, altitude_m=H)
            issues.extend(i for i in run.validate(spec) if i.field in
                          ("airspeed_mps", "altitude_m"))
    return issues


def output_name(aspec: AnalysisSpec) -> str:
    """The report directory's name before its commit: the spec's name, and for a
    study left at its default name, `study-<script>`."""
    if aspec.analysis == "study" and aspec.name == "study":
        return f"study-{aspec.params.get('script', SCRIPT.default)}"
    return aspec.name


def directory_for(root, aspec: AnalysisSpec) -> Path:
    """`{name}-{sha7}` under `root`, suffixed -2, -3 ... if taken. Never overwrites."""
    root = Path(root)
    sha = artifact.git_sha()[:7] or "nogit"
    base = f"{output_name(aspec)}-{sha}"
    candidate, n = root / base, 1
    while candidate.exists():
        n += 1
        candidate = root / f"{base}-{n}"
    return candidate


class AnalysisError(ValueError):
    """The analysis spec has errors. The message lists them."""


class Progress:
    """What an analysis reports as it runs: `progress("a stage")`, and for a
    study, `progress.log(line)` for each line the script prints."""

    def __init__(self, on_stage=None, on_log=None):
        self._stage = on_stage or (lambda message: None)
        self._log = on_log or (lambda line: None)

    def __call__(self, message: str) -> None:
        self._stage(message)

    def log(self, line: str) -> None:
        self._log(line)


def perform(aspec: AnalysisSpec, root, on_stage: Callable[[str], None] | None = None,
            on_log: Callable[[str], None] | None = None) -> Path:
    """Validate, run and write one analysis. Returns its directory.

    A failure removes the partial directory, so a list never shows half an
    analysis. (A study whose script fails is not a failure of the analysis: it
    is written, with the script's own message.)
    """
    stage = Progress(on_stage, on_log)
    problems = [i for i in validate(aspec) if i.level == "error"]
    if problems:
        raise AnalysisError("; ".join(i.message for i in problems))
    try:
        import plotly  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError:
        raise RuntimeError('An analysis writes figures and runs: it needs the `ui` '
                           'extra. Install it with: pip install -e ".[ui]"') from None
    directory = directory_for(root, aspec)
    directory.mkdir(parents=True)
    try:
        report = ANALYSES[aspec.analysis].execute(aspec, directory, stage)
        stage("writing")
        return write_report(directory, report)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
