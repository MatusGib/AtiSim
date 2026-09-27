"""The Lab: the middle tier between the test card and engineering mode.

The test card flies four fixed test points live; engineering mode shows every
value a run has. The Lab flies ANY preset with a few of its values changed,
runs four chosen analyses with their defaults, and reads a flown result back
in plain words. This module is the Lab's content -- which values a case
offers, which analyses, what a result says -- so those choices are tested
here and the app only lays them out (`atisim.apps.lab`). No Dash.

Provenance stays `run`'s: a changed value goes through `run.with_field` or
`run.with_param`, so a sourced number that is edited becomes declared exactly
as it does in engineering mode.
"""

from typing import NamedTuple

import numpy as np

from atisim import checks, run
from atisim.aircraft import CRUISE
from atisim.analysis.runs import verdict

# The field values a case offers, strength first and then size. Everything else
# stays at the preset's value; engineering mode edits it.
KEY_WIND: dict[str, tuple] = {
    "VortexArray": ("v0", "r0"),
    "SingleVortex": ("v0", "r0"),
    "UpdraftColumn": ("w0", "traverse_seconds"),
    "LeeWave": ("w0", "wavelength"),
    "Microburst": ("u_max", "radius"),
    "Sinusoid": ("amplitude", "wavelength"),
    "OneMinusCosine": ("peak", "gradient_distance"),
    "Dryden": ("sigma_w", "seed"),
    "VonKarman": ("sigma_w", "seed"),
    "GaussianDryden": ("sigma_w", "seed"),
    "ModulatedDryden": ("sigma_w", "seed"),
    "manoeuvre": ("pushdown_seconds",),
}
AIRCRAFT = tuple(sorted(CRUISE))  # every aircraft with a cruise condition to start at


def key_params(spec: run.RunSpec) -> tuple:
    """The `run.Param`s of the field values this case offers."""
    by_name = {p.name: p for p in run.PARAMETERS.get(spec.wind.kind, ())}
    return tuple(by_name[n] for n in KEY_WIND.get(spec.wind.kind, ()) if n in by_name)


def edit(base: run.RunSpec, aircraft: str | None = None, airspeed=None, altitude=None,
         wind_values: dict | None = None) -> run.RunSpec:
    """`base` with the Lab's values applied. A value equal to the base's is not
    an edit, so an untouched form flies the preset with its provenance intact."""
    spec = base
    for field, value in (("aircraft", aircraft), ("airspeed_mps", airspeed),
                         ("altitude_m", altitude)):
        if value is not None and value != getattr(spec, field):
            spec = run.with_field(spec, field, value)
    for p in key_params(spec):
        value = (wind_values or {}).get(p.name)
        if value is not None and value != run.param_value(spec.wind, p):
            spec = run.with_param(spec, p.name, value)
    return spec


class Analysis(NamedTuple):
    key: str  # an `atisim.analyses` key
    learn: str  # what a reader learns from it, in one plain sentence


ANALYSES: tuple = (
    Analysis("ensemble", "How much one turbulence case's loads change from one random "
                         "realisation to the next: the spread over seeds, and the TPAWS "
                         "peak factors."),
    Analysis("modes", "The aircraft's natural motions, its longitudinal and lateral "
                      "modes, set against the published values where the record holds "
                      "them."),
    Analysis("convergence", "Whether a run's answer depends on its time step: the run "
                            "flown again at halved steps, and the order of accuracy that "
                            "shows."),
    Analysis("cross-code", "AtiSim against JSBSim, another flight simulator: AtiSim flown "
                           "from JSBSim's recorded state through the vortex JSBSim flew, "
                           "over JSBSim's frozen answer."),
)
LEARN = {a.key: a.learn for a in ANALYSES}


class Summary(NamedTuple):
    """A flown result in plain words: the numbers a results card leads with."""

    peak: float  # g, the largest load factor
    t_peak: float  # s
    low: float  # g, the smallest
    t_low: float  # s
    sigma: float  # g, the RMS normal load (Misaka 2008), NaN when not computed
    severity: str  # `checks.severity_band`'s word
    verdict: str  # `runs.verdict`'s word for the checks


def summarise(t, n_z, run_checks: list[dict]) -> Summary:
    t, n_z = np.asarray(t, dtype=float), np.asarray(n_z, dtype=float)
    hi, lo = int(np.argmax(n_z)), int(np.argmin(n_z))
    rms = next((c for c in run_checks if c.get("name") == "rms normal load"), None)
    sigma = float(rms["value"]) if rms is not None and rms.get("value") is not None \
        else float("nan")
    return Summary(float(n_z[hi]), float(t[hi]), float(n_z[lo]), float(t[lo]), sigma,
                   checks.severity_band(sigma), verdict(run_checks))
