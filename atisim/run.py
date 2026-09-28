"""One turbulence-encounter run, specified as data, flown, checked and saved.

Before this module the orchestration lived only in scripts: `scripts/vortex.py`
trimmed, placed the field, flew, ran the checks, assembled `build_meta` and
called `write_run`, each step inline and each script with its own copy. A run
is now a value -- a `RunSpec` -- and `fly(spec)` does every step once, so the
CLI, the UI and the scripts fly the same experiment the same way.

**A spec is JSON.** `RunSpec.to_json` / `RunSpec.from_json` round-trip exactly,
which is what lets the UI show the command and the spec that reproduce any run
it flew, and lets a spec file sit next to a paper draft. JSON, not TOML:
`requires-python >= 3.10` and `tomllib` arrived in 3.11.

**Provenance is part of the spec, not decoration.** Every wind parameter is
either SOURCED (it is the number in `WindSpec.source`) or DECLARED (a modelling
choice, and `WindSpec.declared` says why). Editing a sourced value makes it
declared, because it is no longer the source's number -- `with_param` does that,
so the UI cannot forget to. `provenance(spec)` is what the UI renders.

**The presets are the scripts' experiments, not new ones.** `vortex-hannibal`,
`updraft` and `manoeuvre` build exactly the metadata `scripts/vortex.py` builds,
so a preset flown here has the same `config_hash` as the script's artifact.

No Dash and no pyarrow at module level: the simulator and every script work
without the `ui` extra. Only `save` needs pyarrow, and it says so when missing.
"""

import json
import math
import time
from pathlib import Path
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import airframe, checks, fieldkinds, loads, trim, viz, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.analysis import artifact, diagnostics
from atisim.atmosphere import density, speed_of_sound
from atisim.units import FT2M, RAD2DEG

# ---------------------------------------------------------------------------
# Wind-field kinds and their parameters
# ---------------------------------------------------------------------------


class Param(NamedTuple):
    """One editable wind parameter: what the property grid shows in a row.

    `check` is the rule `validate` applies: "positive" (the default, every
    physical size), "nonnegative", "integer" (a PRNG seed, zero or more) or
    "choice" (one of `choices`, drawn as a select). `default` is the value a
    spec without the key flies, so a parameter added after a preset was written
    never makes that preset invalid, and `wind_meta` records it only when it
    differs -- which is what keeps every earlier `config_hash` unchanged.
    """

    name: str
    label: str
    unit: str
    check: str = "positive"
    choices: tuple = ()
    default: object = None


_PROFILE = Param("profile", "Core profile", "", "choice", ("rankine", "lamb-oseen"),
                 "rankine")
_SEED = Param("seed", "Realisation seed", "", "integer", (), 0)


def _turbulence(scale_default: float) -> tuple:
    return (
        Param("sigma_w", "Vertical gust rms sigma_w", "m/s"),
        Param("L_w", "Scale length L_w", "m", default=scale_default),
        _SEED,
    )


PARAMETERS: dict[str, tuple[Param, ...]] = {
    "VortexArray": (
        Param("r0", "Core radius r0", "m"),
        Param("v0", "Peak tangential velocity V0", "m/s"),
        Param("spacing", "Core spacing", "m"),
        _PROFILE,
    ),
    "SingleVortex": (
        Param("r0", "Core radius r0", "m"),
        Param("v0", "Peak tangential velocity V0", "m/s"),
        _PROFILE,
    ),
    "MehtaHannibal": (
        Param("replay", "Evaluate on the identified path", "", "choice", ("yes", "no"),
              "yes"),
        Param("form", "Vortex form", "", "choice", ("point", "line"), "point"),
        _PROFILE,
    ),
    "UpdraftColumn": (
        Param("w0", "Peak updraft w0", "m/s"),
        Param("traverse_seconds", "Traverse time", "s"),
        Param("sharpness", "Edge sharpness", "-"),
    ),
    "LeeWave": (
        Param("w0", "Vertical velocity amplitude w0", "m/s"),
        Param("wavelength", "Wavelength", "m"),
        Param("waves", "Wavelengths flown", "-"),
    ),
    "Microburst": (
        Param("u_max", "Peak outflow u_max", "m/s"),
        Param("radius", "Downdraft shaft radius R", "m"),
        Param("z_m", "Altitude of peak outflow z_m", "m"),
    ),
    "Sinusoid": (
        Param("amplitude", "Vertical gust amplitude", "m/s"),
        Param("wavelength", "Wavelength", "m"),
        Param("waves", "Wavelengths flown", "-"),
    ),
    "OneMinusCosine": (
        Param("peak", "Peak vertical gust U_de", "m/s"),
        Param("gradient_distance", "Gradient distance H", "m"),
    ),
    "Dryden": _turbulence(float(wind.DRYDEN_LW)) + (
        Param("components", "Components", "", "choice", ("vertical", "three-axis"),
              "vertical"),
    ),
    "VonKarman": _turbulence(float(wind.VON_KARMAN_LW)),
    "GaussianDryden": _turbulence(float(wind.DRYDEN_LW)),
    "manoeuvre": (
        Param("pushdown_seconds", "Elevator pulse length", "s"),
    ),
    "none": (),
}
if fieldkinds.available("ModulatedDryden"):
    # Archive research code (`2026-09-25-v1.2-plan.md`): offered only where the
    # engine carries it, and never named by the public tree.
    PARAMETERS["ModulatedDryden"] = _turbulence(float(wind.DRYDEN_LW)) + (
        Param("depth", "Envelope depth", "-", "nonnegative"),
        Param("patch_length", "Patch length", "m"),
    )

# The kinds that are a frozen realisation of a random process. They take a seed,
# have no structure to centre a view on, and are what an overlay may be.
STOCHASTIC_KINDS = tuple(k for k in fieldkinds.STOCHASTIC if k in PARAMETERS)

KIND_LABELS = {
    "VortexArray": "Vortex array",
    "SingleVortex": "Single vortex core",
    "MehtaHannibal": "Mehta's five-vortex Hannibal field",
    "UpdraftColumn": "Updraft column",
    "LeeWave": "Lee wave",
    "Microburst": "Microburst",
    "Sinusoid": "Sinusoidal vertical gust",
    "OneMinusCosine": "1 - cos discrete gust",
    "Dryden": "Dryden turbulence",
    "VonKarman": "von Karman turbulence",
    "GaussianDryden": "Dryden turbulence, Gaussian control",
    "ModulatedDryden": "Dryden turbulence in patches (archive)",
    "manoeuvre": "Elevator manoeuvre, zero wind",
    "none": "Still air",
}

# The Parks and Wingrove & Bach cases, and the Fig. 8 manoeuvre that completes
# them. `scripts/fly.py`: "Both are the 747's turbulence cases -- flying a light
# aircraft into them is not a sourced result."
PARKS_WINGROVE_KINDS = ("VortexArray", "SingleVortex", "MehtaHannibal", "UpdraftColumn",
                        "manoeuvre")

# Below this many core radii of lead-in, `scripts/vortex.py`: "the 1/r far field
# launches the aircraft out of equilibrium and contaminates the first core."
MIN_LEAD_IN = 12.0

# How far from an aircraft's CRUISE entry a run may be before `validate` warns.
# Each entry is valid only near CRUISE (PRODUCT.md). Density, not altitude,
# because density is what every aerodynamic force sees; the 747 approach entry
# is flown at a 3% density extrapolation on purpose (aircraft.CRUISE), so 5%
# does not flag that and does flag a light aircraft taken 1.2 km down.
CRUISE_DENSITY_TOLERANCE = 0.05
CRUISE_AIRSPEED_TOLERANCE = 0.10

# DECLARED limits on what `validate` lets fly. None is a physical law. Each turns
# an input that crashed, diverged or ran for hours in testing into a message
# before the run starts.
#
# The step. No preset, script or test flies a step longer than 0.05 s, so a
# longer one is unverified (a warning), and ten times that is refused. The 747's
# short period is near 1 rad/s and RK4 is unstable near |lambda| dt = 2.8, so a
# step of seconds diverges: dt = 5 s gave a run of NaN.
MAX_VERIFIED_DT = 0.05  # s
MAX_DT = 0.5  # s
# A lead-in of 100000 core radii asked for 7.75 million steps (21.5 h of flight)
# with no warning. One million is 2.8 h at dt = 0.01 s.
MAX_STEPS = 1_000_000
# The step must resolve the field: a 1 mm core at 236 m/s is crossed in 8.5
# microseconds, and the run started 4 cm from it.
MIN_STEPS_ACROSS_FIELD = 4
# `atmosphere` models the ISA's two layers, which end at 20 km (ICAO Doc 7488).
# Above that it holds the stratosphere isothermal, which the standard does not.
MAX_ALTITUDE_M = 20000.0
# The aerodynamic data of every aircraft here is subsonic.
MAX_MACH = 1.0
# The run directory is {name}-{aircraft}-{sha}, and one Windows path component
# holds 255 characters. 100 leaves room for the suffix and the runs directory.
MAX_NAME_CHARS = 100

STRIP_CAVEAT = (
    "STRIP loads: quote the loading-shape sensitivity beside any result "
    "-- 2.6% across defensible shapes, 49.7% including the uniform bracket."
)

STAGE_HOLD_CAVEAT = (
    "Wind HELD across the four RK4 stages: first order in a spatially varying "
    "field (ASSUMPTIONS E4). Use it only to reproduce a v1.1 number."
)
GUST_LAG_CAVEAT = (
    "Kussner gust lag ON (ASSUMPTIONS C12): an opt-in v1.2 correction, off in "
    "every published number."
)
WING_TAIL_CAVEAT = (
    "Wing-tail gust delay ON: the pitching gust is the secant between the CG and "
    "the tail (wind.sampled_field_model). An opt-in v1.2 correction, off in every "
    "published number."
)

# The one flight condition with a named source. Other aircraft cite their
# `aircraft.CRUISE` entry, which carries its own source in a comment there.
CONDITION_SOURCES = {"boeing747": "NASA CR-2144 flight condition 9"}

PARKS_SOURCE = "Parks, Wingrove, Bach & Mehta 1985, J. Aircraft 22(2) 124-129"
WINGROVE_SOURCE = "Wingrove & Bach 1994, J. Aircraft 31(4) 753-760"

# Pulse lead-in for the manoeuvre, s. `vortex_viz.manoeuvre` explains why a
# pulse at t = 0 overstates the excursion by about 0.08 g.
PUSHDOWN_LEAD = 2.0

# Still-air runs have no disturbance to size a run from.
STILL_AIR_SECONDS = 30.0

# A stochastic field has no extent either. 60 s is two of the phase-2 baseline's
# 30 s encounters (`scripts/phase2_common.py`), so the TPAWS reduction has two
# windows to work on. DECLARED: a run length, not a property of the turbulence.
TURBULENCE_SECONDS = 60.0

# Lead distance ahead of a 1 - cos gust, in mean chords, and the ring-down
# after it: `scripts/discrete_gust.py`'s 40 and 60 chords plus 40 s, so the peak
# load that can fall after a short gust has passed is inside the run.
GUST_LEAD_CHORDS = 40.0
GUST_RINGDOWN_CHORDS = 60.0
GUST_RINGDOWN_SECONDS = 40.0

# Mehta's array has no calm air outside it (`vortex_viz.fly_in_moving_air`), so
# it is flown from equilibrium in the moving airmass with `fly_mehta`'s 12 r0.
MEHTA_LEAD_IN = 12.0


class WindSpec(NamedTuple):
    """The wind field of a run.

    `params` holds the kind's numbers in SI (plus `case` for a Parks array).
    `declared` maps each DECLARED parameter to the reason it is declared; every
    other parameter is the number `source` gives. `preset` names the preset
    these numbers came from, or None.
    """

    kind: str
    params: dict
    preset: str | None
    source: str
    declared: dict


class RunSpec(NamedTuple):
    """Everything that defines one run. JSON round-trippable.

    `seconds` None means "derived from the field", as each script does.
    `condition_source` "" means "derived from the aircraft's CRUISE entry".
    `declared` holds run-level declared parameters (the microburst altitude, the
    vortex lead-in) with their reasons, the same way `WindSpec.declared` does.

    `overlay` is a stochastic field added on top of `wind` with
    `wind.superpose` (the "Hannibal plus Dryden" experiment), or None.
    `stage_sampled`, `gust_lag` and `wing_tail` are the solver options
    `integrate.step` and `wind.sampled_field_model` take; their defaults are the
    engine's, so a spec written before they existed flies as it did.

    `fidelity` is "standard" or "high". High adds `diagnostics.parquet` to the
    artifact (`analysis.diagnostics`): a probe pass AFTER the flight, so the
    flight, `run.parquet` and the config hash are the same at either setting.
    """

    name: str
    aircraft: str
    airspeed_mps: float
    altitude_m: float
    wind: WindSpec
    dt: float = 0.01
    lead_in: float = 40.0
    strip: bool = False
    seconds: float | None = None
    condition_source: str = ""
    caveats: tuple = ()
    declared: dict = {}
    overlay: WindSpec | None = None
    stage_sampled: bool = True
    gust_lag: bool = False
    wing_tail: bool = False
    fidelity: str = "standard"

    def to_dict(self) -> dict:
        out = self._asdict()
        out["wind"] = dict(self.wind._asdict())
        out["overlay"] = None if self.overlay is None else dict(self.overlay._asdict())
        out["caveats"] = list(self.caveats)
        return out

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, text) -> "RunSpec":
        """From a JSON string or an already-parsed dict."""
        data = json.loads(text) if isinstance(text, (str, bytes)) else dict(text)
        unknown = set(data) - set(cls._fields)
        if unknown:
            raise ValueError(f"unknown RunSpec fields: {', '.join(sorted(unknown))}")
        data["wind"] = _wind_from(data.pop("wind"))
        if data.get("overlay") is not None:
            data["overlay"] = _wind_from(data["overlay"])
        data["caveats"] = tuple(data.get("caveats", ()))
        data.setdefault("declared", {})
        return cls(**data)


def _wind_from(raw) -> WindSpec:
    wind_data = dict(raw)
    wind_data.setdefault("preset", None)
    wind_data.setdefault("declared", {})
    wind_data.setdefault("source", "")
    return WindSpec(**wind_data)


class Issue(NamedTuple):
    """One validation finding. `field` names what the UI should select."""

    level: str  # "error" | "warning"
    field: str
    message: str


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_747 = "boeing747"
_V747 = CRUISE[_747]["airspeed"]
_H747 = CRUISE[_747]["altitude"]

# `scripts/vortex.py`'s `common["caveats"]`, verbatim, on all three of its
# encounters. Kept on all three presets so a preset run is the same experiment
# (same config_hash) as the script's artifact.
_ORDERING_CAVEAT = (
    "Load comparisons are ORDERING ONLY, never values: "
    "both papers' records are DC-10 class and neither identifies an "
    "aircraft type."
)


def _core_caveat(r0: float, aircraft: str) -> str:
    """`scripts/vortex.py`'s point-gust caveat, with the ratio computed.

    The script states 3.07 spans, which is its default case on its default
    aircraft; the Morton core is smaller, so the number is computed rather than
    copied.
    """
    spans = r0 / float(REGISTRY[aircraft].b)
    return (
        f"The Parks core is {spans:.2f} spans, so the point-gust assumption E2 is "
        "marginal for the vortex case specifically."
    )


def _refresh_core_caveat(spec: RunSpec) -> RunSpec:
    """Recompute the core caveat after an edit to the core radius or the aircraft.

    The ratio is the core radius over the span, so a caveat computed for the
    preset went on saying "2.30 spans" beside a core of 1 mm."""
    prefix = "The Parks core is "
    r0 = spec.wind.params.get("r0")
    if (not any(c.startswith(prefix) for c in spec.caveats)
            or spec.aircraft not in REGISTRY or not _number(r0) or r0 <= 0):
        return spec
    new = _core_caveat(r0, spec.aircraft)
    return spec._replace(caveats=type(spec.caveats)(
        new if c.startswith(prefix) else c for c in spec.caveats))


_LEAD_IN_REASON = (
    "start distance upstream, in core radii. Below ~12 the 1/r far field "
    "launches the aircraft out of equilibrium and contaminates the first core."
)


def _vortex_preset(case: str) -> RunSpec:
    c = wind.PARKS_CASES[case]
    return RunSpec(
        name=f"vortex-{case}",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="VortexArray",
            params={"case": case, "r0": c["r0"], "v0": c["v0"], "spacing": c["spacing"]},
            preset=f"vortex-{case}",
            source=PARKS_SOURCE,
            declared={},
        ),
        caveats=(_ORDERING_CAVEAT, _core_caveat(wind.PARKS_CASES["hannibal"]["r0"], _747)
                 if case == "hannibal" else _core_caveat(c["r0"], _747)),
        declared={"lead_in": _LEAD_IN_REASON},
    )


_FIG8_CAVEATS = (_ORDERING_CAVEAT, _core_caveat(wind.PARKS_CASES["hannibal"]["r0"], _747))

PRESETS: dict[str, RunSpec] = {
    "vortex-hannibal": _vortex_preset("hannibal"),
    "vortex-morton": _vortex_preset("morton"),
    "updraft": RunSpec(
        name="updraft",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="UpdraftColumn",
            params={"w0": float(wind.UPDRAFT_W0),
                    "traverse_seconds": float(wind.UPDRAFT_SECONDS),
                    "sharpness": 6.0},
            preset="updraft",
            source=WINGROVE_SOURCE,
            declared={"sharpness": "DECLARED, not sourced -- the paper fixes the "
                                   "magnitude and duration and says nothing about the edge"},
        ),
        caveats=_FIG8_CAVEATS,
    ),
    "lee-wave": RunSpec(
        name="lee-wave",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="LeeWave",
            params={"w0": wind.LEE_WAVE_AMPLITUDE["south"],
                    "wavelength": wind.LEE_WAVE_WAVELENGTH, "waves": 3.0},
            preset="lee-wave",
            source="Doyle et al. 2011 Mon. Wea. Rev. 139 3-23, IOP 4 primary wave, "
                   "south leg",
            declared={
                "wavelength": "DECLARED MODELLING PARAMETER, not source data: Doyle "
                              "et al.'s 20-35 km band is TROPOSPHERIC and the same "
                              "paragraph says stratospheric wavelengths are shorter "
                              "without giving a number.",
                "waves": "wavelengths flown; the run length, not a property of "
                         "the wave",
            },
        ),
        caveats=(
            "The field carries no horizontal perturbation, so the shear term is "
            "zero by construction and F is the vertical term alone.",
        ),
    ),
    "microburst": RunSpec(
        name="microburst",
        aircraft="cherokee",
        airspeed_mps=CRUISE["cherokee"]["airspeed"],
        altitude_m=300.0,
        wind=WindSpec(
            kind="Microburst",
            params={"u_max": 19.03, "radius": 1000.0, "z_m": 150.0},
            preset="microburst",
            source="Oseguera & Bowles 1988, NASA TM-100632; peak outflow is the "
                   "paper's own example, 37 kt",
            declared={
                "radius": "DECLARED: the paper parameterises by it but its example "
                          "value is only in a scanned figure. 1000 m puts peak "
                          "outflow at 1.12 km radius, so a 2.2 km outflow diameter "
                          "-- inside the 1-4 km band Wilson et al. use to call an "
                          "outflow a microburst at all.",
                "z_m": "Midpoint of the paper's stated 100-200 m.",
            },
        ),
        caveats=(
            "The default aircraft is the CHEROKEE, not the 747: the 747's only "
            "derivative set is CR-2144 flight condition 9, Mach 0.8 at 40,000 ft, "
            "and a microburst is a sub-500 m phenomenon met at approach speed.",
            "No terrain, landing gear or ground effect is modelled, so a run that "
            "descends to one wingspan above the ground is cut there.",
        ),
        declared={
            "altitude_m": "penetration altitude AGL. 300 m sits inside the 225-335 m "
                          "band NASA's B-737 used for its 1991-92 microburst flight "
                          "tests.",
        },
    ),
    "manoeuvre": RunSpec(
        name="manoeuvre",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="manoeuvre",
            params={"pushdown_seconds": 6.609},
            preset="manoeuvre",
            source=f"{WINGROVE_SOURCE}, Fig. 8 load band read as an increment",
            declared={"pushdown_seconds": "DECLARED, not sourced -- the paper fixes "
                                          "the load the pilot reached, not how long "
                                          "they held it"},
        ),
        caveats=_FIG8_CAVEATS,
    ),
}

MEHTA_SOURCE = ("Mehta 1987, J. Guidance, Control & Dynamics 10(1) 27-31, "
                "five-vortex solution p. 30")
_MEHTA_ALTITUDE_REASON = ("Mehta p. 29: straight and level at 37,000 ft, the altitude "
                          "the field was identified at")
_MEHTA_LEAD_REASON = ("start distance upstream of the first core, in core radii: "
                      "`vortex_viz.fly_mehta`'s 12. The array has no calm air "
                      "outside it, so the run starts in equilibrium in the moving "
                      "airmass instead of leading in further.")


def _mehta_preset(name: str, overlay: WindSpec | None = None, caveats=()) -> RunSpec:
    return RunSpec(
        name=name,
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=float(wind.MEHTA_HANNIBAL_ALTITUDE),
        wind=WindSpec(
            kind="MehtaHannibal",
            params={"replay": "yes", "form": "point", "profile": "rankine"},
            preset="mehta-hannibal",
            source=MEHTA_SOURCE,
            declared={},
        ),
        lead_in=MEHTA_LEAD_IN,
        overlay=overlay,
        caveats=(
            "The headline form: the field is evaluated on the level path it was "
            "identified along (wind.on_identified_path), because the DC-10's record "
            "(Parks et al. 1985 Fig. 6) shows it held its altitude through cores 3 "
            "and 4.",
        ) + tuple(caveats),
        declared={"altitude_m": _MEHTA_ALTITUDE_REASON, "lead_in": _MEHTA_LEAD_REASON},
    )


def _wingrove_preset(case: str) -> RunSpec:
    c = wind.WINGROVE_FIG4_CASES[case]
    return RunSpec(
        name=f"wingrove-{case}",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=float(wind.WINGROVE_CASE_ALTITUDE[case]),
        wind=WindSpec(
            kind="SingleVortex",
            params={"case": case, "r0": c["r0"], "v0": c["v0"], "profile": "rankine"},
            preset=f"wingrove-{case}",
            source=f"{WINGROVE_SOURCE}, Fig. 4",
            declared={},
        ),
        caveats=(
            "Fig. 4 gives core size and strength and no array spacing, so one core "
            "is flown, on the flight path, as the JSBSim cross-code comparison flies "
            "it.",
            _core_caveat(c["r0"], _747),
        ),
        declared={"altitude_m": "Wingrove & Bach Table 1, p. 754: the altitude the "
                                "incident was flown at",
                  "lead_in": _LEAD_IN_REASON},
    )


_737 = "boeing737"
_TURBULENCE_CAVEAT = (
    "One frozen realisation. A single run's peaks are one draw: an ensemble over "
    "seeds is what a statistic needs."
)
_SIGMA_REASON = ("DECLARED intensity: one of the phase-2 baseline's sigma_w "
                 "(scripts/phase2_baseline_records.py flies 2, 3, 4.46 and 6 m/s)")
_SEED_REASON = "a realisation, not a property of the turbulence"
_TURBULENCE_DT_REASON = "the phase-2 baseline's 0.02 s (scripts/phase2_common.py)"


def _turbulence_preset(name: str, kind: str, source: str) -> RunSpec:
    return RunSpec(
        name=name,
        aircraft=_737,
        airspeed_mps=CRUISE[_737]["airspeed"],
        altitude_m=CRUISE[_737]["altitude"],
        wind=WindSpec(
            kind=kind,
            params={"sigma_w": 3.0, "seed": 0},
            preset=name,
            source=source,
            declared={"sigma_w": _SIGMA_REASON, "seed": _SEED_REASON},
        ),
        dt=0.02,
        caveats=(_TURBULENCE_CAVEAT,),
        declared={"dt": _TURBULENCE_DT_REASON},
    )


def _mehta_overlay() -> WindSpec:
    return WindSpec(
        kind="Dryden",
        params={"sigma_w": round(float(wind.mehta_unmodelled_wind()), 6), "seed": 0},
        preset=None,
        source="MIL-F-8785C Dryden vertical spectrum",
        declared={
            "sigma_w": ("LOWER BOUND on the fluctuation Mehta's fit leaves out "
                        "(wind.mehta_unmodelled_wind). The ceiling is "
                        f"{float(wind.mehta_residual_ceiling()):.2f} m/s "
                        "(wind.mehta_residual_ceiling)."),
            "seed": _SEED_REASON,
        },
    )


PRESETS.update({
    "mehta-hannibal": _mehta_preset("mehta-hannibal"),
    "mehta-hannibal-turbulent": _mehta_preset(
        "mehta-hannibal-turbulent", overlay=_mehta_overlay(),
        caveats=("Dryden turbulence is added on top at the lower bound of what "
                 "Mehta's fit leaves out. Closing the Hannibal load gap needs "
                 "sigma_w of about 4-5 m/s.",)),
    "wingrove-hannibal": _wingrove_preset("hannibal"),
    "wingrove-morton": _wingrove_preset("morton"),
    "wingrove-cimarron": _wingrove_preset("cimarron"),
    "lee-wave-north": PRESETS["lee-wave"]._replace(
        name="lee-wave-north",
        wind=PRESETS["lee-wave"].wind._replace(
            params={**PRESETS["lee-wave"].wind.params,
                    "w0": wind.LEE_WAVE_AMPLITUDE["north"]},
            preset="lee-wave-north",
            source="Doyle et al. 2011 Mon. Wea. Rev. 139 3-23, IOP 4 primary wave, "
                   "north leg",
        ),
    ),
    "sinusoid": RunSpec(
        name="sinusoid",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="Sinusoid",
            params={"amplitude": 0.5, "wavelength": 1200.0, "waves": 12.0},
            preset="sinusoid",
            source="a test input with an exact answer (gust.gust_transfer)",
            declared={
                "amplitude": "DECLARED: gust.measure_gust_transfer's 0.5 m/s, small "
                             "enough to stay linear",
                "wavelength": "DECLARED: one point of the sweep "
                              "scripts/gust_transfer_sweep.py runs",
                "waves": "DECLARED: gust.measure_gust_transfer's 12 periods",
            },
        ),
    ),
    "discrete-gust": RunSpec(
        name="discrete-gust",
        aircraft=_747,
        airspeed_mps=_V747,
        altitude_m=_H747,
        wind=WindSpec(
            kind="OneMinusCosine",
            params={"peak": 1.0,
                    "gradient_distance": 12.5 * float(REGISTRY[_747].c)},
            preset="discrete-gust",
            source="Pratt & Walker, NACA Report 1206: H = 12.5 mean chords, the "
                   "gradient distance K_g was fitted at",
            declared={"peak": "DECLARED: scripts/discrete_gust.py's 1 m/s, small "
                              "enough that neither nonlinearity nor a control "
                              "limit enters"},
        ),
    ),
    "dryden": _turbulence_preset(
        "dryden", "Dryden", "MIL-F-8785C Dryden vertical spectrum, L_w 1750 ft "
                            "(wind.DRYDEN_LW)"),
    "von-karman": _turbulence_preset(
        "von-karman", "VonKarman", "von Karman vertical spectrum, L_w 2500 ft "
                                   "(wind.VON_KARMAN_LW)"),
    "gaussian-dryden": _turbulence_preset(
        "gaussian-dryden", "GaussianDryden",
        "Dryden spectrum with random amplitudes: the Gaussian control "
        "(wind.gaussian_vertical_field)"),
})

BLANK = RunSpec(
    name="run",
    aircraft=_747,
    airspeed_mps=_V747,
    altitude_m=_H747,
    wind=WindSpec(kind="none", params={}, preset=None, source="no wind field",
                  declared={}),
    seconds=STILL_AIR_SECONDS,
)


def preset(name: str) -> RunSpec:
    """A preset by name, or `BLANK` for "blank". Raises KeyError naming the choices."""
    if name == "blank":
        return BLANK
    try:
        return PRESETS[name]
    except KeyError:
        raise KeyError(
            f"unknown preset {name!r}; choose from {', '.join(PRESETS)} or blank"
        ) from None


def wind_preset(name: str) -> WindSpec:
    """The wind field alone, for the Setup page's Preset select."""
    return preset(name).wind


# ---------------------------------------------------------------------------
# Editing and provenance
# ---------------------------------------------------------------------------


class Provenance(NamedTuple):
    status: str  # "sourced" | "declared"
    note: str


def _param(kind: str, name: str) -> Param | None:
    return next((p for p in PARAMETERS.get(kind, ()) if p.name == name), None)


def param_value(wind_spec: WindSpec, param: Param):
    """A parameter's value in `wind_spec`, or its default when the key is absent."""
    return wind_spec.params.get(param.name, param.default)


def _source_value(spec: RunSpec, name: str):
    """The preset's value for a wind parameter, or None if it has no source."""
    if spec.wind.preset is None or spec.wind.preset not in PRESETS:
        return None
    base = PRESETS[spec.wind.preset].wind
    if base.kind != spec.wind.kind or name in base.declared:
        return None
    param = _param(base.kind, name)
    return base.params.get(name, None if param is None else param.default)


def _show(value, unit: str = "") -> str:
    text = f"{value:g}" if isinstance(value, (int, float)) else str(value)
    return f"{text} {unit}".strip()


def with_param(spec: RunSpec, name: str, value) -> RunSpec:
    """Set one wind parameter, keeping provenance truthful.

    A sourced value that is edited becomes DECLARED, because it is no longer the
    source's number. Editing it back to the source's number makes it sourced
    again. A parameter that was already declared keeps its reason.
    """
    params = dict(spec.wind.params, **{name: value})
    declared = dict(spec.wind.declared)
    original = _source_value(spec, name)
    if original is not None:
        if value == original:
            declared.pop(name, None)
        else:
            param = _param(spec.wind.kind, name)
            unit = param.unit if param is not None else ""
            declared[name] = (
                f"edited: not the source's number ({_show(original, unit)} in "
                f"{spec.wind.source})"
            )
    elif name not in declared and spec.wind.preset is None:
        declared[name] = "set by the user; no source"
    spec = spec._replace(wind=spec.wind._replace(params=params, declared=declared))
    return _refresh_core_caveat(spec) if name == "r0" else spec


def reset_param(spec: RunSpec, name: str) -> RunSpec:
    """Put a wind parameter back to its preset's value and provenance."""
    if spec.wind.preset not in PRESETS:
        return spec
    base = PRESETS[spec.wind.preset].wind
    param = _param(base.kind, name)
    value = base.params.get(name, None if param is None else param.default)
    params = dict(spec.wind.params, **{name: value})
    declared = dict(spec.wind.declared)
    declared.pop(name, None)
    if name in base.declared:
        declared[name] = base.declared[name]
    spec = spec._replace(wind=spec.wind._replace(params=params, declared=declared))
    return _refresh_core_caveat(spec) if name == "r0" else spec


# ---------------------------------------------------------------------------
# The turbulence overlay
# ---------------------------------------------------------------------------

OVERLAY_SOURCES = {
    "Dryden": "MIL-F-8785C Dryden vertical spectrum",
    "VonKarman": "von Karman vertical spectrum",
    "GaussianDryden": "Dryden spectrum with random amplitudes (the Gaussian control)",
    "ModulatedDryden": "Dryden spectrum in patches (archive research field)",
}


def with_overlay_kind(spec: RunSpec, kind: str | None) -> RunSpec:
    """Add, change or remove the turbulence overlay. None removes it.

    A new overlay starts at the phase-2 baseline's 3 m/s and seed 0, both
    DECLARED; the scale length starts at the spectrum's own, which is the
    specification's number.
    """
    if kind in (None, "none"):
        return spec._replace(overlay=None)
    if kind not in STOCHASTIC_KINDS:
        raise ValueError(f"an overlay must be one of {', '.join(STOCHASTIC_KINDS)}, "
                         f"not {kind!r}")
    if spec.overlay is not None and spec.overlay.kind == kind:
        return spec
    return spec._replace(overlay=WindSpec(
        kind=kind, params={"sigma_w": 3.0, "seed": 0}, preset=None,
        source=OVERLAY_SOURCES[kind],
        declared={"sigma_w": "set by the user; no source", "seed": _SEED_REASON},
    ))


def with_overlay_param(spec: RunSpec, name: str, value) -> RunSpec:
    """Set one overlay parameter. Every edit is DECLARED: an overlay has no preset."""
    if spec.overlay is None:
        raise ValueError("this run has no turbulence overlay to edit")
    declared = dict(spec.overlay.declared)
    param = _param(spec.overlay.kind, name)
    if param is not None and value == param.default and name == "L_w":
        declared.pop(name, None)
    else:
        declared.setdefault(name, "set by the user; no source")
    return spec._replace(overlay=spec.overlay._replace(
        params={**spec.overlay.params, name: value}, declared=declared))


def _cruise(aircraft: str) -> dict | None:
    return CRUISE.get(aircraft)


def provenance(spec: RunSpec) -> dict[str, Provenance]:
    """Sourced or declared, and why, for every editable number in the spec.

    What the property grid renders beside each row. Computed here, not in the
    UI, so it is tested.
    """
    out: dict[str, Provenance] = {}
    for p in PARAMETERS.get(spec.wind.kind, ()):
        if p.name in spec.wind.declared:
            out[p.name] = Provenance("declared", spec.wind.declared[p.name])
        elif spec.wind.preset is not None:
            out[p.name] = Provenance("sourced", spec.wind.source)
        else:
            out[p.name] = Provenance("declared", "set by the user; no source")
    if spec.overlay is not None:
        for p in PARAMETERS.get(spec.overlay.kind, ()):
            key = f"overlay.{p.name}"
            if p.name in spec.overlay.declared:
                out[key] = Provenance("declared", spec.overlay.declared[p.name])
            else:
                out[key] = Provenance("sourced", spec.overlay.source)

    cruise = _cruise(spec.aircraft)
    cruise_note = CONDITION_SOURCES.get(
        spec.aircraft, f"the aircraft's CRUISE entry (atisim.aircraft.CRUISE['{spec.aircraft}'])"
    )
    for field, key in (("airspeed_mps", "airspeed"), ("altitude_m", "altitude")):
        if field in spec.declared:
            out[field] = Provenance("declared", spec.declared[field])
        elif cruise is not None and getattr(spec, field) == cruise[key]:
            out[field] = Provenance("sourced", cruise_note)
        else:
            out[field] = Provenance(
                "declared",
                "not the aircraft's CRUISE value; its derivatives are valid only "
                "near CRUISE",
            )
    out["lead_in"] = Provenance("declared", spec.declared.get("lead_in", _LEAD_IN_REASON))
    out["dt"] = Provenance("declared", "numerical setting, not source data")
    return out


def condition_source(spec: RunSpec) -> str:
    """Where the flight condition comes from, as the artifact records it."""
    if spec.condition_source:
        return spec.condition_source
    prov = provenance(spec)
    speed, height = prov["airspeed_mps"], prov["altitude_m"]
    if speed.status == height.status == "sourced":
        return speed.note
    return f"airspeed: {speed.note}; altitude: {height.note}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate(spec: RunSpec) -> list[Issue]:
    """Errors stop a run; warnings travel with it. Rules are the codebase's own.

    Errors are values the simulator cannot fly. Warnings are the four cautions
    the scripts already print: a light aircraft in a 747 case, a short vortex
    lead-in, a flight condition far from CRUISE, and strip loads.
    """
    issues: list[Issue] = []

    def error(field, message):
        issues.append(Issue("error", field, message))

    def warning(field, message):
        issues.append(Issue("warning", field, message))

    name = spec.name
    if not isinstance(name, str) or not name.strip():
        error("name", "Run name is empty. Type a name, for example vortex-hannibal.")
    elif any(ch in name for ch in '/\\:*?"<>|') or name.startswith("."):
        error("name", f"Run name {name!r} is not a valid directory name. "
                      "Do not use / \\ : * ? \" < > | or a first dot.")
    elif len(name) > MAX_NAME_CHARS:
        error("name", f"Run name has {len(name)} characters. Use {MAX_NAME_CHARS} "
                      "or fewer: the name becomes a directory name.")

    if spec.aircraft not in REGISTRY:
        error("aircraft", f"Unknown aircraft {spec.aircraft!r}. "
                          f"Choose one of: {', '.join(sorted(REGISTRY))}.")
    if not _number(spec.airspeed_mps) or spec.airspeed_mps <= 0:
        error("airspeed_mps", "Airspeed must be a positive number of m/s.")
    if not _number(spec.altitude_m) or spec.altitude_m < 0:
        error("altitude_m", "Altitude must be zero or a positive number of m.")
    elif spec.altitude_m > MAX_ALTITUDE_M:
        error("altitude_m", f"Altitude {spec.altitude_m:g} m is above {MAX_ALTITUDE_M:,.0f} m, "
                            "the top of the standard atmosphere that AtiSim models. "
                            "Set a lower altitude.")
    elif _number(spec.airspeed_mps) and spec.airspeed_mps > 0:
        sound = float(speed_of_sound(jnp.array(spec.altitude_m)))
        if spec.airspeed_mps / sound >= MAX_MACH:
            error("airspeed_mps", f"Airspeed {spec.airspeed_mps:g} m/s is Mach "
                                  f"{spec.airspeed_mps / sound:.2f} at this altitude. "
                                  "The aerodynamic data of each aircraft is subsonic. "
                                  f"Set less than {sound:.0f} m/s.")
    if not _number(spec.dt) or spec.dt <= 0:
        error("dt", "Time step must be a positive number of s.")
    elif spec.dt > MAX_DT:
        error("dt", f"Time step {spec.dt:g} s is longer than {MAX_DT:g} s, where RK4 "
                    f"diverges for these aircraft. Set {MAX_VERIFIED_DT:g} s or less.")
    elif spec.dt > MAX_VERIFIED_DT:
        warning("dt", f"Time step {spec.dt:g} s is longer than {MAX_VERIFIED_DT:g} s, the "
                      "longest step that a preset, script or test flies. Run the "
                      "step-size convergence analysis to check it.")
    if spec.seconds is not None and (not _number(spec.seconds) or spec.seconds <= 0):
        error("seconds", "Duration must be a positive number of s, or empty to "
                         "derive it from the field.")

    kind = spec.wind.kind
    if kind not in PARAMETERS:
        error("wind", f"Unknown wind field kind {kind!r}. "
                      f"Choose one of: {', '.join(PARAMETERS)}.")
        return issues
    _check_params(spec.wind, error, prefix="")

    if kind in ("VortexArray", "SingleVortex", "MehtaHannibal"):
        if not _number(spec.lead_in) or spec.lead_in <= 0:
            error("lead_in", "Lead-in must be a positive number of core radii.")
        elif kind != "MehtaHannibal" and spec.lead_in < MIN_LEAD_IN:
            warning("lead_in", f"Lead-in is {spec.lead_in:g} core radii. Below "
                               f"{MIN_LEAD_IN:g} the 1/r far field launches the "
                               "aircraft out of equilibrium and contaminates the "
                               "first core. Set 12 or more.")
    if (kind == "MehtaHannibal" and spec.wind.params.get("form") == "line"
            and spec.wind.params.get("profile", "rankine") != "rankine"):
        error("wind", "The line form is Rankine only: wind.line_vortex_wind has no "
                      "Lamb-Oseen sibling. Set the core profile to rankine.")
    if kind == "manoeuvre" and spec.strip:
        error("strip", "Strip loads do not apply to the manoeuvre: it flies at "
                       "zero wind. Turn strip loads off.")
    if kind == "Microburst" and spec.aircraft in REGISTRY and _number(spec.altitude_m):
        span = float(REGISTRY[spec.aircraft].b)
        if spec.altitude_m <= span:
            error("altitude_m", f"Altitude {spec.altitude_m:g} m is within one "
                                f"wingspan ({span:.1f} m) of the ground, where the "
                                "run is cut. Set a higher altitude.")

    if spec.overlay is not None:
        if spec.overlay.kind not in STOCHASTIC_KINDS:
            error("overlay", f"An overlay must be turbulence ({', '.join(STOCHASTIC_KINDS)}), "
                             f"not {spec.overlay.kind!r}.")
        else:
            _check_params(spec.overlay, error, prefix="overlay.")
        if kind == "manoeuvre":
            error("overlay", "The manoeuvre is defined by the absence of turbulence. "
                             "Remove the overlay.")
        elif kind in STOCHASTIC_KINDS:
            warning("overlay", "Turbulence on top of turbulence: the two spectra add. "
                               "Remove the overlay unless that is the experiment.")

    for flag in ("stage_sampled", "gust_lag", "wing_tail", "strip"):
        if not isinstance(getattr(spec, flag), bool):
            error(flag, f"{flag} must be true or false.")
    if spec.fidelity not in FIDELITIES:
        error("fidelity", f"Fidelity must be one of: {', '.join(FIDELITIES)}.")
    if spec.gust_lag is True and not spec.stage_sampled:
        error("gust_lag", "The gust lag needs stage-sampled wind: the filter reads the "
                          "field at every RK4 stage. Turn stage sampling on.")
    if spec.aircraft in REGISTRY:
        ac = REGISTRY[spec.aircraft]
        if (spec.gust_lag is True and _number(spec.dt) and spec.dt > 0
                and _number(spec.airspeed_mps) and spec.airspeed_mps > 0):
            dt_max = float(wind.kussner_max_dt(jnp.array(spec.airspeed_mps), ac.c))
            if spec.dt > dt_max:
                error("dt", f"Time step {spec.dt:g} s is too long for the gust lag's "
                            f"fast pole. Set {dt_max:.4f} s or less "
                            "(wind.KUSSNER_RK4_LIMIT).")
        if spec.wing_tail is True and tail_arm(spec.aircraft) is None \
                and not bool(airframe.tail_arm_is_plausible(ac)):
            error("wing_tail", f"{spec.aircraft} has no usable tail arm: its source "
                               "defines no CL_q and no JSBSim arm is held for it. "
                               "Turn the wing-tail delay off.")
    if spec.stage_sampled is False:
        warning("stage_sampled", STAGE_HOLD_CAVEAT)

    if spec.aircraft in REGISTRY:
        if kind in PARKS_WINGROVE_KINDS and not spec.aircraft.startswith("boeing747"):
            warning("aircraft", f"{spec.aircraft} in a Parks or Wingrove & Bach case "
                                "is not a sourced result: both are the 747's "
                                "turbulence cases.")
        cruise = _cruise(spec.aircraft)
        if (cruise is not None and _number(spec.airspeed_mps) and spec.airspeed_mps > 0
                and _number(spec.altitude_m) and spec.altitude_m >= 0):
            rho = float(density(jnp.array(spec.altitude_m)))
            rho_cruise = float(density(jnp.array(cruise["altitude"])))
            d_rho = abs(rho - rho_cruise) / rho_cruise
            d_v = abs(spec.airspeed_mps - cruise["airspeed"]) / cruise["airspeed"]
            if d_rho > CRUISE_DENSITY_TOLERANCE or d_v > CRUISE_AIRSPEED_TOLERANCE:
                warning("altitude_m" if d_rho > CRUISE_DENSITY_TOLERANCE else "airspeed_mps",
                        f"{spec.aircraft}'s derivatives are valid only near its "
                        f"CRUISE condition ({cruise['airspeed']:.1f} m/s, "
                        f"{cruise['altitude']:.0f} m). This run differs by "
                        f"{d_rho * 100:.0f}% in density and {d_v * 100:.0f}% in "
                        "airspeed.")
    if spec.strip and kind != "manoeuvre":
        warning("strip", STRIP_CAVEAT)
    # These read the whole spec, so they run only when nothing above is wrong.
    if not any(i.level == "error" for i in issues):
        _check_extent(spec, error, warning)
    return issues


# The time the aircraft takes to cross each field's structure, from its spec,
# with the parameter that sets it. None: a field with no one size to resolve.
def _crossing(kind: str, p: dict, V: float) -> tuple[float, str, str, str] | None:
    if kind in ("VortexArray", "SingleVortex"):
        return 2.0 * p["r0"] / V, "a core", "r0", "the core radius"
    if kind == "UpdraftColumn":
        return p["traverse_seconds"], "the column", "traverse_seconds", "the traverse time"
    if kind in ("LeeWave", "Sinusoid"):
        return p["wavelength"] / V, "one wavelength", "wavelength", "the wavelength"
    if kind == "Microburst":
        return 2.0 * p["radius"] / V, "the outflow", "radius", "the radius"
    if kind == "OneMinusCosine":
        return (2.0 * p["gradient_distance"] / V, "the gust", "gradient_distance",
                "the gradient distance")
    return None


# The peak vertical wind of each deterministic field, for the gust angle.
_PEAK_UP = {"VortexArray": "v0", "SingleVortex": "v0", "UpdraftColumn": "w0",
            "LeeWave": "w0", "Sinusoid": "amplitude", "OneMinusCosine": "peak"}


def _check_extent(spec: RunSpec, error, warning) -> None:
    """The run's length in steps, and the field against the step and the speed."""
    kind = spec.wind.kind
    values = {q.name: param_value(spec.wind, q) for q in PARAMETERS[kind]}
    V, dt = float(spec.airspeed_mps), float(spec.dt)
    seconds = geometry(spec).seconds
    steps = int(round(seconds / dt))  # as `vortex_viz` counts them
    long_field = ("seconds" if spec.seconds is not None
                  else "lead_in" if kind in ("VortexArray", "SingleVortex", "MehtaHannibal")
                  else "dt")
    if steps < 2:
        error("seconds" if spec.seconds is not None else "dt",
              f"The run is {seconds:.3g} s long, less than two time steps of {dt:g} s. "
              "Make the duration longer or the time step shorter.")
    elif steps > MAX_STEPS:
        error(long_field, f"The run needs {steps:,} steps ({seconds:,.0f} s at {dt:g} s). "
                          f"The limit is {MAX_STEPS:,}. Make the run shorter or the "
                          "time step longer.")

    crossing = _crossing(kind, values, V)
    if crossing is not None:
        t_cross, what, size, noun = crossing
        if t_cross < MIN_STEPS_ACROSS_FIELD * dt:
            error(size, f"The aircraft crosses {what} in {t_cross:.3g} s, less than "
                        f"{MIN_STEPS_ACROSS_FIELD} time steps of {dt:g} s, so the run cannot "
                        f"resolve it. Make {noun} larger or the time step shorter.")

    if kind == "VortexArray":
        r0, spacing = values["r0"], values["spacing"]
        if spacing < 2.0 * r0:
            warning("spacing", f"Core spacing {spacing:g} m is less than two core radii "
                               f"({2.0 * r0:g} m), so the cores overlap. Parks' two pairs "
                               "are 5.8 and 7.1 core radii apart.")
    peak = _PEAK_UP.get(kind)
    if peak is not None:
        up = float(values[peak])
        angle = math.degrees(math.atan2(up, V))
        if angle > checks.ALPHA_LINEAR_DEG:
            warning(peak, f"A peak vertical wind of {up:g} m/s at {V:g} m/s is a gust angle "
                          f"of attack of {angle:.1f} deg. Past {checks.ALPHA_LINEAR_DEG:g} deg "
                          "the linear aerodynamics do not hold.")


def _check_params(wind_spec: WindSpec, error, prefix: str) -> None:
    """Apply each parameter's `check` rule. `prefix` names the overlay's fields."""
    for p in PARAMETERS[wind_spec.kind]:
        value = param_value(wind_spec, p)
        field = f"{prefix}{p.name}" if prefix else p.name
        if p.check == "choice":
            if value not in p.choices:
                error(field, f"{p.label} must be one of: {', '.join(p.choices)}.")
        elif p.check == "integer":
            if not (isinstance(value, int) and not isinstance(value, bool)) or value < 0:
                error(field, f"{p.label} must be a whole number, zero or more.")
        elif p.check == "nonnegative":
            if not _number(value) or value < 0:
                error(field, f"{p.label} must be zero or a positive number of {p.unit}.")
        elif not _number(value) or value <= 0:
            error(field, f"{p.label} must be a positive number of {p.unit}.")


def errors(spec: RunSpec) -> list[Issue]:
    return [i for i in validate(spec) if i.level == "error"]


# ---------------------------------------------------------------------------
# Geometry: where the run starts, how long it lasts, what the window is
# ---------------------------------------------------------------------------


class Geometry(NamedTuple):
    """The planned straight path, before anything is flown. Also the preview's input."""

    start_north: float  # m
    seconds: float  # s
    end_north: float  # m, start + V * seconds, the straight-line plan
    altitude: float  # m
    window: tuple[float, float] | None  # north, m; None for a time window
    window_name: str
    window_meta: dict  # as `declared_parameters["window"]` records it
    window_rule: str
    structure_north: float | None  # where the field's structure is centred


_OWN_EXTENT = "the disturbance's own extent"


def geometry(spec: RunSpec) -> Geometry:
    """The start point, duration and window each script uses, per field kind."""
    V, H = float(spec.airspeed_mps), float(spec.altitude_m)
    p, kind = spec.wind.params, spec.wind.kind
    rule = _OWN_EXTENT
    if kind == "VortexArray":
        r0, spacing = p["r0"], p["spacing"]
        lead = spec.lead_in * r0
        start, seconds = -lead, (spacing + lead + 6.0 * r0) / V
        window, name = (-r0, r0), "first core"
        meta = {"kind": name, "north_m": [-r0, r0]}
        centre = 0.5 * spacing
    elif kind == "UpdraftColumn":
        radius = 0.5 * p["traverse_seconds"] * V
        start, seconds = -2.0 * radius, 4.0 * radius / V
        window, name = (-radius, radius), "column"
        meta = {"kind": name, "north_m": [-radius, radius]}
        centre = 0.0
    elif kind == "LeeWave":
        # Open on a zero crossing, a quarter wavelength upstream of the trough at
        # north = 0, as `scripts/leewave.py` does: a periodic field has no
        # undisturbed region to lead in through.
        wavelength = p["wavelength"]
        start, seconds = -0.25 * wavelength, p["waves"] * wavelength / V
        end = start + V * seconds
        window, name = (start, end), "wave train"
        meta = {"kind": name, "north_m": [start, end]}
        rule = (f"{_OWN_EXTENT}: a periodic field has no edge, so the window is "
                "the whole train flown")
        centre = 0.0
    elif kind == "Microburst":
        peak = wind.MICROBURST_PEAK_RADIUS_RATIO * p["radius"]
        start, seconds = -3.0 * peak, 6.0 * peak / V
        window, name = (-peak, peak), "outflow ring"
        meta = {"kind": name, "north_m": [-peak, peak]}
        centre = 0.0
    elif kind == "SingleVortex":
        r0 = p["r0"]
        lead = spec.lead_in * r0
        start, seconds = -lead, (lead + 6.0 * r0) / V
        window, name = (-r0, r0), "core"
        meta = {"kind": name, "north_m": [-r0, r0]}
        centre = 0.0
    elif kind == "MehtaHannibal":
        array = wind.mehta_hannibal_array(H)
        r0 = float(array.r0)
        x0, x1 = float(array.north.min()), float(array.north.max())
        start = x0 - spec.lead_in * r0
        seconds = (x1 + spec.lead_in * r0 - start) / V
        window, name = (x0 - 2.0 * r0, x1 + 2.0 * r0), \
            "the identified array, plus 2 r0 either side"
        meta = {"kind": name, "north_m": list(window)}
        centre = 0.5 * (x0 + x1)
    elif kind == "Sinusoid":
        # Open on a zero crossing, as the lee wave does: cos is at its peak at
        # north = 0, so a quarter wavelength upstream is w = 0.
        wavelength = p["wavelength"]
        start, seconds = -0.25 * wavelength, p["waves"] * wavelength / V
        end = start + V * seconds
        window, name = (start, end), "wave train"
        meta = {"kind": name, "north_m": [start, end]}
        rule = (f"{_OWN_EXTENT}: a periodic field has no edge, so the window is "
                "the whole train flown")
        centre = 0.0
    elif kind == "OneMinusCosine":
        chord = float(REGISTRY[spec.aircraft].c) if spec.aircraft in REGISTRY else 0.0
        lead, gust = GUST_LEAD_CHORDS * chord, p["gradient_distance"]
        start = -lead
        seconds = (lead + 2.0 * gust + GUST_RINGDOWN_CHORDS * chord) / V \
            + GUST_RINGDOWN_SECONDS
        end = start + V * seconds
        window, name = (start, end), "whole run"
        meta = {"kind": name, "north_m": [start, end]}
        rule = ("the whole run: the peak load can fall after a short gust has "
                "passed (scripts/discrete_gust.py)")
        centre = gust
    elif kind in STOCHASTIC_KINDS:
        start, seconds = 0.0, TURBULENCE_SECONDS
        if spec.seconds is not None:
            seconds = float(spec.seconds)
        end = start + V * seconds
        window, name = (start, end), "whole run"
        meta = {"kind": name, "north_m": [start, end]}
        rule = "a frozen random field has no edge, so the window is the whole run"
        centre = None
    elif kind == "manoeuvre":
        hold = p["pushdown_seconds"]
        start, seconds = 0.0, PUSHDOWN_LEAD + 3.0 * hold
        window, name = None, "elevator pulse"
        meta = {"kind": name, "seconds": hold, "starts_at_s": PUSHDOWN_LEAD}
        centre = None
    else:  # still air
        start, seconds = 0.0, STILL_AIR_SECONDS
        window, name = None, "whole run"
        meta = None
        rule = "still air has no disturbance, so the window is the whole run"
        centre = None
    if spec.seconds is not None:
        seconds = float(spec.seconds)
    if meta is None:
        meta = {"kind": name, "seconds": seconds, "starts_at_s": 0.0}
    return Geometry(start, seconds, start + V * seconds, H, window, name, meta,
                    rule, centre)


# ---------------------------------------------------------------------------
# The field
# ---------------------------------------------------------------------------


def _zero_field(pos_ned):
    return jnp.zeros(3)


def tail_arm(aircraft: str) -> float | None:
    """The wing-tail arm in metres where a source gives it, else None (derived).

    JSBSim's <htailarm> for the two entries whose source defines no CL_q
    (`airframe.JSBSIM_HTAILARM_FT`); every other aircraft's arm is derived from
    CL_q and Cm_q by `airframe.stations`, behind its plausibility gate.
    """
    feet = airframe.JSBSIM_HTAILARM_FT.get(aircraft)
    return None if feet is None else feet * FT2M


def build_field(spec: RunSpec) -> Callable:
    """The field as a callable of NED position, as the scripts build it.

    The four original kinds keep the scripts' own construction. Every other
    kind, and the overlay, is built by `fieldkinds` from `wind_meta(spec)` --
    the very parameters the artifact stores -- so the rebuilt field is the
    flown one by construction.
    """
    p, kind, H = spec.wind.params, spec.wind.kind, float(spec.altitude_m)
    if kind == "VortexArray" and p.get("profile", "rankine") == "rankine":
        # Two cores on the flightpath, as Parks describes (scripts/vortex.py).
        cores = [(0.0, H), (p["spacing"], H)]
        array = wind.VortexArray(
            north=jnp.array([c[0] for c in cores]),
            down=jnp.array([-c[1] for c in cores]),
            r0=jnp.array(p["r0"]),
            v0=jnp.array(p["v0"]),
        )
        base = lambda pos: wind.vortex_wind(pos, array)  # noqa: E731
    elif kind == "UpdraftColumn":
        radius = 0.5 * p["traverse_seconds"] * float(spec.airspeed_mps)
        column = wind.UpdraftColumn(
            north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(p["w0"]),
            radius=jnp.array(radius), sharpness=jnp.array(p["sharpness"]),
        )
        base = lambda pos: wind.updraft_wind(pos, column)  # noqa: E731
    elif kind == "LeeWave":
        wave = wind.LeeWave(w0=jnp.array(p["w0"]), wavelength=jnp.array(p["wavelength"]),
                            north=jnp.array(0.0))
        base = lambda pos: wind.lee_wave_wind(pos, wave)  # noqa: E731
    elif kind == "Microburst":
        burst = wind.microburst(u_max=p["u_max"], radius=p["radius"], z_m=p["z_m"])
        base = lambda pos: wind.microburst_wind(pos, burst)  # noqa: E731
    elif kind in ("VortexArray", "SingleVortex", "MehtaHannibal"):
        base = fieldkinds.vortex_array(wind_meta(spec)["params"])
    elif kind in fieldkinds.KINDS:
        base = fieldkinds.build(kind, wind_meta(spec)["params"])
    else:
        base = _zero_field
    if spec.overlay is None:
        return base
    return fieldkinds.with_overlay(base, {"overlay": _overlay_meta(spec.overlay)})


def _overlay_meta(overlay: WindSpec) -> dict:
    return {"kind": overlay.kind, "source": overlay.source,
            "params": _stochastic_params(overlay)}


def _stochastic_params(w: WindSpec) -> dict:
    """Every parameter, defaults filled in, so the artifact names the field whole."""
    return {p.name: param_value(w, p) for p in PARAMETERS[w.kind]}


def wind_meta(spec: RunSpec) -> dict:
    """`meta["wind_field"]`, exactly as `scripts/vortex.py` records it.

    `artifact.rebuild_field` is its inverse; `test_run.py` asserts the round
    trip for every preset. A parameter at its default is not written (a
    Rankine core, a point vortex), so the four original kinds' blocks -- and
    their `config_hash` -- are what they were before the options existed.
    """
    out = _base_wind_meta(spec)
    if spec.wing_tail:
        out["omega_gust_estimator"] = "fitted across the airframe: the wing-tail secant"
        out["model"] = "wind.sampled_field_model"
    if spec.overlay is not None:
        out["overlay"] = _overlay_meta(spec.overlay)
    return out


def _base_wind_meta(spec: RunSpec) -> dict:
    p, kind, H = spec.wind.params, spec.wind.kind, float(spec.altitude_m)
    common = {"model": "wind.field_model",
              "omega_gust_estimator": "analytic tangent at CG"}
    profile = {} if p.get("profile", "rankine") == "rankine" else {"profile": p["profile"]}
    if kind == "VortexArray":
        cores = [(0.0, H), (p["spacing"], H)]
        out = {"kind": "VortexArray"}
        if p.get("case"):
            out["case"] = p["case"]
        out.update(
            source=spec.wind.source,
            params={"north": [c[0] for c in cores], "down": [-c[1] for c in cores],
                    "r0": p["r0"], "v0": p["v0"], "spacing": p["spacing"], **profile},
            **common,
        )
        return out
    if kind == "SingleVortex":
        out = {"kind": "VortexArray"}
        if p.get("case"):
            out["case"] = p["case"]
        out.update(source=spec.wind.source,
                   params={"north": [0.0], "down": [-H], "r0": p["r0"], "v0": p["v0"],
                           **profile},
                   **common)
        return out
    if kind == "MehtaHannibal":
        array = wind.mehta_hannibal_array(H)
        params = {"north": [float(x) for x in array.north],
                  "down": [float(x) for x in array.down],
                  "r0": float(array.r0), "v0": float(array.v0),
                  "cos_dpsi": float(array.cos_dpsi), "sin_dpsi": float(array.sin_dpsi),
                  **profile}
        if p.get("form", "point") == "line":
            params["form"] = "line"
        if p.get("replay", "yes") == "yes":
            params["path_altitude"] = H
        return {"kind": "VortexArray", "case": "mehta", "source": spec.wind.source,
                "params": params, **common}
    if kind == "UpdraftColumn":
        radius = 0.5 * p["traverse_seconds"] * float(spec.airspeed_mps)
        return {"kind": "UpdraftColumn", "source": spec.wind.source,
                "params": {"north": 0.0, "east": 0.0, "w0": float(p["w0"]),
                           "radius": float(radius), "sharpness": p["sharpness"]},
                **common}
    if kind == "LeeWave":
        return {"kind": "LeeWave", "source": spec.wind.source,
                "params": {"w0": p["w0"], "wavelength": p["wavelength"], "north": 0.0},
                **common}
    if kind == "Microburst":
        return {"kind": "Microburst", "source": spec.wind.source,
                "params": {"u_max": p["u_max"], "radius": p["radius"], "z_m": p["z_m"],
                           "north": 0.0, "east": 0.0},
                **common}
    if kind == "Sinusoid":
        return {"kind": "Sinusoid", "source": spec.wind.source,
                "params": {"amplitude": p["amplitude"], "wavelength": p["wavelength"],
                           "phase": 0.0, "north": 0.0},
                **common}
    if kind == "OneMinusCosine":
        return {"kind": "OneMinusCosine", "source": spec.wind.source,
                "params": {"peak": p["peak"], "gradient_distance": p["gradient_distance"],
                           "start_north": 0.0, "north": p["gradient_distance"]},
                **common}
    if kind in STOCHASTIC_KINDS:
        return {"kind": kind, "source": spec.wind.source,
                "params": _stochastic_params(spec.wind), **common}
    if kind == "manoeuvre":
        return {"kind": "none (zero wind)",
                "source": "the category is DEFINED by the absence of turbulence",
                "params": {}}
    return {"kind": "none (still air)", "source": spec.wind.source, "params": {}}


# `scripts/vortex.py` names this one differently; kept, so its hash is unchanged.
_PROVENANCE_KEYS = {"pushdown_seconds": "pushdown_provenance"}


def declared_meta(spec: RunSpec, elevator_step: float | None = None) -> dict:
    """`meta["declared_parameters"]`: every declared number with its reason."""
    geo = geometry(spec)
    kind = spec.wind.kind
    out: dict = {}
    if kind in ("VortexArray", "SingleVortex", "MehtaHannibal"):
        out["lead_in_core_radii"] = spec.lead_in
    for name, reason in spec.wind.declared.items():
        if name in spec.wind.params:
            out[name] = spec.wind.params[name]
            out[_PROVENANCE_KEYS.get(name, f"{name}_provenance")] = reason
    if spec.overlay is not None:
        for name, reason in spec.overlay.declared.items():
            if name in spec.overlay.params:
                out[f"overlay_{name}"] = spec.overlay.params[name]
                out[f"overlay_{name}_provenance"] = reason
    if kind == "manoeuvre" and elevator_step is not None:
        out["elevator_deg_from_trim"] = float(elevator_step * RAD2DEG)
        out["elevator_provenance"] = (
            "BISECTED to reach the Fig. 8 band read as an INCREMENT, so the load "
            "is sourced and the angle is an output"
        )
    if spec.seconds is not None:
        out["seconds"] = float(spec.seconds)
        out["seconds_provenance"] = "run length set by the user, not derived from the field"
    for name in ("airspeed_mps", "altitude_m", "dt"):
        if name in spec.declared:
            out[name] = getattr(spec, name)
            out[f"{name}_provenance"] = spec.declared[name]
    out["window"] = geo.window_meta
    out["window_rule"] = geo.window_rule
    return out


# ---------------------------------------------------------------------------
# Flying
# ---------------------------------------------------------------------------


class SpecError(ValueError):
    """The spec has errors. The message lists them."""


class TrimError(ValueError):
    """The trim has no solution that is a flight condition. The message says why."""


class Flown(NamedTuple):
    """A flown and checked run, ready for `save`.

    `diagnostics` is `analysis.diagnostics.Diagnostics` on a High run and None
    on a Standard one. `timing` is the wall time of each stage, in seconds.
    """

    spec: RunSpec
    encounter: vortex_viz.Encounter
    trajectory: viz.Trajectory
    report: list
    meta: dict
    diagnostics: object = None
    timing: dict = {}


FIDELITIES = ("standard", "high")

STAGES = ("validating", "trimming", "flying", "checks", "diagnostics", "writing", "done")


def _trim(spec: RunSpec):
    """Trim, and refuse a solution that is not a flight condition.

    `trim.trim` cannot raise (it is jitted), so both of its failure modes are
    asked here: non-convergence (the residual) and nonsense (`is_physical`).
    """
    ac = REGISTRY[spec.aircraft]
    V, H = float(spec.airspeed_mps), float(spec.altitude_m)
    x, res = trim.trim(jnp.array(V), jnp.array(H), ac)
    residual = float(jnp.linalg.norm(res))
    where = f"{spec.aircraft} at {V:.1f} m/s, {H:.0f} m"
    alpha, elevator, throttle = (float(v) for v in x)
    if not math.isfinite(residual) or residual > 1e-9:
        raise TrimError(
            f"Trim did not converge for {where} (residual {residual:.1e}). "
            f"{_speed_hint(spec)}"
        )
    if not trim.is_physical(x, ac):
        raise TrimError(
            f"Trim for {where} is not a flight condition: alpha "
            f"{alpha * RAD2DEG:.1f} deg, elevator {elevator * RAD2DEG:.1f} deg, "
            f"throttle {throttle:.2f} (limits: |alpha| 15 deg, |elevator| "
            f"{float(ac.elevator_limit) * RAD2DEG:.0f} deg, throttle 0 to 1). "
            f"{_speed_hint(spec)}"
        )
    return x, residual


# m/s, the top of the minimum-drag search in `_speed_hint`. A sweep that ends on
# its last point has not found V_md: the true one is higher. At 100 km the hint
# once quoted "400.0 m/s at this altitude" as if it were the answer.
_V_MD_TOP = 400.0


def _speed_hint(spec: RunSpec) -> str:
    """Name the fix: the airspeed against V_md, and the CRUISE entry."""
    ac = REGISTRY[spec.aircraft]
    v_md = float(trim.minimum_drag_speed(ac, jnp.array(float(spec.altitude_m)),
                                         high=_V_MD_TOP))
    cruise = _cruise(spec.aircraft)
    back = (f" Its CRUISE condition is {cruise['airspeed']:.1f} m/s at "
            f"{cruise['altitude']:.0f} m." if cruise else "")
    if v_md >= _V_MD_TOP - 0.1:  # the sweep's last point
        return (f"At {spec.altitude_m:.0f} m the air is too thin for this aircraft: its "
                f"minimum-drag speed is above {_V_MD_TOP:.0f} m/s. Set a lower "
                f"altitude.{back}")
    if spec.airspeed_mps < v_md:
        return (f"The airspeed is below this aircraft's minimum-drag speed "
                f"({v_md:.1f} m/s at this altitude).{back}")
    return f"Move the flight condition closer to the aircraft's CRUISE entry.{back}"


def _cut(enc: vortex_viz.Encounter, n: int) -> vortex_viz.Encounter:
    """The first `n` samples of an encounter and its log."""
    total = len(enc.t)
    fields = {}
    for name, value in enc._asdict().items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (total,):
            fields[name] = value[:n]
    log = enc.log._replace(**{
        name: np.asarray(getattr(enc.log, name))[:n] for name in viz.Trajectory._fields
    })
    return enc._replace(log=log, **fields)


def fly(spec: RunSpec, on_stage: Callable[[str], None] | None = None) -> Flown:
    """Trim, place the field, fly, run the checks and build the metadata.

    `on_stage(name)` is called as each stage starts ("trimming", "flying",
    "checks"), which is what the UI's progress dock shows. Raises `SpecError`
    for a spec with errors and `TrimError` for a flight condition that does not
    trim; both messages name the fix.
    """
    problems = errors(spec)
    if problems:
        raise SpecError("; ".join(i.message for i in problems))

    timing: dict[str, float] = {}
    clock = {"name": None, "start": 0.0}

    def stage(name: str | None) -> None:
        """Start stage `name`, closing the one before; None only closes it."""
        now = time.perf_counter()
        if clock["name"] is not None:
            timing[clock["name"]] = now - clock["start"]
        clock["name"], clock["start"] = name, now
        if name is not None and on_stage is not None:
            on_stage(name)

    stage("trimming")
    x, residual = _trim(spec)
    ac = REGISTRY[spec.aircraft]
    V, H = float(spec.airspeed_mps), float(spec.altitude_m)
    geo = geometry(spec)
    field = build_field(spec)

    stage("flying")
    elevator_step = None
    if spec.wind.kind == "manoeuvre":
        hold = spec.wind.params["pushdown_seconds"]
        try:
            elevator_step = vortex_viz.elevator_for_load(
                ac, V, H, target=vortex_viz.FIG8_LOAD_INCREMENT, hold=hold,
                seconds=geo.seconds, lead_in=PUSHDOWN_LEAD, dt=spec.dt,
            )
        except ValueError as exc:
            raise TrimError(
                f"The elevator pulse cannot reach the Fig. 8 load increment "
                f"({vortex_viz.FIG8_LOAD_INCREMENT:+.1f} g) for {spec.aircraft}: "
                f"{exc}"
            ) from None
        encounter = vortex_viz.manoeuvre(
            ac, V, H, label="manoeuvre", elevator_step=elevator_step, hold=hold,
            seconds=geo.seconds, lead_in=PUSHDOWN_LEAD, dt=spec.dt,
        )
    elif spec.wind.kind == "MehtaHannibal":
        # From equilibrium in the moving airmass, as `vortex_viz.fly_mehta`
        # flies it: the five-core array has no calm air outside it.
        encounter = vortex_viz.fly_in_moving_air(
            ac, field, V, H, label=f"{spec.aircraft} through Mehta 1987",
            start_north=geo.start_north, seconds=geo.seconds, dt=spec.dt,
            window=geo.window, window_name=geo.window_name,
            load_model=loads.strip_model(field, ac) if spec.strip else None,
            stage_sampled=spec.stage_sampled, gust_lag=spec.gust_lag,
            wind_model=wind_model(spec, field),
        )
    else:
        label = (f"vortex ({spec.wind.params['case']})"
                 if spec.wind.kind == "VortexArray" and spec.wind.params.get("case")
                 else spec.name)
        encounter = vortex_viz.fly(
            ac, field, V, H, label=label, start_north=geo.start_north,
            seconds=geo.seconds, dt=spec.dt,
            window=geo.window if geo.window is not None else (-math.inf, math.inf),
            window_name=geo.window_name, strip=spec.strip,
            stage_sampled=spec.stage_sampled, gust_lag=spec.gust_lag,
            wind_model=wind_model(spec, field),
        )

    extra_caveats = []
    if spec.wind.kind == "Microburst":
        # The ground arrives. Cut at one wingspan, as scripts/microburst.py does:
        # below that the integration is arithmetic, not physics.
        altitude = -np.asarray(encounter.log.pos_ned)[:, 2]
        below = np.flatnonzero(altitude <= float(ac.b))
        if below.size:
            encounter = _cut(encounter, int(below[0]))
            extra_caveats.append(
                f"Run cut at t = {encounter.t[-1]:.2f} s, one wingspan above the "
                "ground."
            )

    stage("checks")
    flown = record(spec, encounter, x, residual_norm=residual,
                   elevator_step=elevator_step, field=field,
                   extra_caveats=extra_caveats)
    if spec.fidelity == "high":
        stage("diagnostics")
        columns, channels = diagnostics.probe(
            encounter.log, ac, field,
            model=(wind.zero_wind if spec.wind.kind == "manoeuvre"
                   else wind_model(spec, field)),
            load_model=loads.strip_model(field, ac) if spec.strip else None,
            lag=encounter.gust_lag,
        )
        stage(None)
        return flown._replace(diagnostics=diagnostics.Diagnostics(columns, channels, timing),
                              timing=timing)
    stage(None)
    return flown._replace(timing=timing)


def wind_model(spec: RunSpec, field):
    """The `wind_model` a run flies: None (the default `field_model`) unless the
    wing-tail delay is on, when the gust rates are fitted across two stations,
    the CG and the tail (`airframe.stations(n_lon=2)`)."""
    if not spec.wing_tail:
        return None
    ac = REGISTRY[spec.aircraft]
    return wind.sampled_field_model(
        field, airframe.stations(ac, n_lon=2, tail_arm=tail_arm(spec.aircraft)))


def solver_meta(spec: RunSpec) -> dict:
    """The solver options that differ from the engine's defaults, for
    `meta["integrator"]`. Empty for a default run, so its hash is unchanged."""
    out: dict = {}
    if not spec.stage_sampled:
        out["stage_sampled"] = False
    if spec.gust_lag:
        out["gust_lag"] = "kussner_jones"
    if spec.wing_tail:
        arm = tail_arm(spec.aircraft)
        if arm is None:
            ac = REGISTRY[spec.aircraft]
            arm, how = float(airframe.effective_tail_arm(ac) * ac.c), "derived from CL_q, Cm_q"
        else:
            how = "JSBSim <htailarm>"
        out["wing_tail"] = {"tail_arm_m": float(arm), "source": how}
    return out


def solver_caveats(spec: RunSpec) -> list[str]:
    return ([STAGE_HOLD_CAVEAT] if not spec.stage_sampled else []) \
        + ([GUST_LAG_CAVEAT] if spec.gust_lag else []) \
        + ([WING_TAIL_CAVEAT] if spec.wing_tail else [])


def record(spec: RunSpec, encounter: vortex_viz.Encounter, x, *,
           residual_norm: float | None = None, elevator_step: float | None = None,
           field=None, extra_caveats=()) -> Flown:
    """Run the checks on a flown encounter and build its metadata.

    Split from `fly` so a script that has already flown (scripts/vortex.py
    flies for its figure) writes the same artifact without flying twice.
    """
    ac = REGISTRY[spec.aircraft]
    field = build_field(spec) if field is None else field
    alpha, elevator, throttle = (float(v) for v in x)
    report = checks.run_checks(
        encounter.log, ac, trim.trimmed_controls(x[1], x[2]), field, encounter.window
    )
    caveats = (list(spec.caveats) + ([STRIP_CAVEAT] if spec.strip else [])
               + solver_caveats(spec) + list(extra_caveats))
    meta = artifact.build_meta(
        aircraft_key=spec.aircraft,
        aircraft=ac,
        flight_condition={"airspeed_mps": float(spec.airspeed_mps),
                          "altitude_m": float(spec.altitude_m),
                          "source": condition_source(spec)},
        trim_solution={"alpha_rad": alpha, "elevator_rad": elevator,
                       "throttle": throttle, "residual_norm": residual_norm,
                       "is_physical": True},
        integrator={"dt_s": spec.dt, "n_steps": len(encounter.t), **solver_meta(spec)},
        wind_field=wind_meta(spec),
        declared_parameters=declared_meta(spec, elevator_step),
        caveats=caveats,
        load_model=("loads.strip_model" if spec.strip else None),
        loading_shape=("elliptic" if spec.strip else None),
    )
    return Flown(spec, encounter, encounter.log, report, meta)


def run_directory(root, spec: RunSpec) -> Path:
    """`{name}-{aircraft}-{sha7}` under `root`, suffixed -2, -3 ... if taken.

    The naming is `scripts/vortex.py`'s. A run is never overwritten: a second
    flight of the same spec is a second artifact.
    """
    root = Path(root)
    sha = artifact.git_sha()[:7] or "nogit"
    base = f"{spec.name}-{spec.aircraft}-{sha}"
    candidate, n = root / base, 1
    while candidate.exists():
        n += 1
        candidate = root / f"{base}-{n}"
    return candidate


def save(flown: Flown, root) -> Path:
    """Write the artifact. Needs the `ui` extra (pyarrow)."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        raise RuntimeError(
            'Writing a run artifact needs pyarrow, from the `ui` extra. '
            'Install it with: pip install -e ".[ui]"'
        ) from None
    return write(flown, run_directory(root, flown.spec))


SPEC_FILE = "spec.json"


def write(flown: Flown, directory) -> Path:
    """Write `flown` into `directory`: the run, the spec that flew it
    (`spec.json`, so the app can fly it again with a change), and on a High run
    its diagnostics. `spec.json` and `diagnostics.parquet` are optional files: a
    run written without them reads exactly as before (plan decision D5)."""
    directory = artifact.write_run(directory, flown.trajectory, flown.meta, flown.report)
    (Path(directory) / SPEC_FILE).write_text(flown.spec.to_json())
    if flown.diagnostics is not None:
        diagnostics.write(directory, flown.diagnostics.columns,
                          flown.diagnostics.channels, flown.diagnostics.timing)
    return directory


def spec_of(directory) -> RunSpec | None:
    """The spec a run was flown from, or None for a run written without one."""
    path = Path(directory) / SPEC_FILE
    try:
        return RunSpec.from_json(path.read_text()) if path.exists() else None
    except (ValueError, KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Reproducing a run from code
# ---------------------------------------------------------------------------


_SETTABLE = ("name", "aircraft", "airspeed_mps", "altitude_m", "dt", "lead_in", "strip",
             "seconds", "stage_sampled", "gust_lag", "wing_tail", "fidelity")


def command(spec: RunSpec, out: str = "runs") -> str:
    """The `atisim run` command that flies this spec, when a preset plus --set can say it."""
    base = PRESETS.get(spec.wind.preset or "", None)
    if base is None or base.wind.kind != spec.wind.kind:
        return f"atisim run {spec.name}.json --out {out}"
    sets = []
    for field in _SETTABLE:
        value = getattr(spec, field)
        if value != getattr(base, field):
            sets.append(f"{field}={value}")
    for name, value in spec.wind.params.items():
        if base.wind.params.get(name) != value:
            sets.append(f"wind.{name}={value}")
    if spec.overlay != base.overlay:
        if spec.overlay is None:
            sets.append("overlay=none")
        else:
            sets.append(f"overlay={spec.overlay.kind}")
            sets.extend(f"overlay.{name}={value}"
                        for name, value in spec.overlay.params.items())
    tail = "".join(f" --set {s}" for s in sets)
    return f"atisim run --preset {spec.wind.preset} --out {out}{tail}"


def snippet(spec: RunSpec, out: str = "runs") -> str:
    """Five lines of Python that fly and save this spec."""
    return (
        "from atisim import run\n"
        f"spec = run.RunSpec.from_json(open({spec.name + '.json'!r}).read())\n"
        "flown = run.fly(spec)\n"
        f"path = run.save(flown, {out!r})\n"
        "print(path, [c.name for c in flown.report if c.passed is False])\n"
    )


def apply_set(spec: RunSpec, assignment: str) -> RunSpec:
    """Apply one `key=value` from the CLI's --set.

    `wind.<param>` edits a wind parameter, `overlay=<kind|none>` adds or
    removes turbulence on top, and `overlay.<param>` edits the overlay.
    """
    if "=" not in assignment:
        raise ValueError(f"--set expects key=value, got {assignment!r}")
    key, raw = (s.strip() for s in assignment.split("=", 1))
    if key.startswith("wind."):
        name = key[5:]
        return with_param(spec, name, _parse(raw))
    if key == "overlay":
        return with_overlay_kind(spec, None if raw.lower() in ("none", "") else raw)
    if key.startswith("overlay."):
        return with_overlay_param(spec, key[8:], _parse(raw))
    if key not in _SETTABLE:
        raise ValueError(f"cannot --set {key!r}; settable: {', '.join(_SETTABLE)}, "
                         "wind.<param>, overlay, overlay.<param>")
    return spec._replace(**{key: _parse(raw)})


def _parse(raw: str):
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null", ""):
        return None
    try:
        return int(raw) if raw.lstrip("+-").isdigit() else float(raw)
    except ValueError:
        return raw


def preview_args(spec: RunSpec) -> dict | None:
    """Keyword arguments for `figures.field_preview`, or None for no field.

    The scale sets the view and the peak pins the colour, per kind, as the
    analysis view's `Loaded` does: r0 and V0 for a vortex, radius and w0 for a
    column, a quarter wavelength and w0 for a lee wave or a sinusoid, R and
    u_max for a microburst, H and U_de for a discrete gust, and L_w and
    3 sigma_w for turbulence, which has no structure to centre on and is drawn
    around the middle of the run. An overlay is in the drawn field; its
    3 sigma_w widens the colour scale so it is not clipped.
    """
    geo = geometry(spec)
    p, kind, H = spec.wind.params, spec.wind.kind, float(spec.altitude_m)
    cores: list = []
    centre = geo.structure_north
    if kind == "VortexArray":
        scale, peak = p["r0"], p["v0"]
        cores = [(0.0, H), (p["spacing"], H)]
    elif kind == "SingleVortex":
        scale, peak = p["r0"], p["v0"]
        cores = [(0.0, H)]
    elif kind == "MehtaHannibal":
        meta = _base_wind_meta(spec)["params"]
        scale, peak = meta["r0"], meta["v0"]
        cores = [(n, -d) for n, d in zip(meta["north"], meta["down"])]
    elif kind == "UpdraftColumn":
        scale, peak = 0.5 * p["traverse_seconds"] * float(spec.airspeed_mps), p["w0"]
    elif kind in ("LeeWave",):
        scale, peak = 0.25 * p["wavelength"], p["w0"]
    elif kind == "Sinusoid":
        scale, peak = 0.25 * p["wavelength"], p["amplitude"]
    elif kind == "OneMinusCosine":
        scale, peak = p["gradient_distance"], p["peak"]
    elif kind == "Microburst":
        scale, peak = p["radius"], p["u_max"]
    elif kind in STOCHASTIC_KINDS:
        scale = param_value(spec.wind, _param(kind, "L_w"))
        peak = 3.0 * p["sigma_w"]
        centre = 0.5 * (geo.start_north + geo.end_north)
    else:
        return None
    if spec.overlay is not None:
        peak = float(peak) + 3.0 * float(spec.overlay.params.get("sigma_w", 0.0))
    return dict(
        start_north=geo.start_north, end_north=geo.end_north, altitude=H,
        structure_north=centre, scale=float(scale), peak=float(peak),
        window=geo.window, window_name=geo.window_name,
        cores=cores, ground=kind == "Microburst", label=KIND_LABELS[kind],
    )


_CONDITION_KEYS = {"airspeed_mps": "airspeed", "altitude_m": "altitude"}


def with_field(spec: RunSpec, field: str, value) -> RunSpec:
    """Set a run-level field (name, aircraft, flight condition, solver settings).

    A run-level declared reason (the microburst's 300 m) describes the preset's
    number; once the value is no longer that number, the reason no longer
    applies and is dropped, so provenance falls back to the CRUISE comparison.
    """
    spec = spec._replace(**{field: value})
    base = PRESETS.get(spec.wind.preset or "")
    if field in spec.declared and (base is None or getattr(base, field) != value):
        declared = dict(spec.declared)
        declared.pop(field)
        spec = spec._replace(declared=declared)
    return _refresh_core_caveat(spec) if field == "aircraft" else spec


def reset_field(spec: RunSpec, field: str) -> RunSpec:
    """Put a flight-condition field back to its source: the preset's declared
    number when the preset declares one for this aircraft, else CRUISE."""
    base = PRESETS.get(spec.wind.preset or "")
    if base is not None and field in base.declared and base.aircraft == spec.aircraft:
        return spec._replace(**{field: getattr(base, field)},
                             declared={**spec.declared, field: base.declared[field]})
    cruise = _cruise(spec.aircraft)
    if cruise is None or field not in _CONDITION_KEYS:
        return spec
    return with_field(spec, field, cruise[_CONDITION_KEYS[field]])
