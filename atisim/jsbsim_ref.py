"""Reader for the frozen JSBSim reference data.

Deliberately does NOT import jsbsim, and nothing that imports this module may.
The reference is generated once by scripts/gen_jsbsim_reference.py and checked
in, so the test suite runs on a machine with no JSBSim installed -- which is the
case for this project's own venv -- and reference drift shows up in git diff
rather than as a mysterious change in test results.

Both the tests and the report read through this one parser, so a schema change
breaks in a single place instead of two.

Design: docs/superpowers/specs/2026-08-20-jsbsim-737-verification-design.md
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np
from scipy.optimize import brentq

from atisim.atmosphere import density

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_737_reference.xml"


class Condition(NamedTuple):
    """The flight condition, and the altitude atisim must be run at.

    `matched_altitude` is the geometric altitude at which atisim's density
    equals the density JSBSim actually flew at. It is **recomputed at load
    time** by `_match_density` rather than read from the reference file, and the
    reason is session 23.

    *** IT USED TO BE FROZEN, AND FREEZING IT WAS THE BUG. *** It is not a
    JSBSim measurement -- it is a property of ATISIM's atmosphere, solved
    against a JSBSim measurement. Freezing it therefore froze a dependency on a
    model this project owns and can change. When session 23 corrected the ISA to
    convert geometric to geopotential, the stored -43.22 ft stopped being a
    correction and became a 0.158% density ERROR, in the same place the original
    0.159% bias had been and in the same direction.

    Recomputing it makes the quantity track the model it is derived from. What
    stays frozen is `density`, which IS a JSBSim measurement.

    After the session-23 fix the match is nearly the identity -- 30,000 ft
    nominal against 29,999.9 ft matched -- because atisim's ISA and JSBSim's now
    agree to 4.8e-6 relative. The mechanism is kept rather than deleted because
    that residual is real: the two codes still differ in their ISA constants,
    and a comparison that assumes they do not would be asserting something
    nobody measured.
    """

    name: str
    altitude: float           # m, what JSBSim was run at
    matched_altitude: float   # m, what atisim must be run at
    density: float            # kg/m^3
    sound_speed: float        # m/s
    airspeed: float           # m/s true
    mass: float               # kg
    inertia: np.ndarray       # (3,3) kg m^2
    density_match_residual: float


class Trim(NamedTuple):
    mode: str
    alpha: float      # rad
    elevator: float   # rad
    throttle: float
    thrust: float     # N, total
    bank: float       # rad


class SweepPoint(NamedTuple):
    # vel_body is the air-relative body velocity as JSBSim reported it.
    # Driving atisim with the VECTOR rather than with (V, alpha, beta)
    # keeps the comparison independent of how either engine defines alpha
    # and beta -- they agree to nine decimals, but this needs no such check.
    vel_body: np.ndarray      # (3,) m/s
    sound_speed: float        # m/s, JSBSim's at this point
    alpha: float
    alphadot: float          # rad/s, as the engine reported it
    ci2vel: float             # c/2V, the pitch non-dimensionalisation
    beta: float
    rates: np.ndarray         # (3,) rad/s, body, air-relative
    controls: np.ndarray      # (3,) rad: elevator, aileron, rudder
    coefficients: np.ndarray  # (6,) CL, CD, CY, Cl, Cm, Cn


class StallPoint(NamedTuple):
    """One point of the full CL(alpha) curve, past the break.

    CL here has the elevator term removed, so it is the CLalpha table alone --
    which is the thing atisim's CL_table_alpha/CL_table_CL must reproduce.
    """

    alpha: float
    CL: float


class Linearization(NamedTuple):
    """JSBSim's 12-state model, ordering re-derived by the generator.

    [vt, alpha, theta, q, beta, phi, p, psi, r, lat, lon, h], inputs
    [throttle, aileron, elevator, rudder]. Longitudinal block is [0:4],
    lateral [4:9].
    """

    A: np.ndarray   # (12,12)
    B: np.ndarray   # (12,4)
    x0: np.ndarray
    u0: np.ndarray

    @property
    def longitudinal(self):
        return self.A[:4, :4]

    @property
    def lateral(self):
        return self.A[4:9, 4:9]


class TrajectorySample(NamedTuple):
    t: float
    vel_body: np.ndarray   # (3,) m/s
    omega: np.ndarray      # (3,) rad/s
    euler: np.ndarray      # (3,) rad
    altitude: float        # m
    controls: np.ndarray   # (3,) rad
    thrust: float          # N


class Reference(NamedTuple):
    jsbsim_version: str
    condition: dict
    trim: dict
    derivatives: dict
    absent: dict
    entry: dict
    sweep: list
    stall_sweep: list
    linearization: Linearization
    trajectory: dict
    diagnostics: dict


def _floats(text):
    return np.array([float(v) for v in text.split()])


def _match_density(rho: float, nominal_m: float) -> tuple[float, float]:
    """Geometric altitude at which atisim's density equals JSBSim's `rho`.

    Solved here rather than read from the file -- see `Condition`. Identical in
    form to `scripts/gen_jsbsim_vortex_reference.matched_altitude`, and it must
    stay so: if the two ever disagree, the frozen vortex encounters and the
    737 layers would be flown at different altitudes for the same reason.

    The bracket is +/-400 m about the nominal, which is ~30x the largest match
    this has ever needed (43.22 ft, before the session-23 ISA fix reduced it to
    under a foot). A `ValueError` out of `brentq` means the two atmospheres have
    diverged far more than any ISA constant difference explains.
    """
    h = brentq(lambda z: float(density(z)) - rho,
               nominal_m - 400.0, nominal_m + 400.0, xtol=1e-12)
    return h, abs(float(density(h)) - rho) / rho


def load(path: Path = REFERENCE) -> Reference:
    root = ET.parse(path).getroot()

    conditions = {}
    for c in root.findall("condition"):
        rho = float(c.findtext("density"))
        matched, residual = _match_density(rho, float(c.findtext("altitude_m")))
        conditions[c.get("name")] = Condition(
            name=c.get("name"),
            altitude=float(c.findtext("altitude_m")),
            matched_altitude=matched,
            density=rho,
            sound_speed=float(c.findtext("sound_speed")),
            airspeed=float(c.findtext("airspeed")),
            mass=float(c.findtext("mass")),
            inertia=_floats(c.findtext("inertia")).reshape(3, 3),
            density_match_residual=residual,
        )

    trims = {
        t.get("mode"): Trim(
            mode=t.get("mode"),
            alpha=float(t.findtext("alpha")),
            elevator=float(t.findtext("elevator")),
            throttle=float(t.findtext("throttle")),
            thrust=float(t.findtext("thrust")),
            bank=float(t.findtext("bank")),
        )
        for t in root.findall("trim")
    }

    lin = root.find("linearization")
    linearization = Linearization(
        A=_floats(lin.findtext("A")).reshape(12, 12),
        B=_floats(lin.findtext("B")).reshape(12, 4),
        x0=_floats(lin.findtext("x0")),
        u0=_floats(lin.findtext("u0")),
    )

    trajectory = {
        tr.get("name"): [
            TrajectorySample(
                t=float(s.get("t")),
                vel_body=_floats(s.get("vel_body")),
                omega=_floats(s.get("omega")),
                euler=_floats(s.get("euler")),
                altitude=float(s.get("altitude")),
                controls=_floats(s.get("controls")),
                thrust=float(s.get("thrust")),
            )
            for s in tr.findall("sample")
        ]
        for tr in root.findall("trajectory")
    }

    return Reference(
        jsbsim_version=root.findtext("provenance/jsbsim_version"),
        condition=conditions,
        trim=trims,
        derivatives={d.get("name"): float(d.get("value"))
                     for d in root.findall("derivatives/derivative")},
        absent={d.get("name"): float(d.get("measured"))
                for d in root.findall("absent_derivatives/derivative")},
        entry={v.get("name"): float(v.text)
               for v in root.findall("aircraft_entry/value")},
        linearization=linearization,
        stall_sweep=[
            StallPoint(alpha=float(p.get("alpha")), CL=float(p.get("CL")))
            for p in root.findall("stall_sweep/point")
        ],
        trajectory=trajectory,
        sweep=[
            SweepPoint(
                vel_body=_floats(p.get("vel_body")),
                sound_speed=float(p.get("sound_speed")),
                alpha=float(p.get("alpha")),
                alphadot=float(p.get("alphadot")),
                ci2vel=float(p.get("ci2vel")),
                beta=float(p.get("beta")),
                rates=_floats(p.get("rates")),
                controls=_floats(p.get("controls")),
                coefficients=_floats(p.get("coefficients")),
            )
            for p in root.findall("sweep/point")
        ],
        diagnostics={d.tag: float(d.text) for d in root.find("diagnostics")},
    )
