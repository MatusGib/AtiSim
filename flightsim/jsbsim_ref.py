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

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_737_reference.xml"


class Condition(NamedTuple):
    """The flight condition, and the altitude flightsim must be run at.

    `matched_altitude` is NOT `altitude`. flightsim's ISA uses geometric
    altitude where the standard uses geopotential, so its density at a nominal
    30,000 ft is 0.159% below JSBSim's. Since qbar is proportional to rho, that
    would put the same-signed bias on every force in every layer. The generator
    solves for the geometric altitude at which flightsim's density equals
    JSBSim's -- 43.22 ft lower -- and that is what must be used.
    """

    name: str
    altitude: float           # m, what JSBSim was run at
    matched_altitude: float   # m, what flightsim must be run at
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
    # Driving flightsim with the VECTOR rather than with (V, alpha, beta)
    # keeps the comparison independent of how either engine defines alpha
    # and beta -- they agree to nine decimals, but this needs no such check.
    vel_body: np.ndarray      # (3,) m/s
    sound_speed: float        # m/s, JSBSim's at this point
    alpha: float
    beta: float
    rates: np.ndarray         # (3,) rad/s, body, air-relative
    controls: np.ndarray      # (3,) rad: elevator, aileron, rudder
    coefficients: np.ndarray  # (6,) CL, CD, CY, Cl, Cm, Cn


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
    linearization: Linearization
    trajectory: dict
    diagnostics: dict


def _floats(text):
    return np.array([float(v) for v in text.split()])


def load(path: Path = REFERENCE) -> Reference:
    root = ET.parse(path).getroot()

    conditions = {}
    for c in root.findall("condition"):
        conditions[c.get("name")] = Condition(
            name=c.get("name"),
            altitude=float(c.findtext("altitude_m")),
            matched_altitude=float(c.findtext("matched_altitude_m")),
            density=float(c.findtext("density")),
            sound_speed=float(c.findtext("sound_speed")),
            airspeed=float(c.findtext("airspeed")),
            mass=float(c.findtext("mass")),
            inertia=_floats(c.findtext("inertia")).reshape(3, 3),
            density_match_residual=float(c.findtext("density_match_residual")),
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
        trajectory=trajectory,
        sweep=[
            SweepPoint(
                vel_body=_floats(p.get("vel_body")),
                sound_speed=float(p.get("sound_speed")),
                alpha=float(p.get("alpha")),
                beta=float(p.get("beta")),
                rates=_floats(p.get("rates")),
                controls=_floats(p.get("controls")),
                coefficients=_floats(p.get("coefficients")),
            )
            for p in root.findall("sweep/point")
        ],
        diagnostics={d.tag: float(d.text) for d in root.find("diagnostics")},
    )
