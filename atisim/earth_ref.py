"""Reader for the frozen JSBSim Earth reference.

Deliberately does NOT import jsbsim, and nothing that imports this module may.
Same rule as jsbsim_ref.py and for the same reason: the suite must run on a
machine with no JSBSim, and reference drift must show up in git diff rather
than as a mysterious change in test results.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

import numpy as np

REFERENCE = Path(__file__).parent / "tests" / "data" / "jsbsim_earth_reference.xml"


class Probe(NamedTuple):
    lat: float
    lon: float
    r_ecef: np.ndarray       # (3,) m
    euler: np.ndarray        # (3,) rad, phi theta psi
    t_l2b: np.ndarray        # (3,3) local NED -> body
    t_e2l: np.ndarray        # (3,3) ECEF -> local NED
    mass: float              # kg
    inertia: np.ndarray      # (3,3) kg m^2, +Ixz convention
    force: np.ndarray        # (3,) N, body, EXCLUDING weight
    weight: np.ndarray       # (3,) N, body
    moment: np.ndarray       # (3,) N m, body
    gravity_magnitude: float # m/s^2
    vel_body: np.ndarray     # (3,) m/s, ECEF-relative, body axes
    vel_ecef: np.ndarray     # (3,) m/s, ECEF frame
    vel_ned: np.ndarray      # (3,) m/s
    pqr: np.ndarray          # (3,) rad/s, body rel ECEF
    pqri: np.ndarray         # (3,) rad/s, body rel ECI
    uvwdot: np.ndarray       # (3,) m/s^2
    pqrdot: np.ndarray       # (3,) rad/s^2
    pqridot: np.ndarray      # (3,) rad/s^2


def _v(node, name, shape=None):
    text = node.find(name).text
    a = np.array([float(x) for x in text.split()])
    return a.reshape(shape) if shape else a


def load(path: Path = REFERENCE) -> list[Probe]:
    root = ET.parse(path).getroot()
    out = []
    for node in root.findall("probe"):
        out.append(Probe(
            lat=float(node.find("lat").text),
            lon=float(node.find("lon").text),
            r_ecef=_v(node, "r_ecef"),
            euler=_v(node, "euler"),
            t_l2b=_v(node, "t_l2b", (3, 3)),
            t_e2l=_v(node, "t_e2l", (3, 3)),
            mass=float(node.find("mass").text),
            inertia=_v(node, "inertia", (3, 3)),
            force=_v(node, "force"),
            weight=_v(node, "weight"),
            moment=_v(node, "moment"),
            gravity_magnitude=float(node.find("gravity_magnitude").text),
            vel_body=_v(node, "vel_body"),
            vel_ecef=_v(node, "vel_ecef"),
            vel_ned=_v(node, "vel_ned"),
            pqr=_v(node, "pqr"),
            pqri=_v(node, "pqri"),
            uvwdot=_v(node, "uvwdot"),
            pqrdot=_v(node, "pqrdot"),
            pqridot=_v(node, "pqridot"),
        ))
    return out
