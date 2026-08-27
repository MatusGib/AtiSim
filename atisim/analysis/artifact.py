"""The run artifact: a flown run, on disk, traceable to what produced it.

**The gap this closes.** `viz.save`/`viz.load` exist and work, but the only
caller is `scripts/fly.py` -- the interactive panel. `scripts/vortex.py`,
`leewave.py` and `microburst.py` each build arrays in memory, draw a PNG and
exit, so every number in PROJECT.md section 4's encounter tables came from a run
that no longer exists. And `viz.Trajectory` could not have carried them anyway:
nine array fields, no aircraft identity, no wind-field identity, no dt, no trim
solution, no declared parameters and no git SHA.

What it does carry is the ARRAYS, and those are right, so this format keeps
`viz.Trajectory` as the payload and adds the metadata around it. `read_run`
returns a real `viz.Trajectory`, so `viz.derived` and `viz.post_flight` work on
an artifact unchanged.

**Format: Parquet for the series, JSON beside it -- and the reason is not
speed.** Measured on a 4,018-sample run: Parquet+zstd is 0.263 MB and reads in
0.81 ms; npz-compressed is 0.198 MB and reads in 2.94 ms. npz is SMALLER, and
neither is remotely a bottleneck. Parquet is chosen because it carries typed
per-column metadata natively -- which is where `units` and `frame` have to live
-- and because a directory of Parquet is a dataset any tool can open. The cost,
stated plainly: `pyarrow` is a ~90 MB wheel against a four-package runtime list,
which is why it is the `ui` extra and not a dependency.

**The field is NOT stored.** Sampling any of the project's four fields on a 64^3
grid costs 0.56-2.84 ms, and that grid is 6.3 MB of float64 -- so reading it back
would be slower than recomputing it, and it would be a second copy of something
already determined exactly by a four-to-six-float NamedTuple. The artifact stores
the field SPECIFICATION and the UI rebuilds the field. What is stored is
`wind_ned` and `omega_gust` AT THE AIRCRAFT, because those are the realisation
actually flown and a stochastic model could not be re-derived.
"""

import hashlib
import json
import subprocess
from pathlib import Path
from typing import NamedTuple

import jax
import numpy as np

from atisim import viz
from atisim.aircraft import Aircraft

# 2: the ECEF state. `pos_n/e/d` became `pos_x/y/z`, the quaternion became
# body -> ECEF, and `anchor` joined the metadata. A version-1 artifact is NOT
# readable as a version-2 one: its columns carry the same shapes under different
# frames, so it would be read in full and wrongly. `read_run` refuses instead.
SCHEMA_VERSION = 2

# Conventions travel WITH the data. `state.py` owns them; restating them in the
# artifact means a run stays interpretable by something that never imported this
# package, which is the whole point of writing an artifact rather than a pickle.
CONVENTIONS = {
    "frame": (
        "ECEF, rotating with the Earth; body x-forward, y-right, z-down. "
        "Position is an OFFSET from the anchor in `anchor`, not an absolute "
        "ECEF coordinate. Local NED is a derived view, taken at the aircraft's "
        "own geodetic position: pos_ned = anchor.T_e2l @ pos_xyz."
    ),
    "quaternion": "[w,x,y,z], unit norm, rotates BODY vectors into ECEF",
    "altitude": (
        "GEODETIC height above the WGS-84 ellipsoid, from ecef_to_geodetic. It "
        "is NOT -pos_d: the anchor's tangent plane falls away from the "
        "ellipsoid as d^2/2R, 785 m at 100 km of ground track."
    ),
    "velocity": "vel_body and omega are relative to ECEF, not to inertial space",
    "vertical_gust_sign": (
        "wind_d is positive DOWN. Plots show -wind_d, i.e. up positive, which is "
        "the sign the sources quote vertical gusts in."
    ),
    "load_factor": "body-normal n_z, +1 in level flight; at trim it is cos(theta)",
}

# column -> (Trajectory field, component index or None, units, frame)
_COLUMNS: list[tuple[str, str, int | None, str, str]] = [
    ("t", "t", None, "s", "elapsed"),
    ("pos_x", "pos_ecef", 0, "m", "ECEF, OFFSET from the anchor"),
    ("pos_y", "pos_ecef", 1, "m", "ECEF, OFFSET from the anchor"),
    ("pos_z", "pos_ecef", 2, "m", "ECEF, OFFSET from the anchor"),
    ("vel_u", "vel_body", 0, "m/s", "body, ECEF-relative"),
    ("vel_v", "vel_body", 1, "m/s", "body, ECEF-relative"),
    ("vel_w", "vel_body", 2, "m/s", "body, ECEF-relative"),
    ("quat_w", "quat", 0, "-", "body->ECEF"),
    ("quat_x", "quat", 1, "-", "body->ECEF"),
    ("quat_y", "quat", 2, "-", "body->ECEF"),
    ("quat_z", "quat", 3, "-", "body->ECEF"),
    ("omega_p", "omega", 0, "rad/s", "body, relative to ECEF"),
    ("omega_q", "omega", 1, "rad/s", "body, relative to ECEF"),
    ("omega_r", "omega", 2, "rad/s", "body, relative to ECEF"),
    ("elevator", "controls", 0, "rad", "positive trailing-edge down"),
    ("aileron", "controls", 1, "rad", "positive right-roll command"),
    ("rudder", "controls", 2, "rad", "positive trailing-edge left"),
    ("throttle", "controls", 3, "0-1", "-"),
    ("mode", "mode", None, "enum", "manual.Mode"),
    ("wind_n", "wind_ned", 0, "m/s", "NED, AS APPLIED (one-step cache)"),
    ("wind_e", "wind_ned", 1, "m/s", "NED, AS APPLIED (one-step cache)"),
    ("wind_d", "wind_ned", 2, "m/s", "NED positive DOWN, AS APPLIED"),
    ("gust_p", "omega_gust", 0, "rad/s", "body, AS APPLIED. SIM TRUTH -- unsensable"),
    ("gust_q", "omega_gust", 1, "rad/s", "body, AS APPLIED. SIM TRUTH -- unsensable"),
    ("gust_r", "omega_gust", 2, "rad/s", "body, AS APPLIED. SIM TRUTH -- unsensable"),
]


class Run(NamedTuple):
    """One artifact, read back.

    `stale` says the artifact was written by a different commit than the one
    reading it. Not an error -- re-reading old runs is the point of writing them
    -- but a figure being compared against today's code needs to say so.
    """

    trajectory: viz.Trajectory
    meta: dict
    checks: list[dict]
    path: Path
    stale: bool


def git_sha() -> str:
    """The commit this run was flown at, or "" if git cannot say.

    Empty rather than a guess: PROJECT.md's standing rule is flag, never invent,
    and a wrong SHA is worse than an absent one.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent.parent,
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return ""


def derivative_hash(ac: Aircraft) -> str:
    """Identity of an aircraft's numbers, so "same aircraft?" is a comparison.

    Hashes the flattened pytree rather than the name: two entries can share a key
    and differ in a derivative, which is exactly the case a comparison view must
    not miss. Sorted by the tree's own leaf order, which `jax.tree.flatten` fixes.
    """
    leaves = jax.tree.leaves(ac)
    digest = hashlib.sha256()
    for leaf in leaves:
        digest.update(np.asarray(leaf, dtype=float).tobytes())
    return digest.hexdigest()


def anchor_spec(anchor) -> dict:
    """The three geodetic scalars an `earth.Anchor` is defined by.

    `r_ecef` and `T_e2l` are DERIVED from these by `earth.anchor_at`, so only the
    definition is stored. Writing the derived fields as well is how an artifact
    ends up disagreeing with itself after a geodesy constant is corrected.
    """
    return {
        "lat": float(anchor.lat),
        "lon": float(anchor.lon),
        "h": float(anchor.h),
        "datum": "WGS-84, geodetic latitude, radians and metres",
    }


def build_meta(*, aircraft_key: str, aircraft: Aircraft, flight_condition: dict,
               trim_solution: dict, integrator: dict, wind_field: dict,
               declared_parameters: dict, caveats: list[str],
               anchor, earth_model,
               load_model: str | None = None,
               loading_shape: str | None = None) -> dict:
    """Assemble the metadata block, and hash the parts that define the experiment.

    `anchor` and `earth_model` are REQUIRED, and they are inside `config_hash`.
    Two runs at the same airspeed and altitude but different latitudes are
    different experiments now -- Coriolis, the gravity magnitude and the trimmed
    bank all move with it -- and a hash that could not tell them apart would let
    a comparison view treat them as repeats of each other.

    `declared_parameters` and `caveats` are not decoration and are not optional.
    The project's standing rule is flag, never invent; a declared parameter that
    does not travel with its result is the exact failure `provenance.py` exists
    to prevent, and `caveats` is what stops an ordering-only result being
    displayed as a value claim.

    `config_hash` deliberately EXCLUDES `created` and the git SHA: it answers
    "is this the same experiment?", and re-flying the same experiment tomorrow
    must produce the same hash.
    """
    meta = {
        "schema_version": SCHEMA_VERSION,
        "git_sha": git_sha(),
        "created": None,  # stamped by write_run
        "conventions": dict(CONVENTIONS),
        "aircraft": {
            "key": aircraft_key,
            "derivative_hash": derivative_hash(aircraft),
        },
        "flight_condition": flight_condition,
        "anchor": anchor_spec(anchor),
        "earth_model": dict(earth_model._asdict()),
        "trim": trim_solution,
        "integrator": {
            "scheme": "RK4 fixed step",
            "float": "float64",
            "wind_sampling": "once per step, held across four stages",
            "observed_order": (
                "4 in still air; 1 in a spatially varying field -- the hold is an "
                "O(h) perturbation inside the step (PROJECT.md section 4)"
            ),
            **integrator,
        },
        "wind_field": wind_field,
        "load_model": load_model,
        "loading_shape": loading_shape,
        "declared_parameters": declared_parameters,
        "caveats": list(caveats),
    }
    payload = {k: v for k, v in meta.items() if k not in ("created", "git_sha")}
    meta["config_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return meta


def _table(traj: viz.Trajectory):
    import pyarrow as pa

    arrays, fields = [], []
    for name, source, index, units, frame in _COLUMNS:
        data = np.asarray(getattr(traj, source))
        column = data if index is None else data[:, index]
        arrays.append(pa.array(np.ascontiguousarray(column)))
        fields.append(
            pa.field(name, arrays[-1].type,
                     metadata={"units": units, "frame": frame})
        )
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def write_run(directory, traj: viz.Trajectory, meta: dict, check_report) -> Path:
    """Write one run. Returns the directory.

    zstd rather than snappy: measured 0.263 MB against 0.293 for the same run,
    at the same 0.8 ms read. Neither matters at these sizes; the smaller one is
    free.
    """
    import pyarrow.parquet as pq

    from datetime import datetime, timezone

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    pq.write_table(_table(traj), directory / "run.parquet", compression="zstd")

    stamped = dict(meta)
    stamped["created"] = datetime.now(timezone.utc).isoformat()
    (directory / "meta.json").write_text(json.dumps(stamped, indent=2, default=str))

    rows = [c if isinstance(c, dict) else c.as_dict() for c in check_report]
    (directory / "checks.json").write_text(json.dumps(rows, indent=2))
    return directory


def read_earth_model(meta: dict):
    """The `earth.EarthModel` a run was flown under, from its metadata.

    Rebuilt rather than looked up by name, so a model that is not one of the
    three module constants still round-trips.
    """
    from atisim import earth

    return earth.EarthModel(**meta["earth_model"])


def read_run(directory) -> Run:
    """Read a run back.

    A GIT SKEW IS REPORTED, NEVER RAISED -- that is what `stale` is for, and
    re-reading old runs is the point of writing them. A SCHEMA skew past the
    ECEF state is a different thing and IS raised. A version-1 artifact's
    `pos_n/e/d` and body -> NED quaternion have exactly the shapes this reader
    wants, so they would load in full and be interpreted in the wrong frame with
    nothing to notice; and there is no anchor recorded to place them against.
    `earth.py` ships no default anchor for precisely that reason.
    """
    import pyarrow.parquet as pq

    from atisim import earth

    directory = Path(directory)
    meta = json.loads((directory / "meta.json").read_text())
    version = meta.get("schema_version", 1)
    if version < 2 or "anchor" not in meta:
        raise ValueError(
            f"{directory} is schema version {version}, written before the ECEF "
            "state. Its position and quaternion columns are local-NED under the "
            "names this reader gives to ECEF quantities, and it records no "
            "anchor to place them against -- so it cannot be read, only re-flown."
        )

    table = pq.read_table(directory / "run.parquet")
    columns = {name: table.column(name).to_numpy(zero_copy_only=False)
               for name, *_ in _COLUMNS}

    def stack(*names):
        return np.stack([columns[n] for n in names], axis=1)

    spec = meta["anchor"]
    traj = viz.Trajectory(
        t=columns["t"],
        pos_ecef=stack("pos_x", "pos_y", "pos_z"),
        vel_body=stack("vel_u", "vel_v", "vel_w"),
        quat=stack("quat_w", "quat_x", "quat_y", "quat_z"),
        omega=stack("omega_p", "omega_q", "omega_r"),
        controls=stack("elevator", "aileron", "rudder", "throttle"),
        mode=columns["mode"],
        wind_ned=stack("wind_n", "wind_e", "wind_d"),
        omega_gust=stack("gust_p", "gust_q", "gust_r"),
        anchor=earth.anchor_at(spec["lat"], spec["lon"], spec["h"]),
    )
    checks_path = directory / "checks.json"
    report = json.loads(checks_path.read_text()) if checks_path.exists() else []

    here = git_sha()
    stale = bool(here and meta.get("git_sha") and meta["git_sha"] != here)
    return Run(trajectory=traj, meta=meta, checks=report, path=directory, stale=stale)


def rebuild_field(meta: dict):
    """Reconstruct the wind field from the stored specification.

    The inverse of what `build_meta` records, and the reason the artifact stores
    four-to-six floats instead of a sampled grid. Measured: sampling any of these
    fields on a 64^3 grid costs 0.56-2.84 ms, and that grid is 6.3 MB of float64
    -- so reading it back from disk would be SLOWER than recomputing it, and it
    would be a second copy of something the parameters already determine exactly.

    Raises on an unknown kind rather than returning zero wind. A field silently
    becoming still air would make every gust panel read as a flat line, which is
    a picture of a working aircraft in calm conditions rather than of a failure.
    """
    import jax.numpy as jnp

    from atisim import wind

    spec = meta["wind_field"]
    kind, p = spec["kind"], spec.get("params", {})

    if kind.startswith("none"):
        return lambda pos_ned: jnp.zeros(3)
    if kind == "VortexArray":
        array = wind.VortexArray(
            north=jnp.array(p["north"]), down=jnp.array(p["down"]),
            r0=jnp.array(p["r0"]), v0=jnp.array(p["v0"]),
        )
        return lambda pos_ned: wind.vortex_wind(pos_ned, array)
    if kind == "UpdraftColumn":
        column = wind.UpdraftColumn(
            north=jnp.array(p.get("north", 0.0)), east=jnp.array(p.get("east", 0.0)),
            w0=jnp.array(p["w0"]), radius=jnp.array(p["radius"]),
            sharpness=jnp.array(p["sharpness"]),
        )
        return lambda pos_ned: wind.updraft_wind(pos_ned, column)
    if kind == "LeeWave":
        wave = wind.LeeWave(
            w0=jnp.array(p["w0"]), wavelength=jnp.array(p["wavelength"]),
            north=jnp.array(p.get("north", 0.0)),
        )
        return lambda pos_ned: wind.lee_wave_wind(pos_ned, wave)
    if kind == "Microburst":
        burst = wind.microburst(
            u_max=p["u_max"], radius=p["radius"], z_m=p["z_m"],
            north=p.get("north", 0.0), east=p.get("east", 0.0),
        )
        return lambda pos_ned: wind.microburst_wind(pos_ned, burst)
    raise ValueError(
        f"unknown wind_field kind {kind!r}: refusing to substitute still air, "
        "which would draw a flat gust trace and look like a calm run"
    )


def list_runs(root) -> list[Path]:
    """Every run directory under `root`, newest metadata first.

    Reads `meta.json` only -- a few kB -- so a sweep table over fifty runs never
    touches a Parquet file. The series are read on selection.
    """
    root = Path(root)
    if not root.exists():
        return []
    found = [p.parent for p in sorted(root.glob("*/meta.json"))]
    return sorted(found, key=lambda p: json.loads(
        (p / "meta.json").read_text()).get("created") or "", reverse=True)
