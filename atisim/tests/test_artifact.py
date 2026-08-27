"""The run artifact: can a figure be traced back to the run that produced it?

Every number in PROJECT.md section 4's encounter tables came from a run that no
longer exists -- `viz.save` is called only by `scripts/fly.py`, and every
analysis script builds arrays in memory, draws a PNG and exits. These tests pin
the format that fixes that.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401
from atisim import checks, earth, integrate, trim, viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.analysis import artifact
from atisim.manual import Mode

pytest.importorskip("pyarrow", reason="the artifact layer needs the `ui` extra")

# 47N at the 747's cruise altitude. The anchor sits AT the flight altitude, which
# is what makes the run's start below a pure north displacement rather than a
# north-and-down one. WGS84_J2: an artifact records a real run.
ANCHOR = earth.anchor_at(np.radians(47.0), 0.0, CRUISE["boeing747"]["altitude"])
EARTH = earth.WGS84_J2


@pytest.fixture(scope="module")
def small_run():
    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    column = wind.UpdraftColumn(
        north=jnp.array(0.0), east=jnp.array(0.0), w0=jnp.array(10.0),
        radius=jnp.array(2000.0), sharpness=jnp.array(6.0),
    )
    field = lambda p: wind.updraft_wind(p, column)  # noqa: E731
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac, ANCHOR, EARTH)
    controls = trim.trimmed_controls(x)
    # `x[3]` is the trimmed BANK, which is non-zero on a rotating Earth. The run
    # starts 4 km south of the column, as a displacement along the ANCHOR frame's
    # north axis -- not `pos_ned = [-4000, 0, -H]`, which under an anchor already
    # at H would have put the aircraft a whole cruise altitude ABOVE it.
    state = trim.trimmed_state(
        jnp.array(float(x[0])), jnp.array(float(x[3])),
        jnp.array(V), jnp.array(H), ANCHOR, jnp.array(0.0),
    )
    state = state._replace(
        pos_ecef=state.pos_ecef + ANCHOR.T_e2l.T @ jnp.array([-4000.0, 0.0, 0.0])
    )
    sim = integrate.init_sim(state, jax.random.PRNGKey(0))
    recorder = viz.Recorder(ANCHOR)
    for i in range(120):
        sim = integrate.step(
            sim, controls, jnp.array(0.01), ac, ANCHOR, EARTH,
            wind_model=wind.field_model(field, ANCHOR),
        )
        recorder.append((i + 1) * 0.01, sim, controls, Mode.MANUAL)
    return recorder.trajectory(), ac, controls, field


def _meta(ac):
    return artifact.build_meta(
        aircraft_key="boeing747",
        aircraft=ac,
        flight_condition={"airspeed_mps": 235.9, "altitude_m": 12192.0},
        trim_solution={"alpha_rad": 0.0809, "elevator_rad": -0.01,
                       "throttle": 0.3, "residual_norm": 1.9e-20,
                       "is_physical": True},
        integrator={"dt_s": 0.01, "n_steps": 120},
        wind_field={"kind": "UpdraftColumn", "source": "Wingrove & Bach 1994",
                    "params": {"w0": 10.0, "radius": 2000.0, "sharpness": 6.0}},
        declared_parameters={"sharpness": 6.0, "window_rule": "column"},
        caveats=["Load comparisons are ORDERING ONLY (PROJECT.md 5)."],
        anchor=ANCHOR,
        earth_model=EARTH,
    )


def test_a_run_round_trips_bit_identically(tmp_path, small_run):
    """`np.array_equal`, not a tolerance. A lossy artifact is a different run."""
    traj, ac, _, _ = small_run
    artifact.write_run(tmp_path / "run", traj, _meta(ac), [])
    back = artifact.read_run(tmp_path / "run")

    for name in viz.Trajectory._fields:
        if name == "anchor":
            # Not a column: `anchor_spec` writes the three geodetic scalars and
            # `read_run` rebuilds `r_ecef` and `T_e2l` through `anchor_at`. The
            # rebuild has to be bit-identical too, which is what this compares.
            for saved, loaded in zip(traj.anchor, back.trajectory.anchor):
                assert np.array_equal(np.asarray(saved), np.asarray(loaded)), name
            continue
        assert np.array_equal(getattr(back.trajectory, name), getattr(traj, name)), name


def test_the_artifact_loads_as_a_viz_trajectory(tmp_path, small_run):
    """`viz.derived` and `viz.post_flight` must work on it with no simulator.

    That is the standard `viz.load` already meets, and the artifact must not be
    a second, incompatible run format.
    """
    traj, ac, _, _ = small_run
    artifact.write_run(tmp_path / "run", traj, _meta(ac), [])
    back = artifact.read_run(tmp_path / "run")
    assert isinstance(back.trajectory, viz.Trajectory)
    d = viz.derived(back.trajectory)
    assert np.isfinite(d.alpha).all()


def test_every_column_carries_its_units_and_frame(tmp_path, small_run):
    """A column without units is a column somebody will plot in the wrong ones."""
    import pyarrow.parquet as pq

    traj, ac, _, _ = small_run
    artifact.write_run(tmp_path / "run", traj, _meta(ac), [])
    schema = pq.read_schema(tmp_path / "run" / "run.parquet")

    for field in schema:
        meta = field.metadata or {}
        assert b"units" in meta, f"{field.name} has no units"
        assert b"frame" in meta, f"{field.name} has no frame"


def test_the_metadata_round_trips_and_carries_provenance(tmp_path, small_run):
    traj, ac, _, _ = small_run
    meta = _meta(ac)
    artifact.write_run(tmp_path / "run", traj, meta, [])
    back = artifact.read_run(tmp_path / "run")

    assert back.meta["aircraft"]["key"] == "boeing747"
    assert back.meta["wind_field"]["kind"] == "UpdraftColumn"
    assert back.meta["declared_parameters"]["sharpness"] == 6.0
    assert back.meta["caveats"]
    assert len(back.meta["git_sha"]) in (0, 40)
    assert back.meta["conventions"]["quaternion"].startswith("[w,x,y,z]")
    assert back.meta["config_hash"] == meta["config_hash"]


def test_the_config_hash_moves_when_the_experiment_moves(small_run):
    """Two runs are the same experiment iff their config hashes agree."""
    _, ac, _, _ = small_run
    a = _meta(ac)
    b = artifact.build_meta(
        aircraft_key="boeing747", aircraft=ac,
        flight_condition={"airspeed_mps": 235.9, "altitude_m": 12192.0},
        trim_solution=a["trim"],
        integrator={"dt_s": 0.005, "n_steps": 240},   # <- halved step
        wind_field=a["wind_field"],
        declared_parameters=a["declared_parameters"],
        caveats=a["caveats"],
        anchor=ANCHOR,
        earth_model=EARTH,
    )
    assert a["config_hash"] != b["config_hash"]


def test_the_derivative_hash_distinguishes_two_aircraft(small_run):
    _, ac, _, _ = small_run
    other = REGISTRY["boeing747_approach"]
    assert artifact.derivative_hash(ac) != artifact.derivative_hash(other)
    assert artifact.derivative_hash(ac) == artifact.derivative_hash(ac)


def test_checks_round_trip_with_their_kind(tmp_path, small_run):
    """The UI reads these; it must never recompute them (design section 3.1)."""
    traj, ac, controls, field = small_run
    window = np.ones(len(traj.t), dtype=bool)
    report = checks.run_checks(traj, ac, controls, field, window, EARTH)
    artifact.write_run(tmp_path / "run", traj, _meta(ac), report)
    back = artifact.read_run(tmp_path / "run")

    assert len(back.checks) == len(report)
    for got, want in zip(back.checks, report):
        assert got["name"] == want.name
        assert got["kind"] == want.kind
        assert got["passed"] == want.passed
        assert got["value"] == pytest.approx(want.value)


def test_a_run_from_another_commit_is_readable_and_flagged(tmp_path, small_run):
    """Reading must not raise -- but the mismatch must be visible.

    An artifact whose figures are being compared against today's code is exactly
    when a silent version skew matters.
    """
    traj, ac, _, _ = small_run
    meta = _meta(ac)
    meta["git_sha"] = "0" * 40
    artifact.write_run(tmp_path / "run", traj, meta, [])
    back = artifact.read_run(tmp_path / "run")

    assert back.meta["git_sha"] == "0" * 40
    assert back.stale is True


def test_the_field_is_rebuilt_from_parameters_not_stored_as_samples(small_run):
    """The stored spec must reproduce the flown field exactly.

    This is what licenses NOT persisting a sampled grid: a 64^3 grid is 6.3 MB
    and costs 0.56-2.84 ms to recompute, so storing it would be slower to read
    than to rebuild and would be a second copy of these six floats.
    """
    traj, ac, _, field = small_run
    rebuilt = artifact.rebuild_field({
        "wind_field": {"kind": "UpdraftColumn",
                       "params": {"w0": 10.0, "radius": 2000.0, "sharpness": 6.0}}
    })
    # `viz.pos_ned`, not a column: the field's argument is a LOCAL NED offset
    # from the anchor, which the log no longer stores and now derives.
    probes = jnp.asarray(viz.pos_ned(traj)[::17])
    got = np.asarray(jax.vmap(rebuilt)(probes))
    want = np.asarray(jax.vmap(field)(probes))
    assert np.array_equal(got, want)


def test_rebuilding_an_unknown_field_raises_rather_than_going_calm(small_run):
    """Silently substituting still air draws a flat gust trace, which reads as a
    working aircraft in calm conditions rather than as a failure."""
    with pytest.raises(ValueError, match="refusing to substitute still air"):
        artifact.rebuild_field({"wind_field": {"kind": "Dryden", "params": {}}})


def test_a_zero_wind_run_rebuilds_as_still_air(small_run):
    """The manoeuvring category IS zero wind, and that is a field, not an absence."""
    f = artifact.rebuild_field({"wind_field": {"kind": "none (zero wind)"}})
    assert np.array_equal(np.asarray(f(jnp.zeros(3))), np.zeros(3))


def test_the_json_files_are_human_readable(tmp_path, small_run):
    """A run directory has to be inspectable without this codebase."""
    traj, ac, _, _ = small_run
    artifact.write_run(tmp_path / "run", traj, _meta(ac), [])
    text = (tmp_path / "run" / "meta.json").read_text()
    assert "\n" in text  # indented, not one line
    assert json.loads(text)["schema_version"] == artifact.SCHEMA_VERSION
