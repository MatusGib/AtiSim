"""The application reaches every simulation the engine can run.

`atisim.coverage` is the map; these tests hold the engine to it. When the engine
gains a wind field, an aircraft or an integrator option that no run kind flies,
a test here fails and names it, which is what keeps "the UI runs every
simulation" true as the engine grows.
"""

import inspect
import re

import jax.numpy as jnp
import numpy as np
import pytest

import atisim  # noqa: F401
from atisim import coverage, integrate, run, trim, wind
from atisim.aircraft import CRUISE, REGISTRY

_ENTRY = re.compile(r"(_wind|_field|_model|_gust)$")


def _wind_entry_points():
    return sorted(
        name for name, fn in inspect.getmembers(wind, inspect.isfunction)
        if fn.__module__ == wind.__name__ and not name.startswith("_") and _ENTRY.search(name)
    )


def test_every_wind_entry_point_is_reached_or_named_as_not_a_run():
    missing = [n for n in _wind_entry_points()
               if n not in coverage.REACHED and n not in coverage.NOT_A_RUN]
    assert missing == [], (
        f"atisim.wind gained {missing}: give it a run kind or option, then list it in "
        "atisim.coverage.REACHED, or list it in NOT_A_RUN with the reason")


def test_the_coverage_map_names_only_functions_that_exist():
    stale = [n for n in {**coverage.REACHED, **coverage.NOT_A_RUN}
             if not hasattr(wind, n)]
    assert stale == []


def _resolve(dotted):
    import importlib

    module, name = dotted.split(".", 1)
    return getattr(importlib.import_module(f"atisim.{module}"), name, None)


def test_every_named_simulation_exists_and_so_does_what_reaches_it():
    from atisim import analyses

    for dotted, where in coverage.SIMULATIONS.items():
        assert _resolve(dotted) is not None, f"{dotted} is gone from the engine"
        if where.startswith("analys"):
            keys = where.split(" ", 1)[1].split(", ")
            assert all(k in analyses.ANALYSES for k in keys), (dotted, keys)
    for dotted in coverage.NOT_YET:
        assert _resolve(dotted) is not None, f"{dotted} is gone from the engine"


def test_every_analysis_reaches_something_named():
    """An analysis that reaches nothing in the map is either unlisted or idle."""
    from atisim import analyses

    named = " ".join(coverage.SIMULATIONS.values())
    # A study runs a script from scripts/, whatever that calls; test_studies.py
    # holds that every script but the two helpers is one.
    missing = [k for k in analyses.ANALYSES if k not in named and k != "study"]
    assert missing == []


def test_the_engines_simulation_functions_are_all_accounted_for():
    """New `fly*`, `measure_*` or verification functions fail here by name."""
    import inspect

    from atisim import gust, verification, vortex_viz

    found = []
    for module, rule in ((vortex_viz, lambda n: n.startswith("fly") or n == "manoeuvre"),
                         (gust, lambda n: n.startswith("measure_")),
                         (verification, lambda n: n != "without_aerodynamics")):
        short = module.__name__.split(".")[-1]
        found += [f"{short}.{n}" for n, fn in inspect.getmembers(module, inspect.isfunction)
                  if fn.__module__ == module.__name__ and not n.startswith("_") and rule(n)]
    missing = [n for n in found if n not in coverage.SIMULATIONS and n not in coverage.NOT_YET]
    assert missing == [], f"reach {missing} from an analysis or run kind, or list it in NOT_YET"


def test_every_integrator_option_is_a_run_setting():
    options = [n for n in inspect.signature(integrate.step).parameters
               if n not in ("sim", "controls", "dt", "ac")]
    assert sorted(options) == sorted(coverage.STEP_OPTIONS)
    for field in coverage.STEP_OPTIONS.values():
        assert field in run.RunSpec._fields


@pytest.mark.parametrize("key", sorted(REGISTRY))
def test_every_aircraft_is_selectable_and_trims_at_cruise(key):
    spec = run.BLANK._replace(aircraft=key, airspeed_mps=CRUISE[key]["airspeed"],
                              altitude_m=CRUISE[key]["altitude"])
    assert not run.errors(spec)
    x, res = trim.trim(jnp.array(spec.airspeed_mps), jnp.array(spec.altitude_m),
                       REGISTRY[key])
    assert float(jnp.linalg.norm(res)) < 1e-9
    assert trim.is_physical(x, REGISTRY[key])


def test_every_field_kind_has_a_preset():
    kinds = {spec.wind.kind for spec in run.PRESETS.values()}
    missing = [k for k in run.PARAMETERS
               if k not in kinds and k not in ("none", "ModulatedDryden")]
    assert missing == []


@pytest.mark.parametrize("name", list(run.PRESETS))
def test_every_preset_flies_briefly(name):
    """Two seconds at a coarse step: enough to trim, build the field, fly and
    check. A preset that cannot fly fails here rather than in front of a user.
    The manoeuvre keeps its own length and step: its elevator is bisected over
    the whole pulse."""
    if name == "manoeuvre":
        spec = run.PRESETS[name]
    else:
        spec = run.PRESETS[name]._replace(seconds=2.0, dt=0.05)
    flown = run.fly(spec)
    assert len(flown.trajectory.t) == int(round(run.geometry(spec).seconds / spec.dt))
    assert np.isfinite(np.asarray(flown.trajectory.pos_ned)).all()
