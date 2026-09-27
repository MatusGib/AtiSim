"""Wind fields rebuilt from the few numbers an artifact stores, for every kind.

`artifact.rebuild_field` began with the four fields the basic UI flies, each an
`if` branch. The phase-2 run kinds (single cores, Mehta's array, the
stochastic spectra, the test inputs) would have made that function the second
copy of every field's construction, beside `run.build_field`. So the new kinds
are built HERE, from their metadata parameters only, and both sides call this
module: `run.build_field` builds a new kind by passing `wind_meta(spec)` to
`build`, which makes "the flown field is the rebuilt field" true by
construction rather than by a test that two copies agree.

The four original kinds keep their own branches in `artifact.rebuild_field`,
untouched, so every existing artifact rebuilds exactly as it did. `vortex_array`
extends that branch with three optional keys (`profile`, `form`,
`path_altitude`), each defaulting to the old behaviour.

Also here: the turbulence OVERLAY, a stochastic field added on top of a
deterministic one with `wind.superpose` -- the "Hannibal plus Dryden" experiment
the v1.2 plan sizes by hand. It is stored as `meta["wind_field"]["overlay"]`, a
second `{kind, params}` block, and `with_overlay` applies it.
"""

import jax.numpy as jnp

from atisim import wind

# The stochastic kinds, and the `wind` function each one is. Every one of them
# takes (sigma, seed) and the same four grid keywords, which `_grid` passes on
# only when the metadata carries them, so the engine's defaults stay the
# engine's.
_GRID_KEYS = ("n_components", "wavelength_min", "wavelength_max")

STOCHASTIC = ("Dryden", "VonKarman", "GaussianDryden", "ModulatedDryden")

# Kinds whose field is a pure function of position with no randomness in it.
DETERMINISTIC = ("Sinusoid", "OneMinusCosine")

KINDS = STOCHASTIC + DETERMINISTIC


def available(kind: str) -> bool:
    """Whether the engine in hand can build `kind`.

    `ModulatedDryden` is research code that the public tree does not carry
    (`2026-09-25-v1.2-plan.md`), so it is offered only where
    `wind.modulated_vertical_field` exists.
    """
    if kind == "ModulatedDryden":
        return hasattr(wind, "modulated_vertical_field")
    return kind in KINDS


def _grid(p: dict) -> dict:
    return {k: p[k] for k in _GRID_KEYS if k in p}


def _vertical(fn, p: dict, scale_key: str = "L_w"):
    kwargs = _grid(p)
    if scale_key in p:
        kwargs[scale_key] = float(p[scale_key])
    return fn(float(p["sigma_w"]), int(p["seed"]), **kwargs)


def build(kind: str, p: dict):
    """The field for a phase-2 kind, as a callable of NED position.

    Raises on a kind it does not know, for the reason `rebuild_field` does: a
    field that silently became still air would draw a calm run.
    """
    if kind == "Dryden":
        if p.get("components", "vertical") == "three-axis":
            kwargs = _grid(p)
            if "L_w" in p:
                kwargs["L"] = float(p["L_w"])
            return wind.dryden_field(float(p["sigma_w"]), int(p["seed"]), **kwargs)
        return _vertical(wind.dryden_vertical_field, p)
    if kind == "VonKarman":
        return _vertical(wind.von_karman_vertical_field, p)
    if kind == "GaussianDryden":
        return _vertical(wind.gaussian_vertical_field, p)
    if kind == "ModulatedDryden":
        if not available(kind):
            raise ValueError(
                "ModulatedDryden needs wind.modulated_vertical_field, which this "
                "tree does not carry (it is archive research code)")
        kwargs = _grid(p)
        if "L_w" in p:
            kwargs["L_w"] = float(p["L_w"])
        return wind.modulated_vertical_field(
            float(p["sigma_w"]), int(p["seed"]), depth=float(p["depth"]),
            patch_length=float(p["patch_length"]), **kwargs)
    if kind == "Sinusoid":
        return wind.sinusoidal_vertical_field(
            float(p["amplitude"]), float(p["wavelength"]), float(p.get("phase", 0.0)))
    if kind == "OneMinusCosine":
        return wind.one_minus_cosine_gust(
            float(p["peak"]), float(p["gradient_distance"]),
            float(p.get("start_north", 0.0)))
    raise ValueError(
        f"unknown wind_field kind {kind!r}: refusing to substitute still air, "
        "which would draw a flat gust trace and look like a calm run"
    )


def vortex_array(p: dict):
    """A `VortexArray` field from stored parameters, with the phase-2 options.

    `profile` "lamb-oseen" selects `wind.lamb_oseen_wind` (same r0 and V0, no
    kink at the core edge). `form` "line" selects `wind.line_vortex_wind`, the
    three-dimensional form that differs off the flight path. `path_altitude`
    replays the field on the level path it was identified along
    (`wind.on_identified_path`), the Mehta headline's form. All three default to
    the field every earlier artifact flew.
    """
    array = wind.VortexArray(
        north=jnp.array(p["north"]), down=jnp.array(p["down"]),
        r0=jnp.array(p["r0"]), v0=jnp.array(p["v0"]),
        cos_dpsi=jnp.array(p.get("cos_dpsi", 1.0)),
        sin_dpsi=jnp.array(p.get("sin_dpsi", 0.0)),
    )
    if p.get("form", "point") == "line":
        if p.get("profile", "rankine") != "rankine":
            raise ValueError("the line form is Rankine only: wind.line_vortex_wind "
                             "has no Lamb-Oseen sibling")
        fn = wind.line_vortex_wind
    elif p.get("profile", "rankine") == "lamb-oseen":
        fn = wind.lamb_oseen_wind
    else:
        fn = wind.vortex_wind

    def field(pos_ned):
        return fn(pos_ned, array)

    if p.get("path_altitude") is not None:
        return wind.on_identified_path(field, float(p["path_altitude"]))
    return field


def with_overlay(field, spec: dict):
    """`field` plus the overlay `spec["overlay"]` describes, if there is one."""
    overlay = spec.get("overlay")
    if not overlay:
        return field
    return wind.superpose(field, build(overlay["kind"], overlay.get("params", {})))
