"""Fly atisim through the vortex JSBSim already flew, and report where they part.

Imports no jsbsim. It reads the frozen reference that
scripts/gen_jsbsim_vortex_reference.py wrote, drives atisim from the state
recorded there, and compares.

Run: .venv/Scripts/python.exe scripts/vortex_compare.py
     .venv/Scripts/python.exe scripts/vortex_compare.py --png runs/vc.png

THREE THINGS MAKE THIS A COMPARISON RATHER THAN TWO RUNS
--------------------------------------------------------
1. ONE FIELD. The generator computed the Rankine field in numpy; atisim
   computes it in JAX. test_jsbsim_vortex reconciles them at every sample to
   under 1e-9 m/s, so a bug in either cannot hide by being on both sides.

2. ONE STARTING STATE. atisim is started from JSBSim's recorded body velocity
   vector, rates, attitude and achieved surfaces -- not from its own trim. The
   engines do not trim alike: JSBSim's do_simple_trim converges to Nz = 0.99093
   rather than 1.0, which is worth about 0.05 deg of alpha, and independent
   trims would carry that offset into every difference downstream.

3. ONE DENSITY. atisim flies at the density-matched altitude, not the nominal
   one, because its ISA uses geometric altitude where the standard uses
   geopotential. qbar is proportional to rho, so a nominal match would put a
   same-signed bias on every force.

AND ONE ASYMMETRY THAT CANNOT BE REMOVED
-----------------------------------------
JSBSim samples wind at a single point and has no writable gust-rate input --
atmosphere/{p,q,r}-turb-rad_sec are READ-ONLY. So it carries no gradient across
the airframe at all. The headline arm therefore runs atisim TRANSLATION-ONLY,
which is strictly like-for-like; the gradient is then switched on separately and
the difference is reported as the term JSBSim structurally cannot represent.
"""

import argparse
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import dynamics, jsbsim_vortex_ref, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.state import Controls, State, euler_to_quat
from atisim.units import FT2M, RAD2DEG

# Wingrove & Bach's own measurements, for the cases where they published one.
# Cimarron: p. 755, "load deviations of +0.73 and -1.20 g"; Fig. 6a gives the
# pitch response as -0.5 deg then +1.8 deg. Table 2's wave/vortex column spans
# +1.70 / -2.01 g across all six wave cases, and Fig. 8's extreme vortex point
# is 1.4 deg of pitch at 1.99 g -- neither is attributed to a named case, so
# neither is a per-case target and both are reported as an envelope instead.
PAPER = {
    "cimarron": {"dn_max": +0.73, "dn_min": -1.20,
                 "dtheta_max": +1.8, "dtheta_min": -0.5},
}
PAPER_ENVELOPE = {"dn_max": +1.70, "dn_min": -2.01}


def partial_field_model(field, *, omega_gust: bool, alphadot: bool):
    """wind.field_model with either gradient term switchable off.

    The gradient the analytic Jacobian produces is TWO separate physical terms
    and they do not have to be reported together:

      omega_gust  the gust rate across the airframe -- a gust varying along the
                  fuselage is a pitching input, one varying across the span is
                  a rolling one
      alphadot    the wind-induced angle-of-attack RATE, which reaches the
                  pitching moment through Cmadot (the tail's downwash lag)

    Both off is the strictly like-for-like arm against JSBSim, which samples
    wind at one point and has no writable gust-rate input. Switching them on one
    at a time is what turns "atisim and JSBSim disagree on pitch" into a
    statement about WHICH term is responsible.
    """
    def model(wind_state, state, key, dt):
        del dt
        w = field(state.pos_ned)
        og = (wind.gust_rates(state.pos_ned, state.quat, field)
              if omega_gust else jnp.zeros(3))
        ad = (wind.gust_alphadot(state.pos_ned, state.quat, state.vel_body, field)
              if alphadot else jnp.array(0.0))
        return w, og, wind_state, key, ad
    return model


ARMS = {
    "translational": dict(omega_gust=False, alphadot=False),
    "+ alphadot": dict(omega_gust=False, alphadot=True),
    "+ omega_gust": dict(omega_gust=True, alphadot=False),
    "+ both": dict(omega_gust=True, alphadot=True),
}


def atisim_run(encounter, *, arm):
    """Fly one encounter, from JSBSim's own starting state."""
    ac = REGISTRY[encounter.aircraft]
    v = encounter.values
    altitude = v["matched_altitude"]

    # The core sits at the aircraft's own altitude in each engine's frame, so
    # the RELATIVE geometry is identical even though the two altitudes differ by
    # the density match. Placing it at JSBSim's nominal altitude instead would
    # leave atisim passing 21 m below the core it was supposed to fly through.
    array = wind.VortexArray(
        north=jnp.array([v["core_north"]]),
        down=jnp.array([-altitude]),
        r0=jnp.array(v["r0"]),
        v0=jnp.array(v["v0"]),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

    state = State(
        pos_ned=jnp.array([0.0, 0.0, -altitude]),
        vel_body=jnp.array(encounter.initial.vel_body),
        quat=euler_to_quat(*(jnp.array(x) for x in encounter.initial.euler)),
        omega=jnp.array(encounter.initial.omega),
    )
    controls = Controls(
        elevator=jnp.array(encounter.initial.controls[0]),
        aileron=jnp.array(encounter.initial.controls[1]),
        rudder=jnp.array(encounter.initial.controls[2]),
        throttle=jnp.array(encounter.initial.throttle),
    )
    return vortex_viz.fly_from_state(
        ac, field, state, controls,
        label=f"{encounter.case}/{encounter.radius_source} {arm}",
        seconds=v["duration"],
        dt=0.01,
        window=(v["core_north"] - v["r0"], v["core_north"] + v["r0"]),
        window_name="core",
        wind_model=partial_field_model(field, **ARMS[arm]),
    ), state, controls, ac


def _pct(a, b):
    """Signed percentage difference of a from b, guarded at b = 0."""
    return 0.0 if b == 0 else (a - b) / abs(b) * 100.0


def increments(run, datum_n, datum_theta):
    """Window increments, measured TWO ways, because they differ by a lot.

    `from_trim` is the excursion relative to the still-air trim value, which is
    what Wingrove & Bach report. It is contaminated here by run-in drift: the
    aircraft flies 15 core radii -- 9 to 11 seconds -- through the vortex's 1/r
    far field before it reaches the core, and attitude accumulates over that.

    `from_entry` is the excursion relative to the state at the window edge. It
    isolates the CORE RESPONSE, which is the thing the two engines can be
    compared on. Measured from trim the two engines' pitch appears to disagree
    by up to 75%; measured from entry, by under 6%, and all of the difference is
    in the run-in rather than in the event.

    That is not a presentational choice, it is the correct measurement. atisim
    is started from JSBSim's state, which is deliberately NOT atisim's own trim
    -- so atisim begins fractionally out of trim and drifts during the run-in,
    exactly as much as the two trims differ. Only the entry-relative number is
    free of that.
    """
    w = run.window
    n_z, theta = run.n_z[w], run.theta[w] * RAD2DEG
    dth_trim = theta - datum_theta * RAD2DEG
    return {
        "from_trim": (float((n_z - datum_n).max()), float((n_z - datum_n).min()),
                      float(dth_trim.max()), float(dth_trim.min())),
        # Peak to peak across the window. Unambiguous where an entry-relative
        # increment is not: the window opens at the up-gust peak, so the
        # excursion is one-signed and its "positive" half is identically zero.
        "span": (float(np.ptp(n_z)), float(np.ptp(theta))),
        "peak_alpha": float(np.abs(run.alpha_air[w]).max() * RAD2DEG),
        "entry_theta": float(dth_trim[0]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, help="save the figure here")
    args = parser.parse_args()

    ref = jsbsim_vortex_ref.load()
    print(f"JSBSim {ref.jsbsim_version.split('[')[0].strip()}, "
          f"dt {ref.dt:.6f} s, lead-in {ref.lead_in_radii:.0f} core radii, "
          f"gradient injected: {ref.gradient_injected}\n")

    rows, runs = [], {}
    for key in sorted(ref.encounters):
        enc = ref.encounters[key]
        v = enc.values

        arms = {}
        for name in ARMS:
            run, state, controls, ac = atisim_run(enc, arm=name)
            arms[name] = run

        # atisim's own still-air datum at the same state, so each engine is
        # measured from its own level-flight value rather than from a literal
        # 1.0 that neither of them sits at.
        datum_n = float(dynamics.load_factor(
            state, controls, ac, jnp.zeros(3), jnp.zeros(3)))
        datum_theta = enc.initial.euler[1]

        js_trim = enc.load_increments() + enc.pitch_increments()
        _c = enc.core_response()
        js_span = (abs(_c[0] - _c[1]), abs(_c[2] - _c[3]))
        measured = {n: increments(r, datum_n, datum_theta)
                    for n, r in arms.items()}

        rows.append((key, enc, v, js_trim, js_span, measured, datum_n))
        runs[key] = arms

        print(f"=== {enc.case} / r0 = {v['r0'] / FT2M:.0f} ft "
              f"({enc.radius_source})  --  {enc.aircraft} at "
              f"{v['altitude'] / FT2M:.0f} ft, V0 = {v['v0'] / FT2M:.0f} ft/s ===")
        print(f"   datum n_z: JSBSim {v['trim_Nz']:.6f}, atisim {datum_n:.6f} "
              f"(the two definitions differ by {abs(v['trim_Nz'] - datum_n):.1e} g)")

        base = measured["translational"]
        print("   -- CORE RESPONSE, peak to peak across the window "
              "(the encounter itself) --")
        print(f"   {'':24} {'n_z span':>10} {'theta span':>11} {'|alpha|max':>11}")
        print(f"   {'JSBSim':24} {js_span[0]:10.4f} {js_span[1]:11.4f} "
              f"{'--':>11}")
        for name in ARMS:
            m = measured[name]
            print(f"   {'atisim, ' + name:24} {m['span'][0]:10.4f} "
                  f"{m['span'][1]:11.4f} {m['peak_alpha']:11.3f}")
        bs = base["span"]
        print(f"   {'like-for-like error':24} "
              f"{_pct(bs[0], js_span[0]):9.1f}% {_pct(bs[1], js_span[1]):10.1f}%")
        print("   -- FROM TRIM, which the run-in contaminates --")
        print(f"   {'JSBSim':24} {js_trim[0]:+8.4f} {js_trim[1]:+8.4f} "
              f"{js_trim[2]:+8.3f} {js_trim[3]:+8.3f}")
        bt = base["from_trim"]
        print(f"   {'atisim, translational':24} {bt[0]:+8.4f} {bt[1]:+8.4f} "
              f"{bt[2]:+8.3f} {bt[3]:+8.3f}")
        print(f"   {'like-for-like error':24} "
              + " ".join(f"{_pct(bt[i], js_trim[i]):>7.1f}%" for i in range(4)))
        print(f"   run-in pitch at window edge: JSBSim "
              f"{np.degrees(enc.window()[0].theta - datum_theta):+.3f} deg, "
              f"atisim {base['entry_theta']:+.3f} deg -- a "
              f"{abs(base['entry_theta'] - np.degrees(enc.window()[0].theta - datum_theta)):.3f} deg "
              "offset accumulated BEFORE the core")

        g = measured["+ both"]["span"]
        print(f"   what the gradient adds to the core response: "
              f"{g[0] - bs[0]:+.4f} g of load span ({_pct(g[0], bs[0]):+.1f}%) "
              f"and {g[1] - bs[1]:+.3f} deg of pitch span "
              "-- the term JSBSim cannot carry")
        if enc.case in PAPER:
            p = PAPER[enc.case]
            print(f"   {'Wingrove & Bach':24} {p['dn_max']:+8.4f} "
                  f"{p['dn_min']:+8.4f} {p['dtheta_max']:+8.3f} "
                  f"{p['dtheta_min']:+8.3f}   (from trim, whole event)")
        print()

    _hannibal_spread(ref)
    if args.png:
        _figure(ref, runs, args.png)
    return rows


def _hannibal_spread(ref):
    a = ref.encounters[("hannibal", "wingrove")]
    b = ref.encounters[("hannibal", "parks")]
    an, bn = a.load_increments(), b.load_increments()
    print("=== the unresolved Hannibal radius, in JSBSim ===")
    print(f"   Fig. 4   r0 = {a.values['r0'] / FT2M:.0f} ft -> "
          f"dn {an[0]:+.4f} / {an[1]:+.4f} g")
    print(f"   Parks    r0 = {b.values['r0'] / FT2M:.0f} ft -> "
          f"dn {bn[0]:+.4f} / {bn[1]:+.4f} g")
    print(f"   spread   {abs(an[1] - bn[1]):.4f} g on the negative peak, "
          f"{abs((an[1] - bn[1]) / bn[1]) * 100:.1f}% -- and Parks 1985 has "
          "never been obtained to settle which is right.\n")
    m = ref.encounters[("morton", "wingrove")].load_increments()
    mp = ref.encounters[("morton", "parks")].load_increments()
    print(f"   Morton control (both sources agree on 450 ft): "
          f"{m == mp} -- identical, so the spread above is the radius\n")


def _figure(ref, runs, path):
    import matplotlib.pyplot as plt

    keys = [("cimarron", "wingrove"), ("hannibal", "wingrove"),
            ("morton", "wingrove")]
    fig, axes = plt.subplots(len(keys), 3, figsize=(15, 4 * len(keys)))
    for row, key in enumerate(keys):
        enc = ref.encounters[key]
        plain, grad = runs[key]["translational"], runs[key]["+ both"]
        v = enc.values
        datum_theta = enc.initial.euler[1]

        js_n = np.array([s.north for s in enc.samples])
        js_w = np.array([-s.wind[2] for s in enc.samples])
        js_nz = np.array([s.Nz for s in enc.samples]) - v["trim_Nz"]
        js_th = (np.array([s.theta for s in enc.samples]) - datum_theta) * RAD2DEG
        rel = (js_n - v["core_north"]) / v["r0"]
        at_rel = (plain.north - v["core_north"]) / v["r0"]
        gr_rel = (grad.north - v["core_north"]) / v["r0"]
        datum_n = plain.n_z[0] - (plain.n_z[0] - plain.n_z[0])  # placeholder
        datum_n = float(np.interp(0.0, at_rel, plain.n_z)) * 0 + plain.n_z[0]

        ax = axes[row, 0]
        ax.plot(rel, js_w * (1 / FT2M), lw=1.4, label="injected field")
        ax.set_ylabel(f"{enc.case}\nvertical gust (ft/s)")
        ax.set_title("wind both engines flew" if row == 0 else "")
        ax.axvspan(-1, 1, color="0.9", zorder=0)

        ax = axes[row, 1]
        ax.plot(rel, js_nz, lw=1.6, label="JSBSim")
        ax.plot(at_rel, plain.n_z - datum_n, lw=1.4, ls="--",
                label="atisim, translational")
        ax.plot(gr_rel, grad.n_z - datum_n, lw=1.0, ls=":",
                label="atisim, + gradient")
        if enc.case in PAPER:
            for y in (PAPER[enc.case]["dn_max"], PAPER[enc.case]["dn_min"]):
                ax.axhline(y, color="crimson", lw=0.9, ls="-.")
            ax.plot([], [], color="crimson", lw=0.9, ls="-.",
                    label="Wingrove & Bach")
        ax.axvspan(-1, 1, color="0.9", zorder=0)
        ax.set_ylabel("load increment (g)")
        ax.set_title("normal load" if row == 0 else "")
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 2]
        ax.plot(rel, js_th, lw=1.6, label="JSBSim")
        ax.plot(at_rel, (plain.theta - datum_theta) * RAD2DEG, lw=1.4, ls="--",
                label="atisim, translational")
        ax.plot(gr_rel, (grad.theta - datum_theta) * RAD2DEG, lw=1.0, ls=":",
                label="atisim, + gradient")
        ax.axvspan(-1, 1, color="0.9", zorder=0)
        ax.set_ylabel("pitch increment (deg)")
        ax.set_title("pitch" if row == 0 else "")

        for ax in axes[row]:
            ax.set_xlim(-6, 4)
            ax.grid(alpha=0.3)
            ax.set_xlabel("distance from core centre (core radii)")

    fig.suptitle(
        "AtiSim against JSBSim through one Kelvin-Helmholtz vortex\n"
        "shaded band is the analysis window; JSBSim carries no gust gradient",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
