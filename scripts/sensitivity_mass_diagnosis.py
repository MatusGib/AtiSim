"""What session 29's `mass` elasticities actually measure.

WHY THIS EXISTS. Session 29 ranks `mass` second on the headline load (-0.649) and
first on cruise phugoid damping (+1.728), and PROJECT.md section 0 uses the pair
`CLa` +0.692 / `mass` -0.649 to price a DC-10 acquisition. Both screens move
`ac.mass` alone, one at a time, with the trim re-solved. This script separates
four things that one-at-a-time move could be carrying, each by one controlled
variant, all by central difference on the SHIPPED paths (`sensitivity.load_history`
is not used, so the check is independent of the AD machinery):

  mirror   -- the gust load goes as CLa q S / W, so E[mass] ~ -E[CLa] (one lever)
  inertia  -- `inertia` is a COUPLED field the screen cannot move, so `mass` was
              moved with the pitch inertia held: ballast at the CG, not a heavier
              aeroplane. Variant: inertia and inertia_inv scaled with it
  path     -- flown with fixed controls the 747 climbs ~150 m through Mehta's
              field before cores 3-4; does mass change that climb? Variant: the
              field evaluated at the nominal altitude whatever the aircraft's own
  Korn     -- mass raises trim CL, which lowers the Korn M_crit, which steepens
              the DECLARED wave-drag rise. Variant: kappa_airfoil x1.25, which
              makes wave drag identically zero at this Mach (PROJECT.md S3)

and one question the screen could not ask: `_boeing_747` DERIVES CLa, CL0, e and
CD0 from CR-2144's weight, so "the printed weight is wrong" is a different
experiment from "the aeroplane is heavier". Variant: the builder re-run with W
scaled, so every derived coefficient moves with it.

On a tree that declares CR-2144's speed derivatives (session 30), CD_M is stored
NET of the Korn slope at the construction CL, so a mass change also changes the
TOTAL drag Mach slope. The extra variant there re-nets CD_M at the perturbed trim.

Run: PYTHONPATH=<abs worktree root> python scripts/sensitivity_mass_diagnosis.py
"""

import inspect

import jax.numpy as jnp
import numpy as np

import atisim
from atisim import aero, aircraft, trim, validation, vortex_viz, wind
from atisim.aircraft import CRUISE, REGISTRY
from atisim.atmosphere import speed_of_sound

RECORDED_PEAK_TO_PEAK = 2.70
H_REL = 0.01

AC = REGISTRY["boeing747"]
V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
DECLARES_MACH_SEAM = float(getattr(AC, "mach_deriv_ref", -1.0)) >= 0.0


def scaled(ac, field, f):
    return ac._replace(**{field: jnp.array(float(getattr(ac, field)) * f)})


def with_inertia(ac, f):
    return ac._replace(mass=jnp.array(float(ac.mass) * f),
                       inertia=ac.inertia * f, inertia_inv=ac.inertia_inv / f)


def no_wave_drag(ac):
    return scaled(ac, "kappa_airfoil", 1.25)


def rebuilt_with_weight(f):
    """`_boeing_747` re-run with CR-2144's W scaled -- every derived field moves."""
    src = inspect.getsource(aircraft._boeing_747)
    if src.count("W = 636636.0") != 1:
        raise RuntimeError("the builder's weight literal moved; update this script")
    ns = dict(vars(aircraft))
    exec(src.replace("W = 636636.0", f"W = 636636.0 * {f!r}"), ns)
    return ns["_boeing_747"]()


def _trim_cl(ac):
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    return float(ac.CL0) + float(ac.CLa) * float(x[0])


def renetted_mass(ac, f):
    """mass x f with CD_M re-netted so the TOTAL drag Mach slope stays as declared."""
    m0 = float(ac.mach_deriv_ref)

    def korn_slope(cl):
        m_crit = float(aero.drag_divergence_mach(cl, ac)) - aero._MDD_OFFSET
        return 80.0 * max(m0 - m_crit, 0.0) ** 3

    total = float(ac.CD_M) + korn_slope(_trim_cl(ac))
    heavier = scaled(ac, "mass", f)
    return heavier._replace(CD_M=jnp.array(total - korn_slope(_trim_cl(heavier))))


def modes(ac):
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    al, de, th = (float(v) for v in x)
    (pw, pz), (sw, sz) = validation.longitudinal_modes(ac, al, de, th, V, H)
    return np.array([pw, pz, sw, sz])


def central(q, build, h=H_REL):
    return (q(build(1 + h)) - q(build(1 - h))) / (2 * h) / q(build(1.0))


def section_modes():
    print("\n" + "=" * 78)
    print("A. CRUISE MODES (40,000 ft, 774 ft/s) -- elasticity, central difference")
    print("=" * 78)
    cl = _trim_cl(AC)
    mach = V / float(speed_of_sound(H))
    m_crit = float(aero.drag_divergence_mach(cl, AC)) - aero._MDD_OFFSET
    print(f"  trim CL {cl:.4f}, M {mach:.4f}, M_crit {m_crit:.4f}; Korn's own drag Mach"
          f" slope there: {80.0 * max(mach - m_crit, 0.0) ** 3:.4f} per Mach")
    print(f"\n  {'variant':34s} {'ph wn':>7s} {'ph zeta':>8s} {'sp wn':>7s} {'sp zeta':>8s}")
    rows = [
        ("mass (session 29's move)", lambda f: scaled(AC, "mass", f)),
        ("mass + inertia", lambda f: with_inertia(AC, f)),
        ("CLa", lambda f: scaled(AC, "CLa", f)),
        ("printed W, builder re-run", rebuilt_with_weight),
    ]
    if DECLARES_MACH_SEAM:
        rows.append(("mass, CD_M re-netted", lambda f: renetted_mass(AC, f)))
    else:
        rows += [("mass, wave drag off", lambda f: scaled(no_wave_drag(AC), "mass", f)),
                 ("mass + inertia, wave drag off", lambda f: with_inertia(no_wave_drag(AC), f))]
    for name, build in rows:
        e = central(modes, build)
        print(f"  {name:34s} " + " ".join(f"{v:+8.3f}" for v in e))


def section_load():
    print("\n" + "=" * 78)
    print("B. HEADLINE LOAD -- Mehta five-core array, 37,000 ft, dt 0.01")
    print("=" * 78)
    Hm = wind.MEHTA_HANNIBAL_ALTITUDE
    array = wind.mehta_hannibal_array(Hm)
    r0 = float(array.r0)
    x0, x1 = float(array.north.min()), float(array.north.max())
    start = x0 - 12.0 * r0
    seconds = (x1 + 12.0 * r0 - start) / V
    window = (x0 - 2.0 * r0, x1 + 2.0 * r0)

    def flown(p):
        return wind.vortex_wind(p, array)

    def replayed(p):
        return wind.vortex_wind(jnp.asarray(p).at[2].set(-Hm), array)

    def peak_to_peak(field):
        def q(ac):
            enc = vortex_viz.fly_in_moving_air(
                ac, field, V, Hm, label="mass diagnosis", start_north=start,
                seconds=seconds, dt=0.01, window=window, window_name="array +/- 2 r0")
            return float(enc.n_z[enc.window].max() - enc.n_z[enc.window].min())
        return q

    for form, field in (("as flown", flown), ("replayed on the nominal altitude", replayed)):
        q = peak_to_peak(field)
        q0 = q(AC)
        print(f"\n  [{form}] base {q0:.6f} g = {100 * q0 / RECORDED_PEAK_TO_PEAK:.2f}% of the record")
        rows = [
            ("mass (session 29's move)", lambda f: scaled(AC, "mass", f)),
            ("CLa", lambda f: scaled(AC, "CLa", f)),
            ("mass + inertia", lambda f: with_inertia(AC, f)),
            ("printed W, builder re-run", rebuilt_with_weight),
        ]
        if not DECLARES_MACH_SEAM:
            rows.append(("mass, wave drag off", lambda f: scaled(no_wave_drag(AC), "mass", f)))
        # TWO STEP SIZES, because a peak-to-peak of a sampled signal can move by
        # one sample under a 1% change and read as a different elasticity. On
        # session 30's replayed headline that happens at +/-1% for CLa (+0.497
        # against +0.740 at 0.25%); where the two columns disagree, neither is
        # the tangent.
        for name, build in rows:
            e = [central(q, build, h) for h in (0.0025, H_REL)]
            flag = "" if abs(e[0] - e[1]) < 0.02 else "   <- step-size dependent"
            print(f"    E[{name:28s}] h 0.25%: {e[0]:+.3f}   h 1%: {e[1]:+.3f}{flag}")


def main():
    print(f"atisim imported from: {atisim.__file__}")
    print("PROJECT.md section 10: if that is not the tree you edited, stop here.")
    print(f"boeing747 declares the Mach seam: {DECLARES_MACH_SEAM}")
    section_modes()
    section_load()


if __name__ == "__main__":
    main()
