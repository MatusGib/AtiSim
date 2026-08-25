"""Why the vortex comparison's two large errors are large.

Imports no jsbsim; reads the frozen reference like scripts/vortex_compare.py.

Run: .venv/Scripts/python.exe scripts/vortex_diagnose.py

The comparison leaves two numbers that look bad next to a core response the two
engines agree on to a few percent:

  A. Measured FROM TRIM, the pitch excursions disagree by 15-54%.
  B. Against Wingrove & Bach's DFDR, both engines under-predict load by 30-43%.

These are not the same kind of error and do not have the same cause. A is
between the two engines and is an artefact of where the datum was put. B is
common to both engines, so by construction it cannot be a solver difference at
all -- it is in the inputs they share. Each experiment below isolates one
candidate and reports what it is worth in the units of the discrepancy.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import dynamics, jsbsim_vortex_ref, trim, vortex_viz, wind
from atisim.aircraft import REGISTRY
from atisim.atmosphere import density
from atisim.state import Controls, State, euler_to_quat
from atisim.units import FT2M, RAD2DEG

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vortex_compare import atisim_run, partial_field_model  # noqa: E402

CASES = [("cimarron", "wingrove"), ("hannibal", "wingrove"), ("morton", "wingrove")]

# Wingrove & Bach p. 755, the only case with a published per-case load pair.
CIMARRON_DFDR = {"dn_plus": +0.73, "dn_minus": -1.20}


def _state_and_controls(enc, altitude):
    return (
        State(
            pos_ned=jnp.array([0.0, 0.0, -altitude]),
            vel_body=jnp.array(enc.initial.vel_body),
            quat=euler_to_quat(*(jnp.array(x) for x in enc.initial.euler)),
            omega=jnp.array(enc.initial.omega),
        ),
        Controls(
            elevator=jnp.array(enc.initial.controls[0]),
            aileron=jnp.array(enc.initial.controls[1]),
            rudder=jnp.array(enc.initial.controls[2]),
            throttle=jnp.array(enc.initial.throttle),
        ),
    )


def experiment_a1(ref):
    """Fly the SAME start state in STILL AIR. Any pitch that appears is trim.

    If atisim drifts nose-down in still air by about as much as it does through
    the far field, then the run-in offset is not the vortex at all -- it is
    atisim settling out of a trim that is not its own, and the vortex is
    incidental to it.
    """
    print("=" * 78)
    print("A1  STILL AIR, from JSBSim's trim state, for the run-in duration")
    print("=" * 78)
    print("    The vortex is switched OFF. Anything here is trim mismatch.\n")
    print(f"    {'case':12} {'run-in s':>9} {'still-air d-theta':>14} "
          f"{'in-vortex d-theta':>14} {'explained':>10}")
    out = {}
    for key in CASES:
        enc = ref.encounters[key]
        altitude = enc.values["matched_altitude"]
        state, controls = _state_and_controls(enc, altitude)
        ac = REGISTRY[enc.aircraft]
        edge = enc.values["core_north"] - enc.values["r0"]
        t_edge = next(s.t for s in enc.samples if s.north >= edge)

        zero = lambda p: jnp.zeros(3)  # noqa: E731
        still = vortex_viz.fly_from_state(
            ac, zero, state, controls, label="still", seconds=t_edge, dt=0.01,
            window=(-1e12, 1e12), window_name="all",
            wind_model=partial_field_model(zero, omega_gust=False, alphadot=False),
        )
        drift = float((still.theta[-1] - enc.initial.euler[1]) * RAD2DEG)

        # What the same run showed with the vortex on, at the same moment.
        run, *_ = atisim_run(enc, arm="translational")
        i = int(np.argmin(np.abs(run.north - edge)))
        in_vortex = float((run.theta[i] - enc.initial.euler[1]) * RAD2DEG)
        js = float(np.degrees(
            next(s.theta for s in enc.samples if s.north >= edge)
            - enc.initial.euler[1]))
        gap = in_vortex - js
        out[key] = (drift, gap)
        print(f"    {key[0]:12} {t_edge:9.2f} {drift:+14.3f} {in_vortex:+14.3f} "
              f"{drift / gap * 100 if gap else float('nan'):9.0f}%")
    print("\n    'explained' is the still-air drift as a fraction of the atisim-"
          "minus-JSBSim\n    offset at the window edge. Near 100% means the "
          "run-in gap is trim, not vortex.\n")
    return out


def experiment_a2(ref):
    """Start atisim from ITS OWN trim instead. Does the from-trim error collapse?

    This is the direct test. If the 15-54% pitch error is the shared-start
    compromise, then removing the compromise removes the error -- at the cost of
    the two engines no longer starting at the same angle of attack.
    """
    print("=" * 78)
    print("A2  FROM ATISIM'S OWN TRIM, through the same vortex")
    print("=" * 78)
    print("    Trades the identical start state for an in-trim start.\n")
    print(f"    {'case':12} {'JSBSim d-theta-':>11} {'shared start':>13} {'err':>7} "
          f"{'own trim':>10} {'err':>7}")
    for key in CASES:
        enc = ref.encounters[key]
        v = enc.values
        ac = REGISTRY[enc.aircraft]
        altitude = v["matched_altitude"]
        array = wind.VortexArray(
            north=jnp.array([v["core_north"]]), down=jnp.array([-altitude]),
            r0=jnp.array(v["r0"]), v0=jnp.array(v["v0"]),
        )
        field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

        x, _ = trim.trim(jnp.array(v["airspeed"]), jnp.array(altitude), ac)
        own = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(v["airspeed"]),
                                 jnp.array(altitude))
        own = own._replace(pos_ned=jnp.array([0.0, 0.0, -altitude]))
        own_run = vortex_viz.fly_from_state(
            ac, field, own, trim.trimmed_controls(x[1], x[2]),
            label="own-trim", seconds=v["duration"], dt=0.01,
            window=(v["core_north"] - v["r0"], v["core_north"] + v["r0"]),
            window_name="core",
            wind_model=partial_field_model(field, omega_gust=False, alphadot=False),
        )
        shared, *_ = atisim_run(enc, arm="translational")

        js_min = enc.pitch_increments()[1]
        sh_min = float(((shared.theta[shared.window] - enc.initial.euler[1])
                        * RAD2DEG).min())
        # Own-trim run measures from ITS OWN trim attitude, which is theta = alpha.
        own_theta0 = float(x[0])
        ow_min = float(((own_run.theta[own_run.window] - own_theta0) * RAD2DEG).min())
        print(f"    {key[0]:12} {js_min:+11.3f} {sh_min:+13.3f} "
              f"{(sh_min - js_min) / abs(js_min) * 100:+6.0f}% {ow_min:+10.3f} "
              f"{(ow_min - js_min) / abs(js_min) * 100:+6.0f}%")
    print()


def experiment_b(ref):
    """What would it take to reach the DFDR's load? Vary the one big lever.

    Both engines under-predict Cimarron's recorded load together, so the cause
    is in what they share. The largest single shared input is the lift-curve
    slope: JSBSim's 737 and B747 both carry CLalpha = 4.348 /rad, from one
    Aeromatic table, and gust load is very nearly proportional to it.
    """
    print("=" * 78)
    print("B   CIMARRON AGAINST THE DFDR: what closes a 30-43% gap?")
    print("=" * 78)
    enc = ref.encounters[("cimarron", "wingrove")]
    v = enc.values
    ac = REGISTRY[enc.aircraft]
    altitude = v["matched_altitude"]
    state, controls = _state_and_controls(enc, altitude)
    datum_n = float(dynamics.load_factor(state, controls, ac, jnp.zeros(3),
                                         jnp.zeros(3)))

    array = wind.VortexArray(
        north=jnp.array([v["core_north"]]), down=jnp.array([-altitude]),
        r0=jnp.array(v["r0"]), v0=jnp.array(v["v0"]),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

    # The quasi-steady expectation, so the mechanism is visible before the sim.
    V = v["airspeed"]
    d_alpha = np.arctan(v["v0"] / V)
    q = 0.5 * float(density(jnp.array(altitude))) * V**2
    CL_trim = float(ac.mass) * 9.80665 / (q * float(ac.S))
    print(f"    peak gust {v['v0'] / FT2M:.0f} ft/s at V = {V / FT2M:.0f} ft/s "
          f"-> d-alpha = {np.degrees(d_alpha):.2f} deg")
    print(f"    CL_trim = {CL_trim:.4f}, CLa = {float(ac.CLa):.3f} /rad")
    print(f"    quasi-steady dn = CLa*d-alpha/CL_trim = "
          f"{float(ac.CLa) * d_alpha / CL_trim:+.3f} g "
          "(before the aircraft responds)\n")

    print(f"    {'CLa /rad':>10} {'ratio':>7} {'dn+':>8} {'dn-':>8} "
          f"{'vs DFDR -1.20':>14}")
    base = float(ac.CLa)
    for scale in (1.0, 1.2, 1.38, 1.5, 1.6):
        # Scale the TABLE, not the CLa field. This entry carries JSBSim's own
        # CL(alpha) table, and aero.coefficients evaluates the table whenever it
        # is non-empty -- so _replace(CLa=...) alone changes nothing at all, and
        # the first version of this experiment silently reported one number five
        # times. Scaling about the zero-incidence point is a pure lift-slope
        # change: the intercept CL(0) = 0.20 is held and every other point moves.
        table = np.asarray(ac.CL_table_CL)
        intercept = float(table[1])            # CL at alpha = 0
        scaled = ac._replace(
            CLa=jnp.array(base * scale),
            CL_table_CL=jnp.array(intercept + scale * (table - intercept)),
        )
        run = vortex_viz.fly_from_state(
            scaled, field, state, controls, label=f"CLa x{scale}",
            seconds=v["duration"], dt=0.01,
            window=(v["core_north"] - v["r0"], v["core_north"] + v["r0"]),
            window_name="core",
            wind_model=partial_field_model(field, omega_gust=False, alphadot=False),
        )
        nz = run.n_z - datum_n
        print(f"    {base * scale:10.3f} {scale:7.2f} {nz.max():+8.3f} "
              f"{nz.min():+8.3f} {nz.min() / CIMARRON_DFDR['dn_minus'] * 100:13.0f}%")

    print("\n    Prandtl-Glauert at M 0.78 divides by sqrt(1-M^2) = 0.626, i.e. an")
    print("    incompressible 4.348 becomes 6.95 /rad in compressible flow -- a")
    print("    factor of 1.60. JSBSim's table carries no Mach correction on CLa.\n")

    # And the other lever the paper itself points at: the model smooths the data.
    print("    The other shared input, from Fig. 4's own overlay: the Cimarron")
    print("    DATA spikes past the fitted model. Scaling the core strength:\n")
    print(f"    {'V0 ft/s':>9} {'dn+':>8} {'dn-':>8} {'vs DFDR -1.20':>14}")
    for v0_ft in (50.0, 60.0, 70.0):
        arr2 = wind.VortexArray(
            north=jnp.array([v["core_north"]]), down=jnp.array([-altitude]),
            r0=jnp.array(v["r0"]), v0=jnp.array(v0_ft * FT2M),
        )
        f2 = lambda p: wind.vortex_wind(p, arr2)  # noqa: E731
        run = vortex_viz.fly_from_state(
            ac, f2, state, controls, label=f"V0 {v0_ft}",
            seconds=v["duration"], dt=0.01,
            window=(v["core_north"] - v["r0"], v["core_north"] + v["r0"]),
            window_name="core",
            wind_model=partial_field_model(f2, omega_gust=False, alphadot=False),
        )
        nz = run.n_z - datum_n
        print(f"    {v0_ft:9.0f} {nz.max():+8.3f} {nz.min():+8.3f} "
              f"{nz.min() / CIMARRON_DFDR['dn_minus'] * 100:13.0f}%")
    print()


def experiment_c(ref):
    """Fly Cimarron's vortex with a HEAVIER airframe.

    The quasi-steady gust load -- what the aircraft would feel if it could not
    respond at all -- is +0.725 g, essentially the DFDR's +0.73. The simulated
    737 reports +0.417, so it sheds 42% of that by pitching away and climbing
    during the traverse. A heavier aircraft with more pitch inertia sheds less.

    Parks identifies the Hannibal and Morton aircraft as DC-10s; Wingrove & Bach
    name none, but every case is a widebody-era airliner on a long route. A 737
    is the wrong mass class for that by a factor of four, and this is what that
    is worth.

    CAVEAT: 33,000 ft is below boeing747_jsbsim's declared band of 35,000-41,000
    ft, so this is a sensitivity and not a result. It is reported as a direction
    and a rough size, not as a number to quote.
    """
    print("=" * 78)
    print("C   THE SAME VORTEX, A HEAVIER AIRCRAFT")
    print("=" * 78)
    print("    Quasi-steady load is what an aircraft that could not respond")
    print("    would feel. Responding sheds some of it; more inertia sheds less.")
    print("    *** 33,000 ft is below the 747 entry's band -- direction, not a")
    print("    number to quote. ***\n")
    enc = ref.encounters[("cimarron", "wingrove")]
    v = enc.values
    altitude = v["matched_altitude"]
    array = wind.VortexArray(
        north=jnp.array([v["core_north"]]), down=jnp.array([-altitude]),
        r0=jnp.array(v["r0"]), v0=jnp.array(v["v0"]),
    )
    field = lambda p: wind.vortex_wind(p, array)  # noqa: E731

    print(f"    {'airframe':20} {'mass t':>8} {'Iyy':>10} {'dn+':>8} {'dn-':>8} "
          f"{'vs DFDR':>9}")
    for name in ("boeing737", "boeing747_jsbsim", "boeing747"):
        ac = REGISTRY[name]
        x, _ = trim.trim(jnp.array(v["airspeed"]), jnp.array(altitude), ac)
        st = trim.trimmed_state(jnp.array(float(x[0])), jnp.array(v["airspeed"]),
                                jnp.array(altitude))
        st = st._replace(pos_ned=jnp.array([0.0, 0.0, -altitude]))
        ctl = trim.trimmed_controls(x[1], x[2])
        run = vortex_viz.fly_from_state(
            ac, field, st, ctl, label=name, seconds=v["duration"], dt=0.01,
            window=(v["core_north"] - v["r0"], v["core_north"] + v["r0"]),
            window_name="core",
            wind_model=partial_field_model(field, omega_gust=False, alphadot=False),
        )
        datum = float(dynamics.load_factor(st, ctl, ac, jnp.zeros(3), jnp.zeros(3)))
        nz = run.n_z[run.window] - datum
        print(f"    {name:20} {float(ac.mass) / 1000:8.1f} "
              f"{float(ac.inertia[1, 1]):10.2e} {nz.max():+8.3f} {nz.min():+8.3f} "
              f"{nz.min() / CIMARRON_DFDR['dn_minus'] * 100:8.0f}%")
    print()


def main():
    ref = jsbsim_vortex_ref.load()
    experiment_a1(ref)
    experiment_a2(ref)
    experiment_b(ref)
    experiment_c(ref)


if __name__ == "__main__":
    main()
