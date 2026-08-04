"""Step 1-5 checkpoint: trim, hold, and validation against the source document.

Run: .venv/Scripts/python.exe scripts/checkpoint.py
"""

import jax
import jax.numpy as jnp
import numpy as np

import flightsim  # noqa: F401  -- enables x64
from flightsim import integrate, trim
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.dynamics import derivatives
from flightsim.state import Controls, State, euler_to_quat
from flightsim.units import RAD2DEG

NAME = "boeing747"
ac = REGISTRY[NAME]
V = CRUISE[NAME]["airspeed"]
H = CRUISE[NAME]["altitude"]


def longitudinal_modes(alpha, elevator, throttle):
    """Linearise about trim in [u, w, q, theta] and return the two modes.

    CR-2144 Table IX-5 tabulates the elevator-transfer-function denominator for
    each flight condition, which is exactly these two second-order modes. That
    makes them an independent check on the whole build-up: aero, dynamics,
    inertia, atmosphere and unit conversions all have to be right together.
    """
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = trim.trimmed_controls(jnp.array(elevator), jnp.array(throttle))

    def f(x):
        u, w, q, theta = x
        state = State(
            pos_ned=jnp.array([0.0, 0.0, -H]),
            vel_body=jnp.array([u, 0.0, w]),
            quat=euler_to_quat(jnp.array(0.0), theta, jnp.array(0.0)),
            omega=jnp.array([0.0, q, 0.0]),
        )
        d = derivatives(state, controls, ac, jnp.zeros(3), jnp.zeros(3))
        return jnp.array([d.vel_body[0], d.vel_body[2], d.omega[1], q])

    A = np.asarray(jax.jacfwd(f)(jnp.array([u0, w0, 0.0, alpha])))
    eig = np.linalg.eigvals(A)
    modes = []
    for lam in eig:
        if lam.imag > 1e-9:
            wn = abs(lam)
            modes.append((wn, -lam.real / wn))
    return sorted(modes)  # phugoid (low wn) first


print(f"=== {NAME}: CR-2144 flight condition 9, 40,000 ft, M 0.80 ===\n")

x, res = trim.trim(jnp.array(V), jnp.array(H), ac)
alpha, elevator, throttle = (float(v) for v in x)

print("TRIM")
print(f"  alpha           {alpha * RAD2DEG:+8.3f} deg     (source tabulates +4.60)")
print(f"  elevator        {elevator * RAD2DEG:+8.3f} deg")
print(f"  throttle        {throttle:8.4f}")
print(f"  residual norm   {float(jnp.linalg.norm(res)):8.2e}  [udot, wdot, qdot]")

dt, seconds = 0.02, 60.0
n = int(seconds / dt)
state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
controls = trim.trimmed_controls(jnp.array(elevator), jnp.array(throttle))
final, hist = integrate.rollout(
    integrate.init_sim(state, jax.random.PRNGKey(0)), controls, jnp.array(dt), ac, n
)
alt = -np.asarray(hist.pos_ned)[:, 2]
spd = np.linalg.norm(np.asarray(hist.vel_body), axis=1)
north = np.asarray(hist.pos_ned)[:, 0]

print(f"\nHOLD  ({seconds:.0f} s, fixed controls, dt = {dt} s)")
print(f"  altitude drift  {alt[-1] - H:+8.4f} m   (max excursion {np.abs(alt - H).max():.4f})")
print(f"  airspeed drift  {spd[-1] - V:+8.4f} m/s")
print(f"  distance north  {north[-1] / 1000:8.3f} km  (expected {V * seconds / 1000:.3f})")
print(f"  quat norm error {np.abs(np.linalg.norm(np.asarray(hist.quat), axis=1) - 1).max():8.2e}")

print("\nLONGITUDINAL MODES vs CR-2144 Table IX-5 (FC 9 denominator)")
modes = longitudinal_modes(alpha, elevator, throttle)
reference = [("phugoid", 0.0673, 0.0489), ("short period", 0.964, 0.387)]
for (wn, zeta), (label, wn_ref, zeta_ref) in zip(modes, reference):
    print(
        f"  {label:13s} wn {wn:6.4f} vs {wn_ref:6.4f} rad/s"
        f"    zeta {zeta:+7.4f} vs {zeta_ref:+6.4f}"
    )
