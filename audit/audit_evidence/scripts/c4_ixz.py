"""Agent C, item 2 (Ixz half): the product-of-inertia sign, end to end.

Three independent checks, all on the 747 cruise set -- the only aircraft in the
registry with Ixz != 0, so the only asymmetric case available.

  (i)   the tensor itself: symmetric, positive definite, off-diagonal = -Ixz
  (ii)  an applied pure rolling moment must produce a yaw acceleration of the
        SAME sign when Ixz > 0
  (iii) THE DECISIVE ONE.  CR-2144 Table IX-8 tabulates PRIMED lateral
        derivatives -- values that already contain the Ixz coupling.  aircraft.py
        strips the coupling out with `_unprime`, and the plant is supposed to put
        it back through I^-1.  So driving the plant with a pure p, beta, da or dr
        must REPRODUCE the tabulated primed number.  If the tensor negated Ixz
        the wrong way, or `_unprime` inverted the wrong system, this round trip
        breaks -- and it breaks asymmetrically, differently for L and for N.
"""
import jax.numpy as jnp
import numpy as np

from flightsim import aero
from flightsim.aircraft import REGISTRY, inertia_tensor, _unprime
from flightsim.state import Controls
from flightsim.units import FT2M, SLUG_FT2_TO_KG_M2

fails = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {detail}")
    if not ok:
        fails.append(name)


ac = REGISTRY["boeing747"]
I = np.asarray(ac.inertia)
Ixz_slugft2 = 970056.0  # CR-2144 Table IX-3, flight condition 9
Ixz_si = Ixz_slugft2 * SLUG_FT2_TO_KG_M2

print("=" * 78)
print("(i) the tensor")
print("=" * 78)
print(I)
check("tensor is symmetric", np.abs(I - I.T).max() < 1e-6, f"{np.abs(I-I.T).max():.2e}")
check("tensor is positive definite", np.all(np.linalg.eigvalsh(I) > 0),
      f"eigs={np.linalg.eigvalsh(I)}")
check("off-diagonal xz element == -Ixz (tabulated Ixz is POSITIVE)",
      abs(I[0, 2] + Ixz_si) < 1e-3 and Ixz_si > 0,
      f"I[0,2]={I[0,2]:.1f}, -Ixz={-Ixz_si:.1f}")
check("Ixy and Iyz are exactly zero (xz plane of symmetry)",
      I[0, 1] == 0.0 and I[1, 2] == 0.0)

print()
print("=" * 78)
print("(ii) roll/yaw inertial coupling direction")
print("=" * 78)
Iinv = np.asarray(ac.inertia_inv)
L_applied = np.array([1.0e7, 0.0, 0.0])  # pure right-rolling moment, N.m
om_dot = Iinv @ L_applied
print(f"   pure L>0 -> omega_dot = {om_dot}")
check("Ixz>0: a pure right-rolling moment yaws the nose RIGHT (rdot>0)",
      om_dot[2] > 0, f"rdot={om_dot[2]:+.3e}")
N_applied = np.array([0.0, 0.0, 1.0e7])
om_dot_n = Iinv @ N_applied
print(f"   pure N>0 -> omega_dot = {om_dot_n}")
check("Ixz>0: a pure nose-right yawing moment rolls RIGHT (pdot>0)",
      om_dot_n[0] > 0, f"pdot={om_dot_n[0]:+.3e}")
# Analytic cross-check, written out here rather than taken from the module.
det = I[0, 0] * I[2, 2] - Ixz_si**2
print(f"   analytic rdot/L = Ixz/(Ix Iz - Ixz^2) = {Ixz_si / det:.6e}, "
      f"model = {om_dot[2] / 1.0e7:.6e}")
check("rdot/L matches Ixz/(Ix Iz - Ixz^2) analytically",
      abs(om_dot[2] / 1.0e7 - Ixz_si / det) < 1e-18)

# Gyroscopic term for a pure roll rate: -omega x (I omega) = (0, -Ixz p^2, 0)
p = 0.2
om = np.array([p, 0.0, 0.0])
gyro = -np.cross(om, I @ om)
print(f"   pure roll rate p={p}: -omega x (I omega) = {gyro}  "
      f"(analytic (0, -Ixz p^2, 0) = (0, {-Ixz_si * p * p:.1f}, 0))")
check("rolling produces a nose-DOWN inertial pitching moment for Ixz>0",
      abs(gyro[1] + Ixz_si * p * p) < 1e-6 and gyro[1] < 0)

print()
print("=" * 78)
print("(iii) round trip against CR-2144 Table IX-8 PRIMED derivatives")
print("=" * 78)
# The exact condition the table was linearised at, in imperial, as aircraft.py
# reads it.  Reproduced here from the source numbers, not imported.
S_ft2, b_ft, U0_fts, qbar_psf = 5500.0, 195.68, 774.0, 177.0
V_si = U0_fts * FT2M
S_si = S_ft2 * FT2M**2
qbar_si = qbar_psf * 4.4482216152605 / FT2M**2   # lbf/ft^2 -> Pa
rho_si = 2.0 * qbar_si / V_si**2
print(f"   V={V_si:.4f} m/s, qbar={qbar_si:.2f} Pa, implied rho={rho_si:.6f} kg/m^3")

TABLE = {  # CR-2144 Table IX-8, primed:  (L'_x, N'_x)
    "beta": (-3.05, 0.598),
    "p":    (-0.465, -0.0316),
    "da":   (0.143, 0.00775),
    "dr":   (0.153, -0.475),
}
ZERO3 = jnp.zeros(3)
NOCTL = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
a_sound = jnp.array(1000.0)  # kept off the wave-drag knee; moments do not use it


def omega_dot_from(vel_rel, omega_rel, ctl):
    _, M = aero.aero_forces_moments(vel_rel, omega_rel, ctl, ac,
                                    jnp.array(rho_si), a_sound)
    return np.asarray(ac.inertia_inv) @ np.asarray(M)


eps = 1e-4
base_v = jnp.array([V_si, 0.0, 0.0])

cases = {}
# beta: perturb the body-y relative velocity
vb = jnp.array([V_si * np.cos(eps), V_si * np.sin(eps), 0.0])
cases["beta"] = (omega_dot_from(vb, ZERO3, NOCTL) - omega_dot_from(base_v, ZERO3, NOCTL)) / eps
# p
cases["p"] = (omega_dot_from(base_v, jnp.array([eps, 0.0, 0.0]), NOCTL)
              - omega_dot_from(base_v, ZERO3, NOCTL)) / eps
# da
cases["da"] = (omega_dot_from(base_v, ZERO3, NOCTL._replace(aileron=jnp.array(eps)))
               - omega_dot_from(base_v, ZERO3, NOCTL)) / eps
# dr
cases["dr"] = (omega_dot_from(base_v, ZERO3, NOCTL._replace(rudder=jnp.array(eps)))
               - omega_dot_from(base_v, ZERO3, NOCTL)) / eps

print(f"   {'input':6s} {'d(pdot)/dx':>13s} {'L-primed':>11s} {'err%':>7s}   "
      f"{'d(rdot)/dx':>13s} {'N-primed':>11s} {'err%':>7s}")
for k, (Lp_ref, Np_ref) in TABLE.items():
    od = cases[k]
    eL = 100.0 * (od[0] - Lp_ref) / abs(Lp_ref)
    eN = 100.0 * (od[2] - Np_ref) / abs(Np_ref)
    print(f"   {k:6s} {od[0]:13.6f} {Lp_ref:11.5f} {eL:+7.3f}   "
          f"{od[2]:13.6f} {Np_ref:11.5f} {eN:+7.3f}")
    check(f"   L' round trip for {k}", abs(eL) < 0.5, f"{eL:+.3f}%")
    check(f"   N' round trip for {k}", abs(eN) < 0.5, f"{eN:+.3f}%")

print()
print("   CONTROL: flip the sign of Ixz in the tensor and repeat -- the round")
print("   trip must BREAK, otherwise the test above proves nothing.")
Ix = 1.82e7 * SLUG_FT2_TO_KG_M2
Iy = 3.31e7 * SLUG_FT2_TO_KG_M2
Iz = 4.97e7 * SLUG_FT2_TO_KG_M2
bad = inertia_tensor(Ix, Iy, Iz, -Ixz_si)   # WRONG sign, deliberately
bad_inv = np.linalg.inv(np.asarray(bad))
ac_bad = ac._replace(inertia=bad, inertia_inv=jnp.array(bad_inv))


def omega_dot_bad(vel_rel, omega_rel, ctl):
    _, M = aero.aero_forces_moments(vel_rel, omega_rel, ctl, ac_bad,
                                    jnp.array(rho_si), a_sound)
    return bad_inv @ np.asarray(M)


od_bad = (omega_dot_bad(base_v, jnp.array([eps, 0.0, 0.0]), NOCTL)
          - omega_dot_bad(base_v, ZERO3, NOCTL)) / eps
eN_bad = 100.0 * (od_bad[2] - TABLE["p"][1]) / abs(TABLE["p"][1])
print(f"   with -Ixz: d(rdot)/dp = {od_bad[2]:.6f} vs N'_p = {TABLE['p'][1]}  "
      f"({eN_bad:+.1f}%)")
check("negated-Ixz control is REJECTED (test has teeth)", abs(eN_bad) > 10.0,
      f"{eN_bad:+.1f}%")

print()
print("=" * 78)
print("(iv) _unprime is the exact inverse of the primed relation")
print("=" * 78)
# Written out from the definition in the docstring, independently.
for k, (Lpr, Npr) in TABLE.items():
    L, N = [float(v) for v in _unprime(Lpr, Npr, Ix, Iz, Ixz_si)]
    d = 1.0 - Ixz_si**2 / (Ix * Iz)
    Lpr_back = (L + (Ixz_si / Ix) * N) / d
    Npr_back = (N + (Ixz_si / Iz) * L) / d
    ok = abs(Lpr_back - Lpr) < 1e-12 * max(1, abs(Lpr)) and \
         abs(Npr_back - Npr) < 1e-12 * max(1, abs(Npr))
    print(f"   {k:6s} raw=({L:+.6f},{N:+.6f})  ->  primed=({Lpr_back:+.6f},"
          f"{Npr_back:+.6f})  table=({Lpr:+.6f},{Npr:+.6f})")
    check(f"   _unprime inverts the primed relation for {k}", ok)
# _unprime must be inertia-unit-invariant (only the ratios Ixz/Ix, Ixz/Iz enter)
L1 = _unprime(-0.465, -0.0316, Ix, Iz, Ixz_si)
L2 = _unprime(-0.465, -0.0316, 1.82e7, 4.97e7, 970056.0)  # slug-ft^2
check("_unprime is invariant to the inertia unit system",
      float(np.abs(np.asarray(L1) - np.asarray(L2)).max()) < 1e-12,
      f"{float(np.abs(np.asarray(L1) - np.asarray(L2)).max()):.2e}")

print()
print("FAILURES:", fails if fails else "none")
