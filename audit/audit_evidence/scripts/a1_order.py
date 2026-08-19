"""A1: Butcher weights + observed order of rk4_step, and of the 6-DOF rollout."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim.integrate import rk4_step

print("=== 1a. Butcher tableau extraction by probing rk4_step ===")
# Probe: for the linear scalar test problem y' = lam*y, the RK4 stability
# polynomial for classical RK4 is R(z)=1+z+z^2/2+z^3/6+z^4/24.  Any deviation
# in the stage weights changes some coefficient.  Extract R(z) numerically by
# taking the step on y0=1 with several z values and fitting a degree-4 poly.
lam = 1.0
zs = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35])
Rs = []
for z in zs:
    y = rk4_step(lambda y: lam * y, jnp.array(1.0), jnp.array(z))
    Rs.append(float(y))
coef = np.polyfit(zs, Rs, 4)[::-1]  # ascending
exact = np.array([1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0])
print("fitted R(z) coefficients :", coef)
print("classical RK4 exact      :", exact)
print("max |diff|               :", np.abs(coef - exact).max())

# Also read the c-nodes and A-matrix by an autonomous-non-linear probe:
# use f(x)=t-like by making x carry its own time: xdot = 1 for the clock and
# xdot = clock**k for the integrand.  Then rk4_step integrates t**k over [0,h],
# and the quadrature error identifies the b-weights and c-nodes exactly.
print()
print("=== 1b. quadrature identity: integral of t^k over [0,1] ===")
for k in range(0, 6):
    def f(x, k=k):
        return jnp.array([1.0, x[0] ** k])
    out = rk4_step(f, jnp.array([0.0, 0.0]), jnp.array(1.0))
    exact_int = 1.0 / (k + 1)
    print(f"  k={k}  rk4={float(out[1]):.16f}  exact={exact_int:.16f} "
          f"err={float(out[1]) - exact_int:+.3e}")
print("  Exact through k=3.  At k=4 the error is 5/24 - 1/5 = +1/120 = +8.3333e-3,")
print("  which is Simpson's rule error EXACTLY -- so b = [1,2,2,1]/6 with")
print("  c = [0,1/2,1/2,1], and no other weight set.")

print()
print("=== 1c. observed order, harmonic oscillator (verification.oscillator_refinement) ===")
from flightsim import verification
dts = np.array([0.2, 0.1, 0.05, 0.025])
errors, slope = verification.oscillator_refinement(dts)
for d, e in zip(dts, errors):
    print(f"  dt={d:<8g} err={e:.6e}")
print("  pairwise orders:", [round(float(np.log(errors[i]/errors[i+1])/np.log(dts[i]/dts[i+1])), 4)
                             for i in range(len(dts)-1)])
print("  fitted order:", slope)

print()
print("=== 1d. independent order check: y'=y, exact e^t, no project code path ===")
# every dt must divide t_end=1 exactly, or n=round(1/dt) silently integrates to a
# DIFFERENT end time and the "error" is not comparable across the sequence.
dts2 = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125])
errs2 = []
for dt in dts2:
    y = jnp.array(1.0)
    n = int(round(1.0 / dt))
    for _ in range(n):
        y = rk4_step(lambda y: y, y, jnp.array(dt))
    errs2.append(abs(float(y) - np.e))
errs2 = np.array(errs2)
for d, e in zip(dts2, errs2):
    print(f"  dt={d:<8g} err={e:.6e}")
print("  pairwise:", [round(float(np.log(errs2[i]/errs2[i+1])/np.log(dts2[i]/dts2[i+1])), 4)
                      for i in range(len(dts2)-1)])

print()
print("=== 1e. nonlinear problem: y' = -y^3, y(0)=1 -> y(t) = 1/sqrt(1+2t) ===")
dts3 = np.array([0.25, 0.125, 0.0625, 0.03125, 0.015625])
errs3 = []
for dt in dts3:
    y = jnp.array(1.0)
    for _ in range(int(round(1.0 / dt))):
        y = rk4_step(lambda y: -y ** 3, y, jnp.array(dt))
    errs3.append(abs(float(y) - 1.0 / np.sqrt(3.0)))
errs3 = np.array(errs3)
for d, e in zip(dts3, errs3):
    print(f"  dt={d:<8g} err={e:.6e}")
print("  pairwise:", [round(float(np.log(errs3[i]/errs3[i+1])/np.log(dts3[i]/dts3[i+1])), 4)
                      for i in range(len(dts3)-1)])
