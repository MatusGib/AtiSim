"""D9: the microburst against its primary source (refs/NASA-TM-100632), and the
missing fourth parameter z_h."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import jax.numpy as jnp
from scipy.optimize import brentq

import flightsim  # noqa: F401
from flightsim import wind
import wcommon as W

b = W.BURST
lam, R, zs, eps = (float(b.lam), float(b.radius), float(b.z_star), float(b.epsilon))
print(f"burst: lam {lam:.6f} 1/s  R {R:.1f} m  z* {zs:.3f} m  eps {eps:.3f} m")

print("\n" + "=" * 78)
print("D9a  THE PAPER'S OWN TWO STATIONARITY EQUATIONS (TM-100632 p.4)")
print("=" * 78)
print("  r-derivative :  2 (r/R)^2 = exp((r/R)^2) - 1")
print("                  [the scan prints exp(-(r/R)^2)-1, which has no positive")
print("                   root; the + sign is forced and is equivalent to the")
print("                   module docstring's exp(-x^2)(2x^2+1) = 1]")
f1 = lambda y: 2 * y + 1 - np.exp(y)
y = brentq(f1, 0.5, 3.0, xtol=1e-15, rtol=1e-15)
print(f"                  root  r/R = {np.sqrt(y):.10f}    paper quotes 1.1212"
      f"   (rel {abs(np.sqrt(y)-1.1212)/np.sqrt(y):.3e})")
print("  z-derivative :  z_m/z* = ln(z*/eps) / ((z*/eps) - 1)")
t = np.log(12.5) / 11.5
print(f"                  at z*/eps = 12.5 -> z_m/z* = {t:.10f}"
      f"   paper quotes 0.22 (rel {abs(t-0.22)/t:.3e})")
k = brentq(lambda kk: np.log(kk) - 0.22 * (kk - 1), 2.0, 50.0, xtol=1e-14)
print(f"                  at z_m/z* = 0.22 -> z*/eps = {k:.10f}"
      f"   paper quotes 12.5 (rel {abs(k-12.5)/k:.3e})")
print("\n  PAPER'S LOGIC (p.4-5, quoted): z_m/z* = 0.22 is an EMPIRICAL TASS")
print("  observation -- 'It was also noted that the ratio z_m/z* = 0.22'. Then")
print("  'Recalling that z_m/z* = 0.22, the values 1.1212 and 12.5 were obtained")
print("  from iteration'. So 0.22 is an INPUT, and 12.5 and 0.2357 are its")
print("  consequences. Only 1.1212 is independent of it.")

print("\n" + "=" * 78)
print("D9b  EQS. 5 AND 6 vs THE CODE, TERM BY TERM")
print("=" * 78)
print("  paper (5): u = (lam R^2 / 2r)[1 - e^{-(r/R)^2}](e^{-z/z*} - e^{-z/eps})")
print("  paper (6): w = -lam e^{-(r/R)^2}[eps(e^{-z/eps} - 1) - z*(e^{-z/z*} - 1)]")
print("  code w_up  = -lam e^{-s} [z*(1 - e^{-z/z*}) - eps(1 - e^{-z/eps})]")
print("             == paper (6) after distributing the minus sign.  identical.")
print("  numerical check of the code against a literal transcription of (5)/(6):")
worst_u = worst_w = 0.0
for r in np.linspace(1.0, 5000.0, 200):
    for z in np.linspace(1.0, 2000.0, 60):
        p = jnp.array([r, 0.0, -z])
        got = np.asarray(W.mb_field(p))
        u_paper = (lam * R**2 / (2 * r)) * (1 - np.exp(-((r / R) ** 2))) * \
                  (np.exp(-z / zs) - np.exp(-z / eps))
        w_paper = -lam * np.exp(-((r / R) ** 2)) * (
            eps * (np.exp(-z / eps) - 1) - zs * (np.exp(-z / zs) - 1))
        worst_u = max(worst_u, abs(got[0] - u_paper) / max(abs(u_paper), 1e-12))
        worst_w = max(worst_w, abs(-got[2] - w_paper) / max(abs(w_paper), 1e-12))
print(f"    worst relative error, u: {worst_u:.3e}   w: {worst_w:.3e}"
      "   (12000 (r,z) points)")

print("\n" + "=" * 78)
print("D9c  THE PAPER'S FOURTH PARAMETER, z_h, IS NOT IN THE MODULE")
print("=" * 78)
print("  TM-100632 p.5: 'Different shears can be modeled by specifying four")
print("  parameters ... 1) a characteristic horizontal dimension; 2) maximum wind")
print("  velocity; 3) altitude of maximum outflow; and 4) DEPTH OF OUTFLOW.'")
print("  wind.microburst() takes three: u_max, radius, z_m. z_h never appears.")
print("  z_h does not enter eqs. 5/6, so nothing is mis-computed -- but it is the")
print("  altitude at which the paper stops the model ('the maximum vertical wind")
print("  is located at r = 0 and z = z_h, by definition'), and without it the")
print("  field has NO CEILING. On the axis:")
print(f"    {'z (m)':>9s} {'|w_up| (m/s)':>14s} {'|w_up| (kt)':>13s}")
for z in [50, 150, 300, 500, 1000, 2000, 5000, 20000, 1e6]:
    wv = float(W.mb_field(jnp.array([0.0, 0.0, -z]))[2])
    print(f"    {z:9.0f} {abs(wv):14.4f} {abs(wv)*1.94384:13.2f}")
print(f"    asymptote lam(z* - eps) = {lam*(zs-eps):.4f} m/s"
      f" = {lam*(zs-eps)*1.94384:.2f} kt, reached monotonically.")
print("  The paper's own example has 21 kt of downflow at z_m and 29 kt at z_h,")
print("  so its z_h is a few hundred metres. The module reaches 29 kt at:")
target = 29 / 1.94384
zh = brentq(lambda z: abs(float(W.mb_field(jnp.array([0.0, 0.0, -z]))[2])) - target,
            1.0, 2000.0, xtol=1e-6)
print(f"    z = {zh:.1f} m  (so the project's 300 m AGL penetration sits just"
      f" {'above' if 300 > zh else 'below'} it)")
print("  scripts/microburst.py flies at 300 m AGL by default; nothing in wind.py")
print("  refuses a caller who asks for 5 km, where the model returns a 45 m/s")
print("  downdraft it was never fitted for.")

print("\n" + "=" * 78)
print("D9d  'about 89 percent of the radius of peak outflow' (TM-100632 p.5)")
print("=" * 78)
print(f"  1 / 1.1212 = {1/1.1212:.6f}    1 / {np.sqrt(y):.6f} = {1/np.sqrt(y):.6f}"
      "   both round to 89%. consistent.")
