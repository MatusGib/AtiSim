"""Agent C, item 1: the ONE dimensional-consistency defect found in
`from_dimensional_longitudinal` -- an axis-definition mismatch, quantified.

CR-2144 Appendix A (printed A-16/A-17) defines its dimensional Z derivatives in
terms of C_N (normal force, positive UP) and C_X (axial, positive AFT).  The
code inverts them as though they were defined in terms of C_L and C_D.  Those
coincide only at alpha = 0; the 747 cruise set is linearised at alpha0 = 4.60 deg
and the approach set at 5.70 deg.

Nothing here is taken on trust from NOTATION.md: the exact relation is written
out from the definitions and the size of the discrepancy is computed.

    C_N =  C_L cos a + C_D sin a
    C_X =  C_D cos a - C_L sin a        (CR-2144's aft-positive C_X)

    dC_N/da = C_La cos a - C_L sin a + C_Da sin a + C_D cos a

The code's relation, CLa = -Zw*m*U0/qS - CD, is the a -> 0 limit of
    C_La = (C_Na + C_L sin a - C_Da sin a - C_D cos a) / cos a
"""
import math

import numpy as np

from flightsim.aircraft import REGISTRY
from flightsim.units import DEG2RAD, FT2M

print("=" * 78)
print("747 CRUISE (flight condition 9), alpha0 = 4.60 deg")
print("=" * 78)
# The numbers aircraft.py itself uses, restated from CR-2144 Table IX-3/IX-4.
S, b, c = 5500.0, 195.68, 27.31
W = 636636.0
U0, qbar = 774.0, 177.0
alpha0 = 4.60 * DEG2RAD
CD_trim = 0.043
Zw = -0.317
Xw = 0.0389
G = 32.174
m = W / G
qS = qbar * S
AR = b * b / S
CL_trim = W / qS

# What the code does (verbatim from aircraft.py lines 263 and 281):
CLa_code = -Zw * m * U0 / qS - CD_trim
CDa_code = (-Xw * m * U0 / qS + CD_trim * math.sin(alpha0)
            + CLa_code * math.sin(alpha0)
            + CL_trim * math.cos(alpha0)) / math.cos(alpha0)

# What Appendix A actually defines.  Dropping the W0 term (a separate, already
# recorded issue) the source relation is Zw = -(qS/(m U0)) * C_Na, so:
CNa = -Zw * m * U0 / qS
ca, sa = math.cos(alpha0), math.sin(alpha0)

# Solve the exact pair for (CLa, CDa) simultaneously -- they are coupled:
#   dC_N/da = CLa ca - CL sa + CDa sa + CD ca  = CNa
#   dC_X/da = CDa ca - CD sa - CLa sa - CL ca  = CXa   (aft-positive C_X)
# with CXa recovered from Xw the same way the code recovers CDa.
CXa = -Xw * m * U0 / qS   # sign convention: same inversion the code performs
A = np.array([[ca, sa], [-sa, ca]])
rhs = np.array([CNa + CL_trim * sa - CD_trim * ca,
                CXa + CD_trim * sa + CL_trim * ca])
CLa_exact, CDa_exact = np.linalg.solve(A, rhs)

print(f"   CL_trim = {CL_trim:.5f}   CD_trim = {CD_trim}   alpha0 = 4.60 deg")
print(f"   C_Na recovered from Zw               = {CNa:+.6f}")
print()
print(f"   CLa as the code computes it          = {CLa_code:.6f}")
print(f"   CLa from the exact C_N/C_L relation  = {CLa_exact:.6f}")
print(f"   difference                           = {CLa_exact - CLa_code:+.6f} "
      f"({100 * (CLa_code / CLa_exact - 1):+.3f}% low)")
print()
print(f"   CDa as the code computes it          = {CDa_code:.6f}")
print(f"   CDa from the exact C_X/C_D relation  = {CDa_exact:.6f}")
print(f"   difference                           = {CDa_exact - CDa_code:+.6f} "
      f"({100 * (CDa_code / CDa_exact - 1):+.3f}%)")
print()
print(f"   ac.CLa actually shipped              = {float(REGISTRY['boeing747'].CLa):.6f}"
      f"   (confirms the code path above)")

# Propagate to e and CD0, which are back-solved from CDa.
def polar(CLa, CDa):
    sweep = 37.5 * DEG2RAD
    cos_s = math.cos(sweep)
    m_dd = 0.87 / cos_s - 0.09 / cos_s**2 - CL_trim / (10.0 * cos_s**3)
    m_crit = m_dd - (0.1 / 80.0) ** (1.0 / 3.0)
    cd_wave = 20.0 * max(0.8 - m_crit, 0.0) ** 4
    dcd_wave = 80.0 * max(0.8 - m_crit, 0.0) ** 3 / (10.0 * cos_s**3) * CLa
    e = 2.0 * CL_trim * CLa / (math.pi * AR * (CDa - dcd_wave))
    CD0 = CD_trim - CL_trim**2 / (math.pi * e * AR) - cd_wave
    return e, CD0


e_c, CD0_c = polar(CLa_code, CDa_code)
e_e, CD0_e = polar(CLa_exact, CDa_exact)
print(f"   e:   code {e_c:.5f}  exact {e_e:.5f}  ({100 * (e_c / e_e - 1):+.3f}%)")
print(f"   CD0: code {CD0_c:.6f}  exact {CD0_e:.6f}  ({100 * (CD0_c / CD0_e - 1):+.3f}%)")

print()
print("=" * 78)
print("747 POWER APPROACH -- NOT affected")
print("=" * 78)
print("   Table IX-2 is already NON-DIMENSIONAL and states '/rad'. No Z/X")
print("   derivative is inverted for this aircraft, so the C_N/C_L question")
print("   cannot arise. CLa = 5.70 is the table value verbatim:",
      float(REGISTRY["boeing747_approach"].CLa))

print()
print("=" * 78)
print("CHEROKEE -- affected in principle, but alpha0 = 0 so the error VANISHES")
print("=" * 78)
print("   aircraft.py: 'theta0 = 0 in level flight, so alpha0 = 0'. C_N == C_L")
print("   and C_X == C_D identically at alpha = 0, so `from_dimensional_"
      "longitudinal`")
print("   is exact for this aircraft. The defect is specific to a set")
print("   linearised at non-zero alpha0, i.e. the 747 cruise set alone.")

print()
print("=" * 78)
print("CESSNA -- only CLde and Cmde go through the helper")
print("=" * 78)
print("   `from_dimensional_longitudinal(source_fc, Zw=0, Zq=0, Mw=0, Mq=0,")
print("    Zde=-17.19, Mde=-36.23, CD=0.0)`. CLde = -Zde*m/qS carries no alpha")
print("   term in Appendix A at all, and Cmde none either, so both are exact.")
print()
print("SUMMARY: the C_N/C_L mismatch reaches exactly ONE aircraft (747 cruise)")
print(f"         and moves CLa by {100 * (CLa_code / CLa_exact - 1):+.2f}%, "
      f"CDa by {100 * (CDa_code / CDa_exact - 1):+.2f}%, "
      f"e by {100 * (e_c / e_e - 1):+.2f}%, CD0 by {100 * (CD0_c / CD0_e - 1):+.2f}%.")
