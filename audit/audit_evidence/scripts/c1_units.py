"""Agent C, item 1: conversion factors against their definitions, exactly.

Every factor is compared against an INDEPENDENTLY WRITTEN literal taken from the
defining standard (NIST SP 811 App. B.9 / ISO 80000 / the 1959 international
yard-and-pound agreement), not against flightsim.units itself.
"""
from fractions import Fraction as F

from flightsim import units as U

FAILS = []


def check(name, got, want, note, exact=True):
    if exact:
        ok = (got == want)
    else:
        ok = abs(got - want) <= 1e-15 * max(1.0, abs(want))
    print(f"{'PASS' if ok else 'FAIL'}  {name:22s} got={got!r:26s} want={want!r:26s}  {note}")
    if not ok:
        FAILS.append(name)


print("=== 1a. Conversion factors vs. defining standards ===")
# 1959 international yard: 1 yd = 0.9144 m exactly -> 1 ft = 0.3048 m exactly.
check("FT2M", U.FT2M, 0.3048, "1959 intl foot, exact")
check("M2FT", U.M2FT, 1.0 / 0.3048, "reciprocal")

# 1 lbf = 0.45359237 kg * 9.80665 m/s^2 = 4.4482216152605 N exactly.
lbf_exact = F(45359237, 10**8) * F(980665, 10**5)
print(f"      lbf exact rational = {lbf_exact} = {float(lbf_exact)!r}")
check("LBF2N", U.LBF2N, float(lbf_exact), "0.45359237 kg x 9.80665 m/s^2, exact")
check("LBF2N literal", U.LBF2N, 4.4482216152605, "NIST SP811 B.9")

check("LB2KG", U.LB2KG, 0.45359237, "1959 intl pound, exact")

# 1 slug = 1 lbf.s^2/ft
check("SLUG2KG", U.SLUG2KG, 4.4482216152605 / 0.3048, "lbf/ft, = 14.5939029372064")
print(f"      SLUG2KG = {U.SLUG2KG!r}   (NIST: 14.59390 kg)")

check("SLUG_FT2_TO_KG_M2", U.SLUG_FT2_TO_KG_M2,
      (4.4482216152605 / 0.3048) * 0.3048 * 0.3048, "slug.ft^2 -> kg.m^2")
print(f"      SLUG_FT2_TO_KG_M2 = {U.SLUG_FT2_TO_KG_M2!r}  (NIST: 1.355818 kg.m^2)")
# Cross-check: slug.ft^2 = lbf.ft.s^2 = 4.4482216152605*0.3048
check("  same, via lbf.ft", U.SLUG_FT2_TO_KG_M2, 4.4482216152605 * 0.3048,
      "identity slug.ft^2 == lbf.ft.s^2")

import math
check("DEG2RAD", U.DEG2RAD, math.pi / 180.0, "definition")
check("RAD2DEG", U.RAD2DEG, 180.0 / math.pi, "definition")
check("PER_DEG_TO_PER_RAD", U.PER_DEG_TO_PER_RAD, 180.0 / math.pi,
      "d/ddeg -> d/drad multiplies by 180/pi")

# International nautical mile = 1852 m exactly (1929 Monaco / SI).
check("KT2MS", U.KT2MS, 1852.0 / 3600.0, "1852 m/h, exact -> 0.514444...")
print(f"      KT2MS = {U.KT2MS!r}  (NIST: 0.5144444 m/s)")
check("FTS2MS", U.FTS2MS, 0.3048, "ft/s -> m/s is the same number as ft->m")

check("SLUG_FT3_TO_KG_M3", U.SLUG_FT3_TO_KG_M3,
      (4.4482216152605 / 0.3048) / 0.3048**3, "slug/ft^3 -> kg/m^3")
print(f"      SLUG_FT3_TO_KG_M3 = {U.SLUG_FT3_TO_KG_M3!r}  (lit: 515.3788 kg/m^3)")

# Mechanical horsepower = 550 ft.lbf/s exactly.
check("HP2W", U.HP2W, 550.0 * 4.4482216152605 * 0.3048, "550 ft.lbf/s")
print(f"      HP2W = {U.HP2W!r}  (NIST: 745.6999 W)")

print()
print("=== 1b. Round-trip / internal consistency ===")
check("FT2M*M2FT", U.FT2M * U.M2FT, 1.0, "round trip", exact=False)
check("DEG2RAD*RAD2DEG", U.DEG2RAD * U.RAD2DEG, 1.0, "round trip", exact=False)
# 1 slug should weigh 32.174049 lbf under g0 -> mass ratio slug/lb
print(f"      SLUG2KG/LB2KG = {U.SLUG2KG / U.LB2KG!r}   (should be g0 in ft/s^2 = "
      f"{9.80665 / 0.3048!r})")
check("slug/lb == g0[ft/s^2]", U.SLUG2KG / U.LB2KG, 9.80665 / 0.3048,
      "1 slug = g0(ft/s^2) lb", exact=False)

print()
print("FAILURES:", FAILS if FAILS else "none")
