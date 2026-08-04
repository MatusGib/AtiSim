"""Unit conversion constants.

Sources for aircraft data are imperial and often per-degree. Convert at the
boundary -- in aircraft.py, at the point of definition -- using these names.
Never inline a conversion factor anywhere else.

All values are exact by definition of the international foot/pound (1959) unless
noted.
"""

import math

# Length
FT2M = 0.3048  # exact
M2FT = 1.0 / FT2M

# Force
LBF2N = 4.4482216152605  # exact

# Mass
LB2KG = 0.45359237  # exact
SLUG2KG = LBF2N / FT2M  # 1 slug = 1 lbf.s^2/ft = 14.5939... kg

# Inertia
SLUG_FT2_TO_KG_M2 = SLUG2KG * FT2M * FT2M

# Angle
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi

# Per-angle derivatives: a coefficient given per degree becomes per radian by
# dividing by DEG2RAD, i.e. multiplying by RAD2DEG.
PER_DEG_TO_PER_RAD = RAD2DEG

# Speed
KT2MS = 1852.0 / 3600.0  # exact; international nautical mile
FTS2MS = FT2M

# Density
SLUG_FT3_TO_KG_M3 = SLUG2KG / (FT2M**3)
