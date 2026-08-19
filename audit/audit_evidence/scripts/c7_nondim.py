"""Agent C, items 5 and 6: per-radian, and the rate non-dimensionalisation.

Item 6 is settled by a round trip, not by reading the code: drive the plant with
a pure body rate and recover the DIMENSIONAL derivative the source tabulates.
q_hat carries c/(2V) and p_hat, r_hat carry b/(2V), so using the wrong reference
length changes the recovered number by c/b -- a factor of 7.2 for the 747 -- and
using the wrong factor of 2 halves or doubles it.  Neither can hide.
"""
import jax.numpy as jnp
import numpy as np

from flightsim import aero, airframe
from flightsim.aircraft import (
    CRUISE, REGISTRY, FlightCondition, from_dimensional_longitudinal,
    from_dimensional_lateral, from_dimensional_controls,
)
from flightsim.state import Controls
from flightsim.units import FT2M, DEG2RAD, RAD2DEG

fails, findings = [], []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name:62s} {detail}")
    if not ok:
        fails.append(name)


ZERO3 = jnp.zeros(3)
NOCTL = Controls(jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0))
A_HI = jnp.array(1.0e6)  # kill wave drag so only the linear terms are in play

print("=" * 78)
print("6.1 747 cruise: recover CR-2144 Table IX-4's Zq and Mq from the plant")
print("=" * 78)
ac = REGISTRY["boeing747"]
V = 774.0 * FT2M
qbar = 177.0 * 4.4482216152605 / FT2M**2
rho = 2.0 * qbar / V**2
m = float(ac.mass)
Iyy = float(np.asarray(ac.inertia)[1, 1])
eps = 1e-5
base = jnp.array([V, 0.0, 0.0])


def FM(vel, om, ctl=NOCTL):
    return aero.aero_forces_moments(vel, om, ctl, ac, jnp.array(rho), A_HI)


F0, M0 = FM(base, ZERO3)
Fq, Mq_ = FM(base, jnp.array([0.0, eps, 0.0]))
Zq_model = float(Fq[2] - F0[2]) / eps / m         # m/s  (SI)
Mq_model = float(Mq_[1] - M0[1]) / eps / Iyy      # 1/s
Zq_tab, Mq_tab = -5.16 * FT2M, -0.339
print(f"   dZ/dq / m  = {Zq_model:+.6f} m/s   table Zq = {-5.16:.3f} ft/s "
      f"= {Zq_tab:+.6f} m/s   err {100*(Zq_model/Zq_tab-1):+.4f}%")
print(f"   dM/dq / Iyy= {Mq_model:+.6f} 1/s   table Mq = {Mq_tab:+.6f} 1/s     "
      f"err {100*(Mq_model/Mq_tab-1):+.4f}%")
check("Zq round trip (q_hat = q*c/2V, chord)", abs(Zq_model / Zq_tab - 1) < 2e-3,
      f"{100*(Zq_model/Zq_tab-1):+.4f}%")
check("Mq round trip (q_hat = q*c/2V, chord)", abs(Mq_model / Mq_tab - 1) < 2e-3,
      f"{100*(Mq_model/Mq_tab-1):+.4f}%")

# The decisive discrimination: what would the SPAN give?
print(f"   -- had q_hat used the span b instead of c, Mq would come out "
      f"{Mq_model * float(ac.b) / float(ac.c):+.4f} (x{float(ac.b)/float(ac.c):.2f}) "
      f"-- not {Mq_tab}")
print(f"   -- had q_hat omitted the factor 2, Mq would come out "
      f"{Mq_model * 2:+.4f}")

# Zw / Mw : the alpha nondimensionalisation
Fw, Mw_ = FM(jnp.array([V, 0.0, eps]), ZERO3)
Zw_model = float(Fw[2] - F0[2]) / eps / m
Mw_model = float(Mw_[1] - M0[1]) / eps / Iyy
# UNIT CARE.  Zw has units (ft/s^2)/(ft/s) = 1/s -- length cancels, so the
# tabulated number is already SI.  Mw has units (rad/s^2)/(ft/s) = 1/(ft.s), so
# the tabulated number must be DIVIDED by FT2M to become 1/(m.s).  Omitting that
# was the first run's spurious 228% failure -- and 228% is exactly M2FT, which
# is how it was diagnosed.
Zw_tab = -0.317              # 1/s, length-free
Mw_tab = -0.00105 / FT2M     # 1/(ft.s) -> 1/(m.s)
print(f"   dZ/dw / m  = {Zw_model:+.6f} 1/s      table Zw = {Zw_tab:+.6f} 1/s     "
      f"err {100*(Zw_model/Zw_tab-1):+.3f}%")
print(f"   dM/dw / Iyy= {Mw_model:+.8f} 1/(m.s)  table Mw = {-0.00105:+.8f} 1/(ft.s) "
      f"= {Mw_tab:+.8f} 1/(m.s)  err {100*(Mw_model/Mw_tab-1):+.3f}%")
check("Zw round trip (alpha = w/U0, per RADIAN)",
      abs(Zw_model / Zw_tab - 1) < 0.05, f"{100*(Zw_model/Zw_tab-1):+.3f}%")
check("Mw round trip (alpha = w/U0, per RADIAN)",
      abs(Mw_model / Mw_tab - 1) < 0.02, f"{100*(Mw_model/Mw_tab-1):+.3f}%")
findings.append(
    "Zw round-trips to ~1% rather than exactly, because aircraft.py's CLa "
    "relation subtracts the trim CD and the plant's dZ/dw at alpha=0 also "
    "picks up the induced-drag slope. This is a linearisation-point effect, "
    "not a unit error: the discrepancy is O(CD), not O(57.3) or O(b/c)."
)

print()
print("=" * 78)
print("6.2 747 cruise: span-referenced p_hat and r_hat (already exact in c4)")
print("=" * 78)
Ixx = float(np.asarray(ac.inertia)[0, 0])
Izz = float(np.asarray(ac.inertia)[2, 2])
_, Mp = FM(base, jnp.array([eps, 0.0, 0.0]))
_, Mr = FM(base, jnp.array([0.0, 0.0, eps]))
# raw (unprimed) L_p, N_p, L_r, N_r recovered from the moments alone
Lp_raw = float(Mp[0] - M0[0]) / eps / Ixx
Np_raw = float(Mp[2] - M0[2]) / eps / Izz
Lr_raw = float(Mr[0] - M0[0]) / eps / Ixx
Nr_raw = float(Mr[2] - M0[2]) / eps / Izz
print(f"   raw Lp={Lp_raw:+.6f}  Np={Np_raw:+.6f}  Lr={Lr_raw:+.6f}  Nr={Nr_raw:+.6f}")
print(f"   (c4_ixz.py already showed these reproduce Table IX-8's PRIMED values "
      f"to machine precision once I^-1 is applied)")
print(f"   -- had p_hat used the chord c, Lp would be "
      f"{Lp_raw * float(ac.c) / float(ac.b):+.5f}, a factor {float(ac.b)/float(ac.c):.2f} out")
check("p_hat and r_hat use the SPAN (Clp is O(-0.35), not O(-2.5))",
      abs(float(ac.Clp)) < 1.0 and float(ac.Clp) < 0, f"Clp={float(ac.Clp):+.4f}")
check("q_hat uses the CHORD (Cmq is O(-20), not O(-2.9))",
      float(ac.Cmq) < -5.0, f"Cmq={float(ac.Cmq):+.4f}")

print()
print("=" * 78)
print("6.3 aircraft.py's helpers use the same reference lengths as aero.py")
print("=" * 78)
# Build a synthetic FlightCondition, push known DIMENSIONAL derivatives through
# the helpers, install the coefficients on a scratch aircraft and read the
# dimensional derivatives back out of the plant.  Anything inconsistent between
# the helper and aero.coefficients shows up as a ratio != 1.
fc = FlightCondition(airspeed=60.0, density=1.05, mass=1000.0, Ixx=1300.0,
                     Iyy=1800.0, Izz=2600.0, S=16.0, b=11.0, c=1.5)
DIMS = dict(Zw=-1.7, Zq=-1.9, Mw=-0.28, Mq=-2.2, Zde=-17.0, Mde=-44.0)
CLa, CLq, CLde, Cma, Cmq, Cmde = from_dimensional_longitudinal(fc, CD=0.0, **DIMS)
LAT = dict(Yv=-0.15, Lv=-0.12, Nv=0.17, Lp=-2.3, Np=-1.7, Lr=1.05, Nr=-1.03)
CYb, Clb, Cnb, Clp, Cnp, Clr, Cnr = from_dimensional_lateral(fc, **LAT)
CTL = dict(Ydr=2.1, Lda=3.1, Nda=0.4, Ldr=0.6, Ndr=-6.5)
CYdr, Clda, Cnda, Cldr, Cndr = from_dimensional_controls(fc, **CTL)

from flightsim.aircraft import inertia_tensor
I = inertia_tensor(fc.Ixx, fc.Iyy, fc.Izz, 0.0)
scratch = ac._replace(
    mass=jnp.array(fc.mass), inertia=I, inertia_inv=jnp.linalg.inv(I),
    S=jnp.array(fc.S), b=jnp.array(fc.b), c=jnp.array(fc.c),
    CD0=jnp.array(0.0), e=jnp.array(1e12), AR=jnp.array(fc.b**2 / fc.S),
    CL0=jnp.array(0.0), CLa=CLa, CLq=CLq, CLde=CLde,
    Cm0=jnp.array(0.0), Cma=Cma, Cmq=Cmq, Cmde=Cmde,
    CYb=CYb, CYp=jnp.array(0.0), CYr=jnp.array(0.0), CYdr=CYdr,
    Clb=Clb, Clp=Clp, Clr=Clr, Clda=Clda, Cldr=Cldr,
    Cnb=Cnb, Cnp=Cnp, Cnr=Cnr, Cnda=Cnda, Cndr=Cndr,
)
U0 = fc.airspeed


def dim(pert_vel=None, pert_om=None, ctl=None):
    v = jnp.array([U0, 0.0, 0.0]) if pert_vel is None else pert_vel
    o = ZERO3 if pert_om is None else pert_om
    c = NOCTL if ctl is None else ctl
    return aero.aero_forces_moments(v, o, c, scratch, jnp.array(fc.density), A_HI)


f0, m0 = dim()
rows = []
fq, mq = dim(pert_om=jnp.array([0.0, eps, 0.0]))
rows += [("Zq", float(fq[2] - f0[2]) / eps / fc.mass, DIMS["Zq"]),
         ("Mq", float(mq[1] - m0[1]) / eps / fc.Iyy, DIMS["Mq"])]
fw, mw = dim(pert_vel=jnp.array([U0, 0.0, eps]))
rows += [("Zw", float(fw[2] - f0[2]) / eps / fc.mass, DIMS["Zw"]),
         ("Mw", float(mw[1] - m0[1]) / eps / fc.Iyy, DIMS["Mw"])]
fe, me = dim(ctl=NOCTL._replace(elevator=jnp.array(eps)))
rows += [("Zde", float(fe[2] - f0[2]) / eps / fc.mass, DIMS["Zde"]),
         ("Mde", float(me[1] - m0[1]) / eps / fc.Iyy, DIMS["Mde"])]
fv, mv = dim(pert_vel=jnp.array([U0, eps, 0.0]))
rows += [("Yv", float(fv[1] - f0[1]) / eps / fc.mass, LAT["Yv"]),
         ("Lv", float(mv[0] - m0[0]) / eps / fc.Ixx, LAT["Lv"]),
         ("Nv", float(mv[2] - m0[2]) / eps / fc.Izz, LAT["Nv"])]
fp, mp = dim(pert_om=jnp.array([eps, 0.0, 0.0]))
rows += [("Lp", float(mp[0] - m0[0]) / eps / fc.Ixx, LAT["Lp"]),
         ("Np", float(mp[2] - m0[2]) / eps / fc.Izz, LAT["Np"])]
fr, mr = dim(pert_om=jnp.array([0.0, 0.0, eps]))
rows += [("Lr", float(mr[0] - m0[0]) / eps / fc.Ixx, LAT["Lr"]),
         ("Nr", float(mr[2] - m0[2]) / eps / fc.Izz, LAT["Nr"])]
fa, ma = dim(ctl=NOCTL._replace(aileron=jnp.array(eps)))
rows += [("Lda", float(ma[0] - m0[0]) / eps / fc.Ixx, CTL["Lda"]),
         ("Nda", float(ma[2] - m0[2]) / eps / fc.Izz, CTL["Nda"])]
fd, md = dim(ctl=NOCTL._replace(rudder=jnp.array(eps)))
rows += [("Ydr", float(fd[1] - f0[1]) / eps / fc.mass, CTL["Ydr"]),
         ("Ldr", float(md[0] - m0[0]) / eps / fc.Ixx, CTL["Ldr"]),
         ("Ndr", float(md[2] - m0[2]) / eps / fc.Izz, CTL["Ndr"])]

print(f"   {'deriv':6s} {'recovered':>14s} {'input':>12s} {'ratio':>10s}")
for nm, got, want in rows:
    ratio = got / want
    print(f"   {nm:6s} {got:14.6f} {want:12.5f} {ratio:10.6f}")
    check(f"   round trip {nm}", abs(ratio - 1.0) < 2e-4, f"ratio={ratio:.8f}")

print()
print("=" * 78)
print("5. PER-RADIAN: no degree-valued quantity reaches the plant")
print("=" * 78)
# (a) A 1 deg deflection must give exactly DEG2RAD times the 1 rad response.
for name in ["boeing747", "cherokee", "cessna172", "boeing747_approach"]:
    a = REGISTRY[name]
    v = jnp.array([float(CRUISE[name]["airspeed"]), 0.0, 0.0])
    _, m1 = aero.aero_forces_moments(v, ZERO3, NOCTL._replace(elevator=jnp.array(1.0)),
                                     a, jnp.array(0.4), A_HI)
    _, m0_ = aero.aero_forces_moments(v, ZERO3, NOCTL, a, jnp.array(0.4), A_HI)
    _, md_ = aero.aero_forces_moments(v, ZERO3,
                                      NOCTL._replace(elevator=jnp.array(DEG2RAD)),
                                      a, jnp.array(0.4), A_HI)
    ratio = float(md_[1] - m0_[1]) / float(m1[1] - m0_[1])
    check(f"   {name}: 1 deg gives exactly DEG2RAD x the 1 rad moment",
          abs(ratio - DEG2RAD) < 1e-12, f"ratio={ratio:.12f} vs {DEG2RAD:.12f}")

# (b) Magnitude plausibility: a per-DEGREE leak is a factor 57.3 error, which
#     always shows as an absurd derivative.  Bands are generous by design.
print()
BANDS = {
    "CLa": (3.0, 7.0), "CLq": (1.0, 15.0), "CLde": (0.1, 1.5),
    "Cma": (-3.0, -0.2), "Cmq": (-40.0, -3.0), "Cmde": (-4.0, -0.3),
    "CYb": (-1.5, -0.1), "Clb": (-0.5, -0.02), "Clp": (-0.8, -0.2),
    "Clr": (0.02, 0.5), "Cnb": (0.005, 0.5), "Cnr": (-0.5, -0.02),
    "Clda": (0.005, 0.5),
}
for name in ["boeing747", "boeing747_approach", "cherokee", "cessna172"]:
    a = REGISTRY[name]
    bad = [k for k, (lo, hi) in BANDS.items() if not (lo <= float(getattr(a, k)) <= hi)]
    vals = "  ".join(f"{k}={float(getattr(a, k)):+.4f}" for k in
                     ["CLa", "Cma", "Cmq", "Clp", "Cnb"])
    print(f"   {name:20s} {vals}")
    check(f"   {name}: every derivative is in its per-RADIAN band",
          not bad, f"outside band: {bad}" if bad else "")

# (c) The Cessna's tables are tabulated per DEGREE of alpha. Confirm the fit was
#     done in radians.
from flightsim.aircraft import CESSNA172_TABLES as T
ad = np.array(T["alpha_deg"])
lin = ad <= 10.0
slope_per_deg = np.polyfit(ad[lin], np.array(T["CL"])[lin], 1)[0]
print(f"\n   Cessna CL table slope = {slope_per_deg:.5f} /deg "
      f"= {slope_per_deg * RAD2DEG:.4f} /rad;  ac.CLa = "
      f"{float(REGISTRY['cessna172'].CLa):.4f}")
check("   Cessna CLa is the PER-RADIAN fit of a per-degree table",
      abs(float(REGISTRY["cessna172"].CLa) - slope_per_deg * RAD2DEG) < 1e-3,
      f"{float(REGISTRY['cessna172'].CLa):.5f} vs {slope_per_deg * RAD2DEG:.5f}")

# (d) alpha_ref used to read the alpha-dependent lateral table must be radians.
c172 = REGISTRY["cessna172"]
qS = 0.5 * 1.055 * 60.0**2 * 16.2
alpha_ref = (1043.3 * 9.80665 / qS - float(c172.CL0)) / float(c172.CLa)
print(f"   Cessna alpha_ref = {alpha_ref:.5f} rad = {alpha_ref * RAD2DEG:.3f} deg")
check("   alpha_ref is a radian value in a sane flight range",
      0.0 < alpha_ref < 0.20,
      f"{alpha_ref:.4f} rad ({alpha_ref * RAD2DEG:.2f} deg)")
clb_interp = float(np.interp(alpha_ref, ad * DEG2RAD, np.array(T["Clb"])))
check("   at_reference interpolates against RADIANS",
      abs(float(c172.Clb) - clb_interp) < 1e-9,
      f"Clb={float(c172.Clb):.6f} vs {clb_interp:.6f}")

# (e) Control limits are radians.
for name in REGISTRY:
    a = REGISTRY[name]
    ok = all(0.01 < float(getattr(a, k)) < 1.0
             for k in ("elevator_limit", "aileron_limit", "rudder_limit"))
    check(f"   {name}: control limits are radians, not degrees", ok,
          f"elev={float(a.elevator_limit):.4f} rad "
          f"({float(a.elevator_limit) * RAD2DEG:.1f} deg)")

# (f) sweep is radians
for name in REGISTRY:
    a = REGISTRY[name]
    check(f"   {name}: sweep is radians", 0.0 <= float(a.sweep) < 1.5,
          f"sweep={float(a.sweep):.4f} rad ({float(a.sweep) * RAD2DEG:.1f} deg)")

print()
print("FAILURES:", fails if fails else "none")
for f in findings:
    print("\nNOTE:", f)
