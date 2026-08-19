"""ATTACK 3: make trim.trim converge to a converged-but-absurd root.

trim.trim runs a FIXED 40 Newton iterations from INITIAL_GUESS and returns
whatever it lands on. is_physical is opt-in and only checks |alpha| <= 15 deg.
Map where in (V, h) the absurd roots start, using the REAL registry
coefficients only.
"""
from common import *  # noqa

from flightsim.trim import ALPHA_LIMIT, INITIAL_GUESS, is_physical

print("=== 3a. (V, h) sweep with the shipped coefficients ===")
print("  reporting: converged (|resid|<1e-8) AND absurd (|alpha|>15 deg,")
print("  or throttle outside [0,1], or elevator beyond its limit)")

for name in NAMES:
    ac = REGISTRY[name]
    Vc = CRUISE[name]["airspeed"]
    hc = CRUISE[name]["altitude"]
    Vs = np.linspace(max(2.0, 0.05 * Vc), 3.0 * Vc, 80)
    hs = np.array([0.0, 1000.0, 3000.0, 6000.0, 9000.0, 12000.0, 15000.0, 20000.0])
    rows = []
    for h in hs:
        for V in Vs:
            x, r = trim.trim(jnp.array(float(V)), jnp.array(float(h)), ac)
            xn = np.asarray(x)
            rn = float(np.linalg.norm(np.asarray(r)))
            rows.append((V, h, xn[0], xn[1], xn[2], rn))
    rows = np.array(rows)
    conv = rows[:, 5] < 1e-8
    alpha_bad = np.abs(rows[:, 2]) > ALPHA_LIMIT
    thr_bad = (rows[:, 4] < 0.0) | (rows[:, 4] > 1.0)
    de_bad = np.abs(rows[:, 3]) > float(ac.elevator_limit)
    absurd = conv & (alpha_bad | thr_bad | de_bad)
    print(f"\n  {name}: {int(conv.sum())}/{len(rows)} converged to |resid|<1e-8; "
          f"of those {int(absurd.sum())} are absurd")
    print(f"    alpha out of +-15 deg: {int((conv&alpha_bad).sum())}   "
          f"throttle out of [0,1]: {int((conv&thr_bad).sum())}   "
          f"|de| over limit: {int((conv&de_bad).sum())}")
    # how close to the real envelope does the first absurd root sit?
    sel = conv & alpha_bad
    if sel.any():
        sub = rows[sel]
        # nearest in V at cruise altitude
        near = sub[np.argmin(np.abs(sub[:, 0] - Vc))]
        print(f"    |alpha|>15 deg nearest to Vcruise={Vc:.1f}: V={near[0]:.1f} "
              f"(={near[0]/Vc:.2f}*Vc) h={near[1]:.0f}  alpha="
              f"{np.degrees(near[2]):.1f} deg thr={near[4]:.3f} resid={near[5]:.1e}")
        worst = sub[np.argmax(np.abs(sub[:, 2]))]
        print(f"    worst alpha: V={worst[0]:.1f} h={worst[1]:.0f} alpha="
              f"{np.degrees(worst[2]):.1f} deg de={np.degrees(worst[3]):.1f} deg "
              f"thr={worst[4]:.3f}  |resid|={worst[5]:.2e}")
    sel = conv & thr_bad
    if sel.any():
        sub = rows[sel]
        near = sub[np.argmin(np.abs(sub[:, 0] - Vc))]
        print(f"    throttle out of range nearest Vcruise: V={near[0]:.1f} "
              f"(={near[0]/Vc:.2f}*Vc) h={near[1]:.0f}  thr={near[4]:.3f} "
              f"alpha={np.degrees(near[2]):.1f} deg  is_physical="
              f"{is_physical(jnp.array(near[2:5]))}")
        lo = sub[:, 4].min(); hi = sub[:, 4].max()
        print(f"    throttle range over absurd roots: [{lo:.3f}, {hi:.3f}]")

print("\n=== 3b. how far can a CONVERGED root go? extreme (V,h), real data ===")
print(f"{'aircraft':20s} {'V':>8s} {'h':>8s} {'alpha_deg':>11s} {'de_deg':>9s} "
      f"{'throttle':>10s} {'|resid|':>10s} {'is_physical':>12s}")
cases = [(1.0, 0.0), (2.0, 0.0), (5.0, 0.0), (10.0, 0.0), (5.0, 20000.0),
         (10.0, 20000.0), (900.0, 0.0), (900.0, 20000.0), (400.0, 20000.0)]
for name in NAMES:
    ac = REGISTRY[name]
    for V, h in cases:
        x, r = trim.trim(jnp.array(V), jnp.array(h), ac)
        xn, rn = np.asarray(x), float(np.linalg.norm(np.asarray(r)))
        print(f"{name:20s} {V:8.1f} {h:8.0f} {np.degrees(xn[0]):11.2f} "
              f"{np.degrees(xn[1]):9.2f} {xn[2]:10.3f} {rn:10.2e} "
              f"{str(is_physical(x)):>12s}")

print("\n=== 3c. does trim converge at NEGATIVE altitude? (no ground exists) ===")
for name in NAMES:
    ac = REGISTRY[name]
    Vc = CRUISE[name]["airspeed"]
    for h in [-500.0, -5000.0, -20000.0]:
        x, r = trim.trim(jnp.array(Vc), jnp.array(h), ac)
        xn, rn = np.asarray(x), float(np.linalg.norm(np.asarray(r)))
        print(f"  {name:20s} V={Vc:6.1f} h={h:8.0f}  alpha={np.degrees(xn[0]):7.2f} deg "
              f"thr={xn[2]:7.3f}  |resid|={rn:.1e}  is_physical={is_physical(x)}")

print("\n=== 3d. degenerate coefficients: what does trim return? ===")
base = REGISTRY["cessna172"]
V, h = 60.0, 1500.0
degens = {
    "CLa -> 0.01 (nearly no lift slope)": dict(CLa=jnp.array(0.01)),
    "CLa -> -5.73 (SIGN FLIPPED)": dict(CLa=jnp.array(-5.7296)),
    "Cma -> +0.81 (statically UNSTABLE)": dict(Cma=jnp.array(0.8075)),
    "Cma -> 0 (neutral)": dict(Cma=jnp.array(0.0)),
    "Cmde -> 0 (no pitch control)": dict(Cmde=jnp.array(0.0)),
    "max_thrust -> 0": dict(max_thrust=jnp.array(0.0)),
    "mass -> 1e6 kg": dict(mass=jnp.array(1e6)),
    "CD0 -> 0 (no parasite drag)": dict(CD0=jnp.array(0.0)),
    "e -> 1e-4 (huge induced drag)": dict(e=jnp.array(1e-4)),
    "S -> 0.001 m^2": dict(S=jnp.array(0.001)),
}
for lbl, kw in degens.items():
    ac = base._replace(**kw)
    x, r = trim.trim(jnp.array(V), jnp.array(h), ac)
    xn, rn = np.asarray(x), float(np.linalg.norm(np.asarray(r)))
    fin = np.all(np.isfinite(xn))
    print(f"  {lbl:38s} alpha={np.degrees(xn[0]):12.2f} deg de={np.degrees(xn[1]):10.2f} "
          f"deg thr={xn[2]:12.4g} |resid|={rn:9.2e} finite={fin} "
          f"is_physical={is_physical(x) if fin else 'n/a'}")

print("\n=== 3e. is_physical's blind spots: roots it PASSES that are absurd ===")
print("  is_physical checks |alpha| <= 15 deg ONLY -- not throttle, not elevator,")
print("  not the residual. Roots that pass but are not flight conditions:")
for name in NAMES:
    ac = REGISTRY[name]
    Vc = CRUISE[name]["airspeed"]
    found = []
    for h in [0.0, 5000.0, 12000.0, 20000.0]:
        for V in np.linspace(2.0, 3.0 * Vc, 200):
            x, r = trim.trim(jnp.array(float(V)), jnp.array(float(h)), ac)
            xn = np.asarray(x)
            if not np.all(np.isfinite(xn)):
                continue
            if is_physical(x) and (xn[2] < -0.01 or xn[2] > 1.01
                                   or abs(xn[1]) > float(ac.elevator_limit)):
                found.append((V, h, xn))
    if found:
        V, h, xn = found[0]
        print(f"  {name:20s} {len(found)} such roots; e.g. V={V:.1f} h={h:.0f}: "
              f"alpha={np.degrees(xn[0]):.2f} deg (PASSES) but thr={xn[2]:.3f}, "
              f"de={np.degrees(xn[1]):.2f} deg")
    else:
        print(f"  {name:20s} none found")
