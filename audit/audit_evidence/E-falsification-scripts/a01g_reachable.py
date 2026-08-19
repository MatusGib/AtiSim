"""ATTACK 1g: is the energy-creating region reached from a TRIMMED start with
nothing but a control input a pilot can make? Quantify the energy created.

Metric: the largest rise in total mechanical energy from any local minimum,
in a run with throttle = 0 in still air. Any positive value is energy the model
created out of motionless, uniform air.
"""
from common import *  # noqa

DT = 0.002
SEC = 90.0
N = int(SEC / DT)


def max_rise(E):
    """Largest E[j] - E[i] for j > i (best possible 'energy created' in the run)."""
    run_min = np.minimum.accumulate(E)
    return float((E - run_min).max()), int(np.argmax(E - run_min))


print(f"=== energy created in still air, throttle 0, dt={DT}s, {SEC:.0f}s ===")
print(f"{'aircraft':20s} {'input':24s} {'E0 (J)':>12s} {'max rise (J)':>13s} "
      f"{'rise/E0':>11s} {'peak dE/dt (W)':>15s} {'max|q|':>7s}")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    el, al, rl = (float(ac.elevator_limit), float(ac.aileron_limit),
                  float(ac.rudder_limit))
    inputs = {
        "elevator +limit": ctrl(de=el),
        "elevator -limit": ctrl(de=-el),
        "elevator +limit, aileron": ctrl(de=el, da=al),
        "elev+ail+rud all +limit": ctrl(de=el, da=al, dr=rl),
        "elev -limit, ail+rud +": ctrl(de=-el, da=al, dr=rl),
        "trim elevator (control)": ctrl(de=float(x[1])),
    }
    for lbl, c in inputs.items():
        _, hist = run(st, c, ac, DT, N)
        E, ke, rke, pe = energy(hist, ac)
        if not np.all(np.isfinite(E)):
            print(f"{name:20s} {lbl:24s}  NON-FINITE")
            continue
        rise, i = max_rise(E)
        dE = np.diff(E)
        q = np.asarray(hist.omega)[:, 1]
        print(f"{name:20s} {lbl:24s} {E[0]:12.5g} {rise:13.5g} "
              f"{rise/abs(E[0]):11.3e} {dE.max()/DT:15.5g} "
              f"{np.abs(q).max():7.2f}")

print("\n=== the same, but ALSO check E never exceeds E0 ===")
for name in NAMES:
    ac, st, x, r = trim_state(name)
    el = float(ac.elevator_limit)
    worst = 0.0
    for c in [ctrl(de=el), ctrl(de=-el), ctrl(de=el, da=float(ac.aileron_limit)),
              ctrl(de=-el, da=float(ac.aileron_limit),
                   dr=float(ac.rudder_limit))]:
        _, hist = run(st, c, ac, DT, N)
        E, *_ = energy(hist, ac)
        worst = max(worst, float((E - E[0]).max()))
    print(f"  {name:20s} max (E - E0) over all inputs = {worst:+.6g} J")
