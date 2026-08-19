"""ATTACK 1: energy gain in still air, throttle = 0, aero on."""
from common import *  # noqa

import itertools

DT = 0.01


def scan_case(name, ac, st, controls, seconds, label):
    n = int(seconds / DT)
    _, hist = run(st, controls, ac, DT, n)
    E, ke, rke, pe = energy(hist, ac)
    if not np.all(np.isfinite(E)):
        return dict(label=label, name=name, status="NONFINITE",
                    first_bad=int(np.argmax(~np.isfinite(E))), E0=float(E[0]))
    # positive drift relative to running minimum-so-far start
    dE = np.diff(E)
    worst_gain_step = dE.max()
    net = E[-1] - E[0]
    # max excursion above initial
    above = (E - E[0]).max()
    return dict(
        label=label, name=name, status="ok",
        E0=float(E[0]), Efinal=float(E[-1]), net=float(net),
        above_initial=float(above),
        above_frac=float(above / abs(E[0])),
        worst_step_gain_W=float(worst_gain_step / DT),
        n_pos_steps=int((dE > 0).sum()), n_steps=len(dE),
        pos_frac=float((dE > 0).mean()),
        h_final=float(-np.asarray(hist.pos_ned)[-1, 2]),
        V_final=float(np.linalg.norm(np.asarray(hist.vel_body)[-1])),
    )


rows = []
for name in NAMES:
    ac, st, x, r = trim_state(name)
    print(f"\n=== {name}: trim alpha={float(x[0]):.4f} de={float(x[1]):.4f} "
          f"thr={float(x[2]):.4f} resid={float(np.linalg.norm(r)):.2e}")
    # (a) trimmed attitude, throttle cut, elevator held at trim
    rows.append(scan_case(name, ac, st, ctrl(de=float(x[1]), thr=0.0), 300.0,
                          "trim attitude, throttle 0, de=trim"))
    # (b) throttle 0, all controls 0
    rows.append(scan_case(name, ac, st, ctrl(), 300.0, "trim attitude, all controls 0"))
    # (c) large initial rates
    for om in ([1.0, 0.5, -0.3], [3.0, 2.0, 1.5]):
        st2 = st._replace(omega=jnp.array(om))
        rows.append(scan_case(name, ac, st2, ctrl(), 120.0, f"omega0={om}"))
    # (d) extreme attitudes
    for (phi, th, psi) in [(0.0, 1.2, 0.0), (np.pi, 0.0, 0.0), (0.5, -1.4, 1.0)]:
        st3 = st._replace(quat=euler_to_quat(jnp.array(phi), jnp.array(th), jnp.array(psi)))
        rows.append(scan_case(name, ac, st3, ctrl(), 120.0,
                              f"euler=({phi:.2f},{th:.2f},{psi:.2f})"))
    # (e) full elevator / aileron held, throttle 0
    rows.append(scan_case(name, ac, st, ctrl(de=float(ac.elevator_limit)), 120.0,
                          "full +elevator, thr 0"))
    rows.append(scan_case(name, ac, st, ctrl(de=-float(ac.elevator_limit)), 120.0,
                          "full -elevator, thr 0"))
    rows.append(scan_case(name, ac, st, ctrl(da=float(ac.aileron_limit)), 120.0,
                          "full aileron, thr 0"))
    rows.append(scan_case(name, ac, st, ctrl(dr=float(ac.rudder_limit)), 120.0,
                          "full rudder, thr 0"))

hdr = f"{'aircraft':20s} {'case':38s} {'status':9s} {'dE_above_E0/E0':>15s} {'worstW':>12s} {'posfrac':>8s}"
print("\n" + hdr)
print("-" * len(hdr))
for r in rows:
    if r["status"] != "ok":
        print(f"{r['name']:20s} {r['label']:38s} {r['status']:9s}")
        continue
    print(f"{r['name']:20s} {r['label']:38s} {r['status']:9s} "
          f"{r['above_frac']:15.3e} {r['worst_step_gain_W']:12.4g} {r['pos_frac']:8.4f}")

bad = [r for r in rows if r["status"] == "ok" and r["above_frac"] > 1e-12]
print(f"\ncases with energy above initial by >1e-12 relative: {len(bad)} / {len(rows)}")
for r in sorted(bad, key=lambda r: -r["above_frac"])[:10]:
    print(f"  {r['name']:20s} {r['label']:38s} above={r['above_initial']:.4g} J "
          f"({r['above_frac']:.3e})  worst step power {r['worst_step_gain_W']:.4g} W")
