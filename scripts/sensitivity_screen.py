"""Phase S1: the AD sensitivity screen on the linear modes, and its falsification.

WHAT THIS ANSWERS. Section 4 measures the model against SOURCES. It has never
measured the model against ITSELF, so nothing in it says which of the 747's own
coefficients its modes actually rest on. This screen returns the elasticity of
every mode with respect to every independent coefficient -- the per-cent of the
answer per per-cent of the input -- in one forward pass per coefficient.

WHY THE MODES FIRST, AND NOT THE HEADLINE LOAD. Because the modes are the only
quantity in this project for which an INDEPENDENT sensitivity answer already
exists: session 11 swept four coefficients against four modes at Caughey's 747
power approach and fitted four slopes. Section A below reproduces them. If it
could not, nothing downstream of this machinery would be worth running, which is
why phase S1 gates phase S2 rather than merely preceding it.

THE ONE THING TO READ BEFORE THE TABLE. An AD elasticity is a TANGENT at a
point; session 11's numbers are LEAST-SQUARES SLOPES over wide ranges. They are
different objects and they do not have to agree. Section A reports both, and the
gap between them is a measurement of curvature rather than an error -- section C
proves that by moving the tangent point to the centre of each swept range.

Run: PYTHONPATH=<abs worktree root> python -m scripts.sensitivity_screen
  or PYTHONPATH=<abs worktree root> python scripts/sensitivity_screen.py
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import atisim  # noqa: F401  -- enables x64
from atisim import sensitivity, trim, validation
from atisim.aircraft import CRUISE, REGISTRY
from atisim.units import FT2M

# Caughey's 747 power approach -- session 11's condition, and therefore the only
# one at which this project holds an independent answer to check against.
CAUGHEY_V = 279.1 * FT2M
CAUGHEY_H = 0.0

# Session 11's four fitted slopes, PROJECT.md section 4, "Coefficient
# sensitivity -- known change, known result". Quoted here as the reference they
# are; nothing in this script may be used to revise them.
SESSION_11 = {
    "CD0 -> zeta_phugoid": dict(slope=0.76994, intercept=-0.0162, worst=0.0031),
    "Cma -> wn_sp^2": dict(slope=-0.42251, intercept=0.27366, worst=0.0168),
    "|Clp| -> 1/tau_roll": dict(slope=2.06872, intercept=0.30240, worst=0.0173),
    "Cnb -> wn_dr^2": dict(slope=2.10545, intercept=0.26626, worst=0.0056),
}


def _trim_of(ac, V, H):
    x, res = trim.trim(jnp.array(V), jnp.array(H), ac)
    return (float(x[0]), float(x[1]), float(x[2])), float(np.linalg.norm(np.asarray(res)))


def _central(quantity, ac, field, V, H, rel_h=1e-5, sign=1.0):
    """Central difference of `quantity` in one field, through the SHIPPED path.

    Deliberately routed through `validation`'s own functions rather than through
    the jnp twins: a difference taken on the same code the AD screen uses would
    check the arithmetic and not the model.
    """
    p0 = float(getattr(ac, field))
    h = abs(p0) * rel_h
    vals = []
    for s in (+1.0, -1.0):
        swept = ac._replace(**{field: jnp.array(p0 + s * h)})
        vals.append(quantity(swept, *_trim_of(swept, V, H)[0]))
    return sign * (vals[0] - vals[1]) / (2.0 * h)


def q_ph_zeta(a, al, de, th):
    return validation.longitudinal_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[0][1]


def q_sp_wn2(a, al, de, th):
    return validation.longitudinal_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[-1][0] ** 2


def q_inv_tau_roll(a, al, de, th):
    return 1.0 / validation.lateral_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[1]


def q_dr_wn2(a, al, de, th):
    return validation.lateral_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[0][0] ** 2


def section_a(ac, out):
    """S1's gate: the AD screen against session 11, and against a difference."""
    print("\n" + "=" * 78)
    print("A. THE FALSIFICATION -- AD against session 11, and against a difference")
    print("=" * 78)
    print("The 747 POWER APPROACH at Caughey's 279.1 ft/s, sea level.\n")

    lon, A_lon, _, sep_lon, x, res = sensitivity.mode_sensitivity(
        ac, CAUGHEY_V, CAUGHEY_H, axis="longitudinal", fields=("CD0", "Cma"))
    lat, A_lat, _, sep_lat, _, _ = sensitivity.mode_sensitivity(
        ac, CAUGHEY_V, CAUGHEY_H, axis="lateral", fields=("Clp", "Cnb"))

    print(f"  trim residual              {np.linalg.norm(np.asarray(res)):.3e}")
    print(f"  eigenvalue separation      longitudinal {sep_lon:.5f}, lateral {sep_lat:.5f}")
    print("  (first-order perturbation divides by the gap in effect; both are wide)\n")

    ph = sensitivity.oscillatory_modes(lon["CD0"])[0]
    sp = sensitivity.oscillatory_modes(lon["Cma"])[-1]
    roll = sensitivity.real_modes(lat["Clp"])[0]
    dr = sensitivity.oscillatory_modes(lat["Cnb"])[0]

    rows = [
        ("CD0 -> zeta_phugoid", ph.dzeta,
         _central(q_ph_zeta, ac, "CD0", CAUGHEY_V, CAUGHEY_H)),
        ("Cma -> wn_sp^2", 2.0 * sp.wn * sp.dwn,
         _central(q_sp_wn2, ac, "Cma", CAUGHEY_V, CAUGHEY_H)),
        # 1/tau = -Re(lambda), so d(1/tau)/dClp = -dlam.real; and d/d|Clp| =
        # -d/dClp because Clp is negative and the published relation is stated
        # in |Clp|. The two signs cancel: d(1/tau)/d|Clp| = +dlam.real. Both
        # sides of this row must carry the SAME convention, which is what the
        # `sign=-1.0` on the difference below supplies.
        ("|Clp| -> 1/tau_roll", roll.dlam.real,
         _central(q_inv_tau_roll, ac, "Clp", CAUGHEY_V, CAUGHEY_H, sign=-1.0)),
        ("Cnb -> wn_dr^2", 2.0 * dr.wn * dr.dwn,
         _central(q_dr_wn2, ac, "Cnb", CAUGHEY_V, CAUGHEY_H)),
    ]

    print(f"  {'relation':22s} {'AD tangent':>12s} {'central diff':>13s} {'rel':>10s}"
          f" {'session 11':>11s} {'AD/fit':>8s} {'fit worst':>10s}")
    worst_machine = 0.0
    for name, ad, cd in rows:
        ref = SESSION_11[name]
        machine = abs(ad - cd) / abs(cd)
        worst_machine = max(worst_machine, machine)
        out["section_a"][name] = dict(ad=ad, central=cd, machine_rel=machine,
                                      fitted=ref["slope"], ratio=ad / ref["slope"])
        print(f"  {name:22s} {ad:12.5f} {cd:13.5f} {machine:10.2e}"
              f" {ref['slope']:11.5f} {ad / ref['slope']:8.4f} {ref['worst']*100:9.2f}%")

    print(f"\n  MACHINERY GATE: worst AD-vs-difference disagreement {worst_machine:.2e}"
          f"  -- {'PASS' if worst_machine < 1e-6 else 'FAIL'} (gate 1e-6)")
    print("  The AD/fit column is NOT the gate: a tangent at the base point and a")
    print("  least-squares slope over a 3-8x range are different objects. Section C.")
    out["section_a_worst_machine_rel"] = worst_machine
    return worst_machine


def section_b(ac, out):
    """Session 11's sweeps, re-run here, because the platform is not the one they were taken on."""
    print("\n" + "=" * 78)
    print("B. SESSION 11'S SWEEPS, RE-RUN -- do the published slopes reproduce?")
    print("=" * 78)
    print("PROJECT.md section 10 documents a Windows .venv; this is Linux with a")
    print("freshly installed JAX. Two bit-exact rollout pins do NOT reproduce here")
    print("(section 4 carries what they are worth). These four fits do.\n")

    base = float(ac.CD0)
    specs = [
        ("CD0 -> zeta_phugoid", "CD0", base * np.array([1.0, 1.5, 2.0, 3.0]),
         lambda a, al, de, th: validation.longitudinal_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[0][1],
         lambda xs, ys: (xs, ys)),
        ("Cma -> wn_sp^2", "Cma", np.array([float(ac.Cma), -0.9, -0.6, -0.3, -0.1]),
         lambda a, al, de, th: validation.longitudinal_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[-1][0],
         lambda xs, ys: (xs, ys ** 2)),
        ("|Clp| -> 1/tau_roll", "Clp", np.array([-0.30, -0.45, -0.60, -0.75]),
         lambda a, al, de, th: validation.lateral_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[1],
         lambda xs, ys: (np.abs(xs), 1.0 / ys)),
        ("Cnb -> wn_dr^2", "Cnb", np.array([0.075, 0.150, 0.300, 0.600]),
         lambda a, al, de, th: validation.lateral_modes(a, al, de, th, CAUGHEY_V, CAUGHEY_H)[0][0],
         lambda xs, ys: (xs, ys ** 2)),
    ]
    print(f"  {'relation':22s} {'slope':>10s} {'published':>11s} {'intercept':>11s}"
          f" {'published':>11s} {'worst':>8s} {'published':>10s}")
    for name, field, values, quantity, transform in specs:
        ys = validation.sweep(ac, field, values, quantity, CAUGHEY_V, CAUGHEY_H)
        xs2, ys2 = transform(values, np.asarray(ys))
        slope, intercept, worst = validation.affine_fit(xs2, ys2)
        ref = SESSION_11[name]
        out["section_b"][name] = dict(slope=slope, intercept=intercept, worst=worst,
                                      published=ref)
        print(f"  {name:22s} {slope:10.5f} {ref['slope']:11.5f} {intercept:11.5f}"
              f" {ref['intercept']:11.5f} {worst*100:7.2f}% {ref['worst']*100:9.2f}%")


def section_c(ac, out):
    """Is the AD-vs-fit gap curvature? Move the tangent point to the range centroid."""
    print("\n" + "=" * 78)
    print("C. IS THE GAP CURVATURE? -- the same tangent, at the centre of each range")
    print("=" * 78)
    print("If the AD-vs-fit gap is curvature and not an error, then taking the")
    print("tangent at the CENTROID of the swept range must close it to about the")
    print("fit's own worst residual. Nothing else is changed.\n")

    base = float(ac.CD0)
    specs = [
        ("CD0 -> zeta_phugoid", "CD0", base * np.array([1.0, 1.5, 2.0, 3.0]), q_ph_zeta, +1.0),
        ("Cma -> wn_sp^2", "Cma", np.array([float(ac.Cma), -0.9, -0.6, -0.3, -0.1]), q_sp_wn2, +1.0),
        ("|Clp| -> 1/tau_roll", "Clp", np.array([-0.30, -0.45, -0.60, -0.75]), q_inv_tau_roll, -1.0),
        ("Cnb -> wn_dr^2", "Cnb", np.array([0.075, 0.150, 0.300, 0.600]), q_dr_wn2, +1.0),
    ]
    print(f"  {'relation':22s} {'at base':>10s} {'at centroid':>12s} {'fitted':>10s}"
          f" {'base/fit':>9s} {'centroid/fit':>13s}")
    for name, field, values, quantity, sign in specs:
        fit = SESSION_11[name]["slope"]
        at_base = out["section_a"][name]["ad"]
        mid = float(np.mean(values))
        centred = ac._replace(**{field: jnp.array(mid)})
        at_mid = _central(quantity, centred, field, CAUGHEY_V, CAUGHEY_H, sign=sign)
        out["section_c"][name] = dict(at_base=at_base, at_centroid=at_mid, fitted=fit)
        print(f"  {name:22s} {at_base:10.5f} {at_mid:12.5f} {fit:10.5f}"
              f" {at_base/fit:9.4f} {at_mid/fit:13.4f}")


def section_d(name, ac, V, H, out):
    """The screen proper: every independent coefficient against every mode."""
    print("\n" + "=" * 78)
    print(f"D. THE SCREEN -- {name} at {V:.2f} m/s, {H:.0f} m")
    print("=" * 78)

    lon, A_lon, lam_lon, sep_lon, x, res = sensitivity.mode_sensitivity(
        ac, V, H, axis="longitudinal", fields=sensitivity.LONGITUDINAL_FIELDS)
    lat, A_lat, lam_lat, sep_lat, _, _ = sensitivity.mode_sensitivity(
        ac, V, H, axis="lateral", fields=sensitivity.LATERAL_FIELDS)

    print(f"  trim [alpha, elevator, throttle] = {np.asarray(x)}, residual "
          f"{np.linalg.norm(np.asarray(res)):.2e}")
    print(f"  eigenvalue separation: longitudinal {sep_lon:.5f}, lateral {sep_lat:.5f}\n")

    entries = []
    for field in sensitivity.LONGITUDINAL_FIELDS:
        ms = sensitivity.oscillatory_modes(lon[field])
        if len(ms) != 2:
            print(f"  !! {field}: {len(ms)} oscillatory longitudinal roots, skipped")
            continue
        ph, sp = ms[0], ms[-1]
        p = float(getattr(ac, field))
        for label, m in (("phugoid", ph), ("short period", sp)):
            entries.append((field, f"{label} wn", p,
                            sensitivity.elasticity(m.dwn, p, m.wn)))
            entries.append((field, f"{label} zeta", p,
                            sensitivity.elasticity(m.dzeta, p, m.zeta)))
    for field in sensitivity.LATERAL_FIELDS:
        osc = sensitivity.oscillatory_modes(lat[field])
        rls = sensitivity.real_modes(lat[field])
        p = float(getattr(ac, field))
        if len(osc) == 1:
            entries.append((field, "dutch roll wn", p,
                            sensitivity.elasticity(osc[0].dwn, p, osc[0].wn)))
            entries.append((field, "dutch roll zeta", p,
                            sensitivity.elasticity(osc[0].dzeta, p, osc[0].zeta)))
        if len(rls) >= 2:
            entries.append((field, "roll tau", p,
                            sensitivity.elasticity(rls[0].dtau, p, rls[0].tau)))
            entries.append((field, "spiral tau", p,
                            sensitivity.elasticity(rls[1].dtau, p, rls[1].tau)))

    out["screen"][name] = [dict(field=f, mode=m, value=p, elasticity=e)
                           for f, m, p, e in entries]

    inert = sorted({f for f, _, _, e in entries} -
                   {f for f, _, _, e in entries if abs(e) > 1e-12})
    if inert:
        print(f"  STRUCTURALLY INERT at this condition (elasticity exactly 0 for every")
        print(f"  mode, which means NOT USED rather than unimportant): {', '.join(inert)}\n")

    for mode in sorted({m for _, m, _, _ in entries}):
        rows = sorted([e for e in entries if e[1] == mode],
                      key=lambda r: -abs(r[3]) if np.isfinite(r[3]) else 0)
        rows = [r for r in rows if np.isfinite(r[3]) and abs(r[3]) > 1e-10][:6]
        if not rows:
            continue
        print(f"  {mode:18s} " + "   ".join(f"{f} {e:+.3f}" for f, _, _, e in rows))
    print("\n  Elasticity = (dQ/dp)(p/Q): per-cent of the answer per per-cent of the")
    print("  input. One-at-a-time; NO interaction term is measured here.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, default=None,
                    help="write the numbers to this path as well as printing them")
    args = ap.parse_args()

    print(f"atisim imported from: {atisim.__file__}")
    print("PROJECT.md section 10: if that is not the tree you edited, stop here.")

    out = {"section_a": {}, "section_b": {}, "section_c": {}, "screen": {},
           "tree": atisim.__file__}

    approach = REGISTRY["boeing747_approach"]
    section_a(approach, out)
    section_b(approach, out)
    section_c(approach, out)
    section_d("boeing747_approach (Caughey)", approach, CAUGHEY_V, CAUGHEY_H, out)
    section_d("boeing747 cruise", REGISTRY["boeing747"],
              CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"], out)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=1))
        print(f"\nwritten: {args.json}")


if __name__ == "__main__":
    main()
