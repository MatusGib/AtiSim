"""A5b: trim uniqueness, with the 2*pi alpha aliases folded out."""
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import trim
from flightsim.aircraft import CRUISE, REGISTRY
from flightsim.dynamics import derivatives

TWO_PI = 2.0 * np.pi

ALPHAS = np.radians(np.array([-90., -60., -30., -20., -15., -10., -8., -5., -3., 0.,
                              3., 5., 8., 10., 15., 20., 30., 60., 90.]))
ELEVS = np.array([-0.6, -0.35, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.35, 0.6])
THROT = np.array([-1.0, -0.2, 0.0, 0.1, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0])
guesses = jnp.array([[a, e, t] for a in ALPHAS for e in ELEVS for t in THROT])
print(f"grid {len(ALPHAS)}x{len(ELEVS)}x{len(THROT)} = {len(guesses)} starts per aircraft")


def wrap(a):
    """alpha modulo 2*pi into (-pi, pi]."""
    return (np.asarray(a) + np.pi) % TWO_PI - np.pi


for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac = REGISTRY[name]
    xs, rs = jax.vmap(lambda g: trim.trim(jnp.array(V), jnp.array(H), ac, guess=g))(guesses)
    xs = np.asarray(xs); rn = np.linalg.norm(np.asarray(rs), axis=1)
    ok = np.isfinite(xs).all(axis=1) & (rn < 1e-9)

    # fold alpha into (-pi, pi]; elevator/throttle enter linearly so they do NOT alias
    folded = xs.copy()
    folded[:, 0] = wrap(xs[:, 0])
    clusters = []
    for i in np.where(ok)[0]:
        r = folded[i]
        for c in clusters:
            if np.max(np.abs(r - c["x"]) / np.maximum(1.0, np.abs(c["x"]))) < 1e-6:
                c["n"] += 1; c["idx"].append(i); break
        else:
            clusters.append({"x": r, "n": 1, "idx": [i]})
    clusters.sort(key=lambda c: -c["n"])
    print(f"\n{'='*74}\n{name}  V={V:.3f} m/s  H={H:.1f} m   converged {ok.sum()}/{len(xs)}")
    print(f"PHYSICALLY DISTINCT roots (alpha folded mod 2pi): {len(clusters)}")
    print(f"  elevator_limit = +-{float(ac.elevator_limit):.3f} rad, "
          f"throttle is physical on [0, 1]")
    for k, c in enumerate(clusters):
        a, e, t = c["x"]
        # verify by re-evaluating the residual at the folded root
        res = derivatives(trim.trimmed_state(jnp.array(a), jnp.array(V), jnp.array(H)),
                          trim.trimmed_controls(jnp.array(e), jnp.array(t)),
                          ac, jnp.zeros(3), jnp.zeros(3))
        rr = float(jnp.linalg.norm(jnp.array([res.vel_body[0], res.vel_body[2], res.omega[1]])))
        flags = []
        if abs(a) > np.radians(15.0):
            flags.append("alpha>15deg (is_physical=False)")
        if abs(e) > float(ac.elevator_limit):
            flags.append(f"|elevator|>{float(ac.elevator_limit):.2f} rad")
        if not (0.0 <= t <= 1.0):
            flags.append("throttle outside [0,1]")
        g = np.asarray(guesses)[c["idx"]]
        print(f"  root {k}: alpha={np.degrees(a):+10.4f} deg  de={e:+.6f} rad "
              f"({np.degrees(e):+8.3f} deg)  thr={t:+12.6f}")
        print(f"           basin {c['n']:>5d}/{len(xs)}  |residual at folded root| = {rr:.3e}")
        print(f"           guess alpha {np.degrees(g[:,0]).min():+6.1f}..{np.degrees(g[:,0]).max():+6.1f} deg,"
              f"  de {g[:,1].min():+.2f}..{g[:,1].max():+.2f},  thr {g[:,2].min():+.2f}..{g[:,2].max():+.2f}")
        print(f"           VERDICT: {'PHYSICAL' if not flags else 'ABSURD -- ' + '; '.join(flags)}")

    # how close to the default guess does the basin boundary lie?
    print("  -- basin boundary probe around the default guess [0.05, 0.0, 0.5] --")
    good = None
    for a0 in np.radians(np.arange(-90, 95, 5.0)):
        g = jnp.array([a0, 0.0, 0.5])
        xg, rg = trim.trim(jnp.array(V), jnp.array(H), ac, guess=g)
        aw = wrap(float(xg[0]))
        phys = abs(aw) <= np.radians(15.0)
        if good is None:
            good = (np.degrees(aw), phys)
        if not phys or abs(np.degrees(aw) - good[0]) > 1e-6:
            print(f"     guess alpha {np.degrees(a0):+6.1f} deg -> converged alpha "
                  f"{np.degrees(aw):+10.4f} deg (folded), thr={float(xg[2]):+.4f} "
                  f"{'<-- DIFFERENT ROOT' if abs(np.degrees(aw)-good[0])>1e-6 else ''}")
    # trim jacobian conditioning at the nominal root
    xd, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    J = np.asarray(jax.jacfwd(trim.residual)(xd, jnp.array(V), jnp.array(H), ac))
    print(f"  Newton Jacobian at the nominal root: cond = {np.linalg.cond(J):.4e}")
    print(f"     J = \n{np.array2string(J, precision=6)}")
