"""A5: trim convergence rate and root uniqueness."""
import itertools
import numpy as np
import jax, jax.numpy as jnp
import flightsim
from flightsim import trim, verification
from flightsim.aircraft import CRUISE, REGISTRY

np.set_printoptions(precision=10, suppress=True)

print("=" * 78)
print("### 5a. NEWTON CONVERGENCE RATE (verification.newton_residual_history)")
print("=" * 78)
for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac = REGISTRY[name]
    h = verification.newton_residual_history(jnp.array(V), jnp.array(H), ac, 10)
    print(f"\n{name}  (V={V:.3f} m/s, H={H:.1f} m)  start = {np.asarray(trim.INITIAL_GUESS)}")
    for i, r in enumerate(h):
        print(f"   iter {i:>2d}: |residual| = {r:.6e}")
    usable = np.array(list(itertools.takewhile(lambda r: r > 1e-13, h)))
    if len(usable) >= 3:
        e = np.log10(usable)
        ratios = (e[2:] - e[1:-1]) / (e[1:-1] - e[:-2])
        print(f"   log-residual exponent ratios (2 = quadratic): {np.round(ratios, 4)}")
    else:
        print(f"   only {len(usable)} usable iterates above 1e-13")
    # how many iterations does `trim` actually need?  it runs a fixed 40.
    for it in (1, 2, 3, 4, 5, 6, 8, 10, 40, 200):
        x, r = trim.trim(jnp.array(V), jnp.array(H), ac, iterations=it)
        print(f"   trim(iterations={it:>3d}) -> alpha={float(x[0]):+.12f} "
              f"de={float(x[1]):+.12f} thr={float(x[2]):+.12f} |res|={float(jnp.linalg.norm(r)):.3e}")

print()
print("=" * 78)
print("### 5b. ROOT UNIQUENESS: sweep the initial guess on a wide grid")
print("=" * 78)
ALPHAS = np.radians(np.array([-60., -30., -15., -8., -3., 0., 3., 5., 8., 15., 30., 60., 90.]))
ELEVS = np.array([-0.6, -0.35, -0.2, -0.05, 0.0, 0.05, 0.2, 0.35, 0.6])
THROT = np.array([-1.0, -0.2, 0.0, 0.2, 0.5, 0.8, 1.0, 2.0, 5.0])
print(f"grid: {len(ALPHAS)} alphas x {len(ELEVS)} elevators x {len(THROT)} throttles "
      f"= {len(ALPHAS)*len(ELEVS)*len(THROT)} starts per aircraft")
print(f"alphas (deg): {np.degrees(ALPHAS)}")
print(f"elevators   : {ELEVS}")
print(f"throttles   : {THROT}")

guesses = jnp.array([[a, e, t] for a in ALPHAS for e in ELEVS for t in THROT])


def solve_batch(V, H, ac, guesses, iterations=40):
    f = jax.vmap(lambda g: trim.trim(jnp.array(V), jnp.array(H), ac, guess=g,
                                     iterations=iterations))
    return f(guesses)


for name in REGISTRY:
    V, H = CRUISE[name]["airspeed"], CRUISE[name]["altitude"]
    ac = REGISTRY[name]
    xs, rs = solve_batch(V, H, ac, guesses)
    xs = np.asarray(xs); rs = np.asarray(rs)
    rn = np.linalg.norm(rs, axis=1)
    finite = np.isfinite(xs).all(axis=1) & np.isfinite(rn)
    converged = finite & (rn < 1e-9)
    print(f"\n--- {name}: V={V:.3f} m/s H={H:.1f} m ---")
    print(f"  starts               : {len(xs)}")
    print(f"  non-finite result    : {int((~finite).sum())}")
    print(f"  |residual| >= 1e-9   : {int((finite & ~converged).sum())}")
    print(f"  converged (<1e-9)    : {int(converged.sum())}")

    # cluster the converged roots
    roots = xs[converged]
    clusters = []
    labels = np.full(len(roots), -1)
    for i, r in enumerate(roots):
        for j, c in enumerate(clusters):
            if np.max(np.abs(r - c[0])) < 1e-6:
                c[1].append(i); labels[i] = j; break
        else:
            clusters.append((r, [i])); labels[i] = len(clusters) - 1
    print(f"  DISTINCT converged roots (tol 1e-6): {len(clusters)}")
    order = np.argsort([-len(c[1]) for c in clusters])
    for k in order:
        c, members = clusters[k]
        alpha_deg = np.degrees(c[0])
        phys = trim.is_physical(jnp.array(c))
        # basin: which guesses landed here
        gidx = np.where(converged)[0][members]
        g = np.asarray(guesses)[gidx]
        print(f"    root {k}: alpha={alpha_deg:+10.4f} deg  de={c[1]:+.6f} rad "
              f"({np.degrees(c[1]):+8.3f} deg)  thr={c[2]:+.6f}   "
              f"basin={len(members):>4d}/{len(xs)}  is_physical={phys}")
        print(f"             guess-alpha range {np.degrees(g[:,0]).min():+.1f}..{np.degrees(g[:,0]).max():+.1f} deg, "
              f"guess-thr range {g[:,2].min():+.2f}..{g[:,2].max():+.2f}")
    # anything reached by the DEFAULT guess?
    xd, rd = trim.trim(jnp.array(V), jnp.array(H), ac)
    print(f"  default INITIAL_GUESS -> alpha={np.degrees(float(xd[0])):+.6f} deg, "
          f"de={float(xd[1]):+.6f}, thr={float(xd[2]):+.6f}, |res|={float(jnp.linalg.norm(rd)):.3e}, "
          f"is_physical={trim.is_physical(xd)}")
