# Closing the verification gaps — plan and handover

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** close the gaps session 11's verification pass identified, in the order their
consequences bite, and leave the assumption register (`docs/ASSUMPTIONS.md`) true.

**Status at handover:** session 11 is merged to `master` at `0c8ae9c`. 318 tests + 1
skipped, plus a 12-cell notebook gate. Nothing in the validated baseline moved.

---

## Part 1 — Handover: what session 11 did, for a session that was not there

### The ask

External review said, in substance: *establish the solver's validity before adding more
layers, ideally through a notebook where known coefficient changes give known results from
the literature.* A follow-up asked whether a 1972 source document is itself a source of
error, and that all underlying assumptions be well founded.

### What was built

| Artifact | What it is |
|---|---|
| `atisim/verification.py` | **tier 0** — checks that take no aircraft data as a reference |
| `atisim/validation.py` | **tiers 1–2** — analytic laws and published worked examples; also now holds the linearisation, moved out of `tests/modes.py` (which is a re-export, so the two baseline mode files are untouched) |
| `atisim/tests/test_verification.py` | 8 tests |
| `atisim/tests/test_validation.py` | 14 tests |
| `notebooks/solver-validation.ipynb` | thin front end, no arithmetic, run by `pytest --nbval-lax notebooks/` |
| `docs/ASSUMPTIONS.md` | the assumption register, with a measured bound on each |
| `atisim/integrate.py` | `rk4_step` split out of `step`, bit-identically |

### The answer on source vintage, which is worth carrying forward

Age was the wrong axis. Every tier-2 check is **closed-loop against a document's own
arithmetic**: if CR-2144's derivatives were 10% away from the real aeroplane, the solver
must *still* reproduce CR-2144's own transfer-function factors from CR-2144's own
derivatives. The design spec ranks every check by how much it depends on any source at all
(`docs/superpowers/specs/2026-08-11-solver-validation-design.md`, "Source qualification").

The real risks are in `ASSUMPTIONS.md`, and the largest is that CR-2144's §IX data is the
**flexible** airframe against a rigid-body model — a mismatch that matters far more than
1972 and had not been recorded anywhere.

### What was found — none of it in the solver

The core passed: RK4 is fourth-order (3.99982 / 3.98913), Galilean invariance holds to
1e-11, the Newton trim solve converges quadratically, torque-free rotation matches its
Jacobi elliptic closed form to 1e-8, and Caughey's independently computed plant matrix is
reproduced element by element with the omitted α̇ terms **reconstructed** to 0.03%.

Three defects surfaced, all in documentation or measurement:

1. **`PROJECT.md` §5 had misdescribed the model since session 1.** It claimed `Xu, Zu, Mu`
   were excluded. The model *has* Xu and Zu — they fall out of dynamic-pressure variation,
   and A[0,0] lands within 1.2% of Caughey's Xu with none entered anywhere. What is missing
   is their **Mach content**, which is why the phugoid error is 17.8% at M 0.80 and 0.4% at
   M 0.25 from identical code.
2. **The first order-of-accuracy window fitted through the round-off floor** and read 3.82.
3. **`trim.trim` converges to physically absurd roots** — CLa = 0.1 gives α = −633° at a
   residual of 1.6e-15.

And three of four planned sensitivity laws were the wrong functional form. Everything is
**affine with a non-zero intercept**, and the intercept is the term the textbook
approximation drops. A test asserting ωn_sp → 0 at the neutral point was wrong physics; the
model kept its `Zα·Mq/u₀` term and was right.

### Two things a future session will want and should not re-derive

- **CR-2144 documents ten aircraft, not one:** NT-33A, F-104A, F-4C, X-15, HL-10,
  **Jetstar**, **CV-880M**, B-747, **C-5A**, XB-70A — each with derivatives,
  transfer-function factors and handling-qualities parameters under the same Appendix
  A/B/C conventions this code implements. The cheapest next two are the **Jetstar**
  (Table VII-1, power approach, non-dimensional, **body axis**) and the **C-5A**
  (Table X-1, same form, **stability axis**, so `stability_to_body` applies). Both are in
  the identical already-non-dimensional form as Table IX-2.
- **The 747's other flight conditions are expensive.** CR-2144 pp. 229–236 are scanned
  line-printer output whose text layer OCRs to noise, and the cruise non-dimensional
  derivatives are published as **plots against Mach**, not tables.

---

## Part 2 — The plan

Ordered by consequence, not by effort. Tasks 1 and 2 are the ones that change what the
project can claim.

### Conventions

Working directory is the repo root. `$PY` is
`"C:/Users/mateusz/UROP/Claude_Flight_Sim/.venv/Scripts/python.exe"`. From a **worktree**,
every command needs `PYTHONPATH` set — and the notebook gate needs it **absolute**, because
nbval starts its kernel with cwd in `notebooks/`.

Two standing rules from `PROJECT.md`: **do not edit a tolerance to make a test pass**, and
**flag, never invent**.

---

### Task 1: Test the `−m·dW/dt` seam

`PROJECT.md` §2 names two classic gust-modelling errors. Session 11's Galilean test catches
the first (substituting `vel_rel` into the Coriolis term). It **structurally cannot** catch
the second, because a steady uniform wind has zero material derivative. This is the seam
Dryden will load, so it should be closed before Dryden, not after.

**Files:** Modify `atisim/tests/test_verification.py`

- [ ] **Step 1: Write the failing test**

```python
def test_a_time_varying_uniform_wind_adds_no_spurious_force():
    """The other error PROJECT.md section 2 names, which the steady-wind test cannot see.

    A uniform wind that changes with time is still uniform in SPACE, so at every
    instant the aerodynamics see a relative flow identical to a still-air flight
    whose ground velocity is offset by W(t). The aircraft feels the wind only
    through that relative flow -- there is no -m dW/dt body force, because the
    air mass accelerating does not push on the aeroplane, it only changes the
    flow the wings see.

    So: fly still air, and fly a time-varying uniform wind from a start state
    offset by W(0). If an -m dW/dt term has been added anywhere, the two
    trajectories' ATTITUDE and RATES diverge. If it has not, they cannot.

    The steady-wind test asserts a stronger position relation (pos differs by
    exactly W*t) that does NOT hold here, because integrating a varying W(t)
    gives a different displacement. Attitude and rates are the invariant.
    """
    import jax
    from atisim.state import quat_to_dcm

    W0 = jnp.array([9.0, -4.0, 0.0])
    OMEGA = 0.7  # rad/s, several times the phugoid, comparable to the short period

    def varying_wind(wind_state, state, key, dt):
        # WindState is an empty tuple today, so time is carried by neither the
        # carry nor the signature. Derive it from along-track distance instead:
        # the aircraft's north position over its trim speed. Crude but monotone,
        # and all this test needs is a wind that CHANGES.
        phase = state.pos_ned[0] / 235.92
        return W0 * jnp.cos(OMEGA * phase), jnp.zeros(3), wind_state, key

    ac = REGISTRY["boeing747"]
    V, H = CRUISE["boeing747"]["airspeed"], CRUISE["boeing747"]["altitude"]
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    state = trim.trimmed_state(x[0], jnp.array(V), jnp.array(H))
    controls = trim.trimmed_controls(x[1] + 0.01, x[2])

    dcm = quat_to_dcm(state.quat)
    shifted = state._replace(vel_body=state.vel_body + dcm.T @ W0)

    dt, n = jnp.array(0.02), 1000
    _, still = integrate.rollout(
        integrate.init_sim(state, jax.random.PRNGKey(0)), controls, dt, ac, n
    )
    _, blown = integrate.rollout(
        integrate.init_sim(shifted, jax.random.PRNGKey(0)),
        controls, dt, ac, n, wind_model=varying_wind,
    )

    np.testing.assert_allclose(
        np.asarray(blown.omega), np.asarray(still.omega), atol=1e-9
    )
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=. $PY -m pytest atisim/tests/test_verification.py -q -k time_varying
```

**Expect this to need thought, not to pass first time.** A spatially-uniform wind derived
from position is not exactly the same experiment as one derived from time — as the two
aircraft separate, they sample different phases. If the tolerance cannot be met, the
correct fix is to **add a time field to `WindState`** rather than to loosen it: that is
required for Dryden anyway (§7's extensibility table, item 1) and this test is the reason
to do it now rather than later.

- [ ] **Step 3: If a time field is needed, add it**

`wind.WindState` is an empty `NamedTuple` today. Give it a `t: Array` field, have
`zero_wind_state()` initialise it to 0.0, and have `integrate.step` advance it by `dt`.
`zero_wind` must return it untouched apart from the increment, and the existing
bit-identity tests for the zero-wind path must still pass — **that is the check that this
change was neutral**.

- [ ] **Step 4: Update `docs/ASSUMPTIONS.md` §E4 and `PROJECT.md` §8**

E4's verdict changes from "unverified" to whatever was measured. Remove the §8 open
question, or narrow it to whatever remains.

- [ ] **Step 5: Commit**

```bash
git add atisim/ docs/
git commit -m "Close the other gust seam: a time-varying wind adds no spurious force"
```

---

### Task 2: Guard `trim.trim` itself, not just `validation.sweep`

Session 11 put the |α| bound in `validation.sweep`, which is where it was needed. But the
defect is in `trim.trim` — it returns α = −633° with a residual of 1.6e-15 and says nothing
— and every other caller is equally exposed.

**Files:** Modify `atisim/trim.py`, `atisim/tests/test_trim.py`

- [ ] **Step 1: Decide the shape, because this one has a real trade-off**

`trim.trim` is `jax.jit`-ed and called inside `vmap` in `trim.minimum_drag_speed` and
elsewhere. **It therefore cannot raise.** Options:

1. Return a validity flag alongside the solution — `(x, residual, ok)`. Changes the public
   signature and every call site.
2. Leave `trim` alone and add `trim.check(x)` that callers may use, with `validation.sweep`
   and the scripts calling it. Smallest change; relies on callers remembering.
3. Return `NaN` in the solution when |α| exceeds a bound, so `jax_debug_nans` catches it in
   tests and it propagates loudly in production.

**Recommendation: option 2**, and say why in the docstring. Option 1 churns every call site
for a case that has never occurred with real data; option 3 turns a diagnosable condition
into a crash and fights `jax_debug_nans` in tests that legitimately sweep wide.

- [ ] **Step 2: Write the failing test**

```python
def test_trim_reports_a_physically_absurd_solution():
    """trim.trim converges to alpha = -633 deg for CLa = 0.1, at a residual of
    1.6e-15. Convergence and sense are different questions, and until session 11
    nothing in the project asked the second one."""
    ac = REGISTRY["boeing747_approach"]._replace(CLa=jnp.array(0.1))
    x, res = trim.trim(jnp.array(85.0), jnp.array(0.0), ac)
    assert float(jnp.linalg.norm(res)) < 1e-12, "this case converges; that is the point"
    assert not trim.is_physical(x), f"alpha {np.degrees(float(x[0])):.1f} deg passed"
```

- [ ] **Step 3: Implement `trim.is_physical`**

```python
# PROJECT.md section 7 puts the linear-aero ceiling at |alpha| ~ 10-12 deg and says
# a run outside it "is not evidence of anything". 15 deg is a little beyond the
# amber band, so a legitimate trim is never rejected.
ALPHA_LIMIT = math.radians(15.0)


def is_physical(x: Array) -> bool:
    """Is a trim solution a flight condition, as opposed to merely converged?

    NOT folded into `trim` itself, deliberately. `trim` is jitted and vmapped
    (see `minimum_drag_speed`), so it cannot raise, and returning a flag would
    churn every call site for a case that has never arisen with real aircraft
    data. This is a separate question, asked by the callers that sweep.
    """
    return bool(abs(float(x[0])) <= ALPHA_LIMIT)
```

- [ ] **Step 4: Point `validation.sweep` at it** so the bound lives in one place, and
      delete `validation.ALPHA_LIMIT`.

- [ ] **Step 5: Run the whole suite** — `test_trim.py` is a §4 baseline file.

```bash
PYTHONPATH=. $PY -m pytest -q
```

- [ ] **Step 6: Commit**

---

### Task 3: Record the newly bounded assumptions in `PROJECT.md` §5

`ASSUMPTIONS.md` now carries measured bounds that §5 does not mention. §5 is where a reader
looks for "what is this model not allowed to claim", so the two load-bearing ones belong
there too, with a pointer rather than a copy.

**Files:** Modify `docs/PROJECT.md` §5

- [ ] **Step 1: Add the vortex span-ratio caveat**

The Parks vortex core is **2.30–3.07 wingspans** for the 747, against 39.6 for the updraft
and 104.8 for the lee wave. The gust field is sampled at a point with a first-order
gradient correction, so the vortex is where that approximation is weakest — and the vortex
is the headline turbulence result. It compounds with §5's existing ±25% parameter band and
§2's note that the rotational gust already exceeds full aileron authority by ~1.5×.

- [ ] **Step 2: Add the constant-gravity bound**

`g = 9.80665` is **+0.383% high** at the 747's cruise altitude, and Lanchester maps that
1:1 into phugoid frequency. It is invisible inside the cruise phugoid's 17.8% gap, but it
is **the same order as the project's tightest agreements** (Dutch roll ωn 0.4%, spiral
0.8%, roll 0.9%). Any future sub-0.5% claim at altitude needs `g(h)` first.

- [ ] **Step 3: Add a pointer to `ASSUMPTIONS.md` at the head of §5**, and add the file to
      §2's architecture table.

- [ ] **Step 4: Commit**

---

### Task 4: Model `g(h)`, or decide not to

Task 3 records the bias. This decides what to do about it.

**Files:** `atisim/atmosphere.py`, `atisim/tests/test_atmosphere.py`, plus every
baseline that moves

- [ ] **Step 1: Measure the blast radius first, before changing anything**

```bash
PYTHONPATH=. $PY -c "
import jax.numpy as jnp, numpy as np, atisim
from atisim import trim, validation
from atisim.aircraft import REGISTRY, CRUISE
# how much do the 747 cruise modes move if g falls 0.383%?
"
```

Write the script to recompute the CR-2144 mode comparison with `G0` scaled by
`(R/(R+h))²` and report how each of the five modes moves.

- [ ] **Step 2: Decide, and record the decision either way**

**This is a genuine judgement call and the plan does not pre-empt it.** Adding `g(h)` is a
few lines, but `G0` is imported by `dynamics`, `trim`, `aero` and `vortex_viz`, and it
would move §4 baselines that are explicitly off-limits to feature work. The case for doing
it: the bias is systematic, altitude-dependent, and the same size as the agreements being
claimed. The case against: every existing result is quoted at one altitude per aircraft, so
a constant `g` is *exactly* right per-run, and the bias only matters for comparisons across
altitudes.

If the measured movement in Step 1 is below the tolerances in §4, **do not make the
change** — record the measurement in `ASSUMPTIONS.md` §A2 as the bound and close it.

- [ ] **Step 3: Commit the decision and its evidence**, whichever way it went.

---

### Task 5: Repository hygiene

Six `claude/*` branches are fully merged into `master` — zero commits ahead each — and four
worktrees sit under `.claude/worktrees/`, two on stale or detached refs.

- [ ] **Step 1: Confirm they are still fully merged**

```bash
for b in $(git branch --format='%(refname:short)' | grep '^claude/'); do
  printf "%-46s ahead:%s\n" "$b" "$(git rev-list --count master..$b)"
done
```

Every count must be 0. **If any is not, stop** — that branch has work that was never
merged, and deleting it would lose it.

- [ ] **Step 2: Check no live session owns a worktree** before removing it. These are
      harness-managed; removing one out from under a running session causes phantom state.

- [ ] **Step 3: Remove worktrees, then delete branches** — in that order, since
      `git branch -d` refuses while a worktree holds the branch.

```bash
git worktree remove <path>
git worktree prune
git branch -d <branch>
```

`-d` not `-D`: it refuses anything not fully merged, which is the safety property wanted
here.

---

## Part 3 — What is deliberately not in this plan

- **New aircraft.** The Jetstar and C-5A are costed in Part 1 and remain the cheapest next
  step, but they add breadth to a model whose assumptions are now the binding constraint,
  not its coverage.
- **Digitising CR-2144's derivative-vs-Mach plots.** It would bound `ASSUMPTIONS.md` §C3,
  the largest unbounded assumption — but chart reads are weaker evidence than the tabulated
  set already in use, and §7 declines them for that reason. Revisit only if a result is
  ever quoted far from a trim point.
- **A rigid-airframe derivative set.** Would close §B1, and no source the project holds can
  supply one.
- **Dryden.** Still §7 step 4, and still blocked on `WindState` being parameterised — which
  Task 1 may do as a side effect. If it does, say so in §7.
