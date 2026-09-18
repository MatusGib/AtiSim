# Next session — the third Fig. 8 cluster

Date written: 2026-08-07, at the end of session 6.

**This plan refines PROJECT.md §7 step 5. §7 remains the standing plan** — anything decided
here that changes the order or the content of the numbered steps must be written back into
§7, not left only here.

**Goal:** get the manoeuvring point onto the Fig. 8 comparison, so the discriminator has
the three clusters the paper has rather than two.

---

## Where things stand

Session 6 landed the free-air flying interface: wind through the live loop, a basic-T panel
with a flight-test overlay, a ramped stick and pitch trim, `panel.py` split out of `viz.py`,
and a `sense`/`accelerometers` jit fix worth 13.7 → 26.4 fps. 256 tests, all green, working
tree clean.

`vortex_viz.FIG8_REFERENCE` already carries `{"vortex": 1.4, "updraft": 6.2,
"manoeuvring": 12.0}`. The first two have model points. The third does not, and until it
does, "the timescale separation is the whole discriminator" is a claim about two categories
being different rather than about three falling where the paper puts them.

---

## Item 0 — the record is wrong, and this is owed

**Do this first.** PROJECT.md's own update rules were not followed at the end of session 6,
because the performance work and the PDF landed *after* the §9 entry was written. Three
concrete inconsistencies:

- [ ] **§8 asks a question that is now answered.** "Whether the panel still holds 20 fps"
      — it did not. Measured headless on Agg: 13.7 fps before the jit fix, 26.4 after.
      Replace the open question with the measurement, and leave a narrower one behind: the
      interactive TkAgg rate has still not been re-taken.
- [ ] **§10 quotes a superseded number.** "Measured on TkAgg: 19.9 fps, real-time ratio
      0.9994" predates the re-layout and is no longer true of this panel. §4's rule is
      supersede, never delete.
- [ ] **§4 has no timing rows.** Add them, with the method (medians of 120 runs, Agg,
      12-core machine, 7 Aug 2026):

      | Item | Before | After |
      |---|---|---|
      | jitted RK4 step | 0.114 ms | — |
      | `sense` | 6.56 ms eager | 0.028 ms jitted |
      | `accelerometers` | 10.70 ms eager | ~0.03 ms jitted |
      | blit, 14 axes / 72 artists | 28.6 ms | 30.1 ms |
      | whole frame | 73.0 ms | 37.8 ms |
      | achieved rate | 13.7 fps | 26.4 fps |

- [ ] **§9 needs a session 7 entry** covering the jit fix and the summary PDF, or the
      session 6 entry extending to cover them. Prefer a new entry — session 6's is already
      long and the perf work was a different investigation.

Verify: read §4, §8, §9 and §10 back and confirm no statement about frame rate contradicts
any other.

---

## Item 1 — the α gauge is one-sided, and it will lie during item 3

A defect introduced in session 6, found while probing item 3. `AlphaGauge` does:

```python
ax.set_xlim(0.0, ALPHA_SPAN_DEG)          # 0 to 15 deg
x = float(np.clip(self._deg, 0.0, ALPHA_SPAN_DEG))
if self._deg >= ALPHA_INVALID_DEG: return "invalid"
```

At α = −16° it pegs the needle at zero and reports **"linear"**. `aero.py` is
`CL = CL0 + CLa·α` with no stall, which is exactly odd-symmetric in Δα, so **|α| is what
decides validity, not α**. The gauge as built cannot see half the invalid range.

This matters here and not merely in principle: a pushdown drives α *negative*, and item 3's
run reaches −8.8° to −15.5° (measured, below). The instrument whose entire job is to say
"this run left the model's valid range" would say "linear" throughout.

- [ ] **Step 1: failing test.** Assert `state() == "invalid"` at α = −14° and
      `"marginal"` at −11°. Confirm both fail today.
- [ ] **Step 2:** make the gauge symmetric — `set_xlim(-ALPHA_SPAN_DEG, ALPHA_SPAN_DEG)`,
      mirrored `axvspan` bands, and `state()` comparing `abs(self._deg)`.
- [ ] **Step 3:** the existing positive-α tests must pass **unchanged** — that is the
      regression guard.
- [ ] **Step 4:** commit on its own, before item 3 touches anything.

Note while in there: `NZ_RANGE` is `(-1.0, 3.0)`, and item 3 reaches n_z = −1.55 at 12° of
elevator. The load-factor gauge will clip. Widening it is a display change with no physics
in it, but it should be a *declared* change like the others, not a silent one.

---

## Item 2 — two decisions that gate item 3, and the numbers that force them

Both are already open questions in §8. Neither was decidable from the text before; both are
now decidable on the model's own evidence, which is a better position than picking one.

**Coarse probe, measured 7 Aug 2026.** 747 at the CR-2144 cruise condition, zero wind, open
loop, elevator stepped from trim and held 12 s, dt 0.01. Trim n_z = 0.9967.

| elevator step | n_z min | Δn from trim | α min | α max | Δθ over 12 s |
|---|---|---|---|---|---|
| +2.0° | +0.578 | −0.419 | +1.28° | +4.64° | 10.40° |
| +4.0° | +0.152 | −0.845 | −2.08° | +4.64° | 21.06° |
| +6.0° | −0.277 | −1.274 | −5.42° | +4.63° | 31.96° |
| +8.0° | −0.706 | −1.703 | −8.76° | +4.63° | 43.08° |
| +10.0° | −1.130 | −2.127 | −12.11° | +4.63° | 54.36° |
| +12.0° | −1.551 | −2.548 | −15.45° | +4.63° | 65.77° |

### Decision A — is the Fig. 8 load band absolute, or an increment?

§8 records this as unresolvable from the paper's text. It has been harmless until now. It
stops being harmless here, because **the two readings land on opposite sides of the model's
validity boundary**:

- **Increment**, Δn = −1.9: between +8° and +10° of elevator, so |α| lands between 8.8° and
  12.1° — straddling the declared 10–12° band. **Marginal.**
- **Absolute**, n_z = −1.9: beyond +12° (which reaches only −1.551), so |α| worse than
  15.5°. **Outside**, and by a wide margin.

**Recommendation: fly the increment reading, and report the absolute one as out of reach.**
That is not a dodge — "one of the two readings of this band cannot be flown inside the
model's linear range" is a finding about the model's ceiling, and it belongs in §5 next to
the ±g asymmetry. Pin the exact elevator angle by bisection (the probe above is a coarse
sweep; a bisection was started and abandoned as too slow to be worth blocking on — do it
with a vmapped rollout rather than a Python loop over `load_factor`).

### Decision B — the analysis window

Also §8, "Which window is canonical for Fig. 8". Currently an explicit argument printed in
the figure's provenance footer, which was fine for two clusters that used the same one.

Δθ over the full 12 s is 43–54°, against the paper's manoeuvring reference of 12.0°. So the
window does not merely shade the answer, **it dominates it by a factor of three or four**.
Whatever is chosen must be the same convention the vortex and updraft points already use,
and the figure must keep saying which it was.

- [ ] Decide A and B before writing code. Record both in §8, replacing the open questions
      with the decision and the evidence that forced it.

---

## Item 3 — the manoeuvring case (§7 step 5)

**`vortex_viz.fly` cannot do this.** It flies "with fixed controls" by design — its
docstring explains why, and the reason is good: the Fig. 8 discriminator separates
turbulence from manoeuvring by whether pitch correlates with elevator, so an autopilot would
blur the categories. But a manoeuvre needs a *time-varying* elevator, which `fly` has no way
to express.

- [ ] **Step 1: failing test.** The manoeuvring point separates from both existing clusters
      on the (Δθ, Δn) plane. Confirm it fails — there is no third point yet.
- [ ] **Step 2:** add a sibling to `fly` taking a control schedule — `elevator(t)` — rather
      than fixed `Controls`, sharing the same `Encounter` return type and the same window
      handling so the three points are computed identically. Resist making `fly` itself
      general; a schedule that is constant is a strictly larger surface than a constant.
- [ ] **Step 3:** the elevator schedule shape is **a declared modelling choice, not source
      data** — the paper constrains the load, not how the pilot got there. A step is the
      simplest defensible choice. Name it in the same words `--sharpness` is named in, and
      report the load it produces alongside it.
- [ ] **Step 4:** add the point to `scripts/vortex.py`'s figure and to `FIG8_REFERENCE`'s
      comparison.
- [ ] **Step 5:** verify the third cluster separates, and that |α| stayed inside the band
      (or say plainly that it did not).
- [ ] **Step 6:** §4 gains the measured Δθ and Δn; §7 step 5 becomes `[DONE]`.

**Success criterion:** three model points and three paper references on one plane, with the
ordering preserved. **Not** agreement in absolute value — §5 already forbids claiming that,
since the paper never identifies an aircraft type.

---

## Item 4 — the twice-flagged dead test, if time allows

`test_viz.py::test_derived_agrees_with_the_aero_module` has now been flagged as
structurally-unable-to-fail in two session logs (§6b, and session 6's "deliberately not
done"). Its replacement landed in `test_sensors.py:196` in session 5. Two flags is enough.

- [ ] Delete it, or give it an independent expectation so it can fail. Deleting is
      defensible precisely because the replacement exists and is cited.

---

## Deferred, with reasons

- **§7 step 4, Dryden.** Blocked on a genuine external dependency: MIL-F-8785C Fig. 7's σ at
  40 kft is a chart to be read off, not a formula. Until that is digitised the layer cannot
  be written honestly. Its three structural changes (`WindState` carrying filter states,
  `init_sim`/`batch_sim` parameterised, a separate `omega_gust` filter, and the ensemble
  contract asserted in a test) stand exactly as §7 writes them — session 6 did **not**
  touch them, despite threading a wind *model* through the live loop.
- **Re-tuning `ManualGains`.** Needs a human flying it, so it wants its own short
  interactive session. It should be done together with tightening `test_manual.py`'s
  response bounds, which at ±2°/±45° are far too loose to pin any tuning.
- **The remaining panel cost.** Blitting is now 30 ms of a 37.8 ms frame. The three strips
  redraw 1,201 points each, every frame; sampling them at 5 Hz would cut roughly fourfold.
  Worth doing, but it is optimisation of something already fast enough to fly.
- **Re-measuring suite runtime.** §8 wants a quiet machine. The jit fix moved it from
  ~208 s to ~96 s, so the old spread is stale anyway.

---

## What could go wrong

- **The manoeuvring point may not separate.** If Δθ and Δn for a pushdown land on top of the
  updraft cluster, that is a real result about the model, not a bug to be tuned away. Report
  it. The timescale argument predicts separation; it is allowed to be wrong.
- **Both load-band readings may prove unflyable in the linear range.** Decision A's
  recommendation is the increment reading precisely because it is marginal rather than
  hopeless — but "marginal" means the answer may come back with |α| at 11-12°, inside the
  amber band. That is reportable, not fatal, provided the figure says so.
- **Item 0 is boring and will be tempting to skip.** It is first because the record is
  currently self-contradictory about frame rate, and a project whose central discipline is
  "every number carries its source" cannot afford a §8 that asks a question §9 has already
  answered.
