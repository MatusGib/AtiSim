# NOTATION — symbol and convention mapping across the three sources

Phase 1B of `AUDIT_PROMPT.md`. **Any numerical comparison made without an entry
in this table is invalid.** Where two sources genuinely conflict, the conflict is
recorded rather than resolved by picking one.

## The three sources, fully identified

| Tag | Document |
|---|---|
| **FD2e** | Stengel, Robert F., *Flight Dynamics*, Second Edition. Princeton, NJ: Princeton University Press, 2022. ISBN 9780691220253 (hardback), 9780691237046 (ebook); LCCN 2021052561. Copyright © 2004, 2022. 914 pp. In repo at `./Flight_Dynamics_-_Second_Edition.pdf`. Bibliographic details read from the book's own title and copyright pages (PDF pp. 5–6), not assumed. |
| **CR-2144** | Heffley, Robert K., and Wayne F. Jewell, *Aircraft Handling Qualities Data*, NASA CR-2144, December 1972. Systems Technology, Inc., Hawthorne, California; Technical Report 1004-1; Contract NAS 4-1729. 352 PDF pages. **Fetched during this audit** from NTRS (`https://ntrs.nasa.gov/api/citations/19730003312/downloads/19730003312.pdf`) and saved to `./refs/NASA-CR-2144.pdf`; it was NOT in the repository before. Section IX is the B-747. |
| **code** | `flightsim/`, this repository. |

**Page offset for CR-2144:** the scan carries front matter, so **PDF page =
printed page + 6** in Section IX. The text layer for the line-printer tables is
OCR noise; every value below was read from a **rendered image** of the page, not
from extracted text.

**Tables used:** IX-1 landing config non-dim (printed 216); IX-2 power-approach
non-dim (printed 217); IX-3 dimensional/mass/flight-condition (printed 229);
IX-4 longitudinal dimensional (printed 230); IX-5 elevator transfer-function
factors (printed 231); IX-6 thrust transfer-function factors (printed 232);
IX-8 lateral-directional dimensional (printed 234); Appendix A §5 dimensional
stability-derivative definitions (printed A-16 to A-18).

---

## 1. Axis systems and frames

| Quantity | FD2e | CR-2144 | code | conversion |
|---|---|---|---|---|
| Body axes | x forward, y right (starboard), z down | same; Appendix A §1 "AXIS SYSTEMS" | `state.py`: x-fwd, y-right, z-down | none |
| Earth frame | inertial, z down | — | NED, z down, treated as inertial | none |
| Axis system of the data | n/a | **Tables IX-4, IX-5, IX-6, IX-8 each print "(BODY AXIS SYSTEM)" in their own headers** — verified by reading the pages | body throughout | **none needed; verified, not assumed** |
| **Table IX-2 (approach non-dim)** | n/a | **states h, V_To, α₀, δ_s and NO axis system at all** | used verbatim as body-axis | **CONFLICT/GAP — see §7.1** |
| Stability axes | body rotated about y by α | "The same symbols are used for body- and stability-axis dimensional derivatives. Care should be exercised so that a consistent set of quantities are used." (A-16) | `validation.to_stability_axes`, `aircraft.stability_to_body` | rotate by α₀ |
| Altitude | up positive | `H(FT)` | `-pos_ned[2]` | sign flip |

## 2. Incidence angles

| Quantity | FD2e | CR-2144 | code | conversion |
|---|---|---|---|---|
| α | positive nose-up relative to the relative wind | `ALPHA(DEG)`, Table IX-3; α₀ = 4.60° (FC9), 5.70° (FC2) | `aero.air_data`: `atan2(w,u)`, positive nose-up | deg → rad at the boundary |
| β | positive with relative wind from the right | not tabulated for the 747 | `asin(v/V)`, docstring says "positive nose-left (relative wind from the right)" | — |
| θ₀ | — | `GAMMA(DEG) = 0`, so θ₀ = α₀ | `trim` enforces γ = 0, θ = α | none |
| δe | — | positive trailing-edge down implied by C_mδe < 0 | positive trailing-edge down (`aero.py` docstring) | consistent |
| δa | — | **Table IX-2 footnote: "δa = total deflection of right inboard aileron plus left inboard aileron with the effect of outboard ailerons included"** | taken as given; positive = right-roll command | **a compound control the code treats as a single angle — declared, not converted** |
| δr | — | positive trailing-edge left implied by C_nδr < 0 | positive trailing-edge left | consistent |

## 3. Force and moment coefficients — the one that actually bites

| Quantity | FD2e | CR-2144 | code | conversion |
|---|---|---|---|---|
| Normal force | C_Z (down positive) | **C_N, positive UP** (Appendix A §4a) | uses C_L, not C_N | `C_N = C_L cos α + C_D sin α` |
| Axial force | C_X (forward positive) | **C_X, positive AFT** — established in this audit by checking that only the aft-positive reading reproduces Table IX-4's `XW = +0.106` sign at FC2 | uses C_D, not C_X | `C_X = C_D cos α − C_L sin α` |
| Lift | C_L | C_L (Tables IX-1, IX-2) | `CL` | none |
| Drag | C_D | C_D | `CD` | none |
| Rolling moment | C_l | C_l | `Cl` | none |
| Pitching moment | C_m | C_m | `Cm` | none |
| Yawing moment | C_n | C_n | `Cn` | none |

**This is the highest-consequence row in the table.** CR-2144's Appendix A
defines its dimensional Z and X derivatives in terms of **C_N and C_X**, and the
code inverts them as though they were defined in terms of **C_L and C_D**. At
α₀ ≈ 5° the two differ at the ~1% level. Quantified in `AUDIT.md`.

## 4. Rate derivatives and non-dimensionalisation

| Quantity | FD2e | CR-2144 | code | conversion |
|---|---|---|---|---|
| Pitch rate | `q̂ = qc/2V` (eq. 3.4-10) — **half-chord** | implied by `M_q = (ρSc²V_To/4I_y)C_mq` (A-17) ⇒ same `q̂ = qc/2V` | `q_hat = q*ac.c/(2V)` | **none — all three agree** |
| Roll rate | `p̂ = pb/2V` (eq. 3.4-39) — **half-span** | implied by `L_p = (ρSV_To b²/4I_x)C_lp` (A-17) ⇒ same | `p_hat = p*ac.b/(2V)` | **none — all three agree** |
| Yaw rate | `r̂ = rb/2V` | implied by `N_r = (ρSV_To b²/4I_z)C_nr` (A-18) ⇒ same | `r_hat = r*ac.b/(2V)` | **none — all three agree** |
| Angular units | per radian | **"/rad" printed explicitly beside every entry of Table IX-2** | per radian | none |
| Reference length, pitch | c̄ | c̄ = 27.31 ft (Table IX-3 header) | `ac.c` | ft → m |
| Reference length, lateral | b | b = 195.68 ft (Table IX-3 header) | `ac.b` | ft → m |
| Reference area | S | S = 5500 ft² (Table IX-3 header) | `ac.S` | ft² → m² |

## 5. Speed symbols — a real trap, and the code walks into it

| Quantity | CR-2144 | code | note |
|---|---|---|---|
| `V_To` | total true airspeed. `VTO(FPS) = 774.` at FC9, `278.` at FC2 | read as `U0 = 774.0` / `V0 = 278` | |
| `U_0` | the **body-x component**, `U₀ = V_To cos α₀` — Appendix A uses `U_0` in `X_u`, `X_w`, `Z_u`, `Z_w`, `M_u`, `M_w` and `V_To` in `M_q`, `M_δe`, `Z_δe` and every lateral relation | the code uses one number, named `U0`, for both | **CONFLICT.** At α₀ = 4.60° the two differ by 0.32%. For the lateral and control relations the code's choice is exactly right; for `X_w`, `Z_w`, `M_w` it is not. |
| `W_0` | `W₀ = V_To sin α₀`; appears explicitly in Appendix A's `X_u/X_w/Z_u/Z_w/M_u/M_w` | **never formed** | **the code omits the whole `W₀` term** — see §7.2 |
| `q̄` | `Q(PSF)` = 177 (FC9), 92.2 (FC2) | `qbar` taken from the table | none |

## 6. Inertia, mass and the moment reference

| Quantity | CR-2144 | code | conversion |
|---|---|---|---|
| `I_x, I_y, I_z` | `IX/IY/IZ (SLUG-FT SQ)`, Table IX-3 | `Ix, Iy, Iz` | `SLUG_FT2_TO_KG_M2` |
| **`I_xz` sign** | `IXZ(SLUG-FT SQ) = 970056.` (FC9), `870050.` (FC2), positive. Appendix A A-18 defines `L' = (L_β + I_xz N_β/I_x)G`, `G = 1/(1 − I_xz²/(I_x I_z))` | `inertia_tensor` places **`−Ixz`** in the (0,2)/(2,0) slots; `_unprime` inverts exactly the A-18 relation | **verified against A-18 — the code's docstring relation and CR-2144's are identical** |
| Primed vs unprimed | **Table IX-8 tabulates PRIMED values** (`LB'`, `LP'`, `NB'` …) | `_unprime` recovers raw values before non-dimensionalising | correct; using primed values directly would double-count `I_xz` |
| Weight | `W(LBS)` = 636636 (FC9), 564032 (FC2) | `W * LB2KG` | 2.4e-6 relative slip vs the `/32.174` slug route (`g₀` is 32.17405 ft/s²); negligible |
| **CG / moment reference** | **`C.G.(MGC) = .250` for BOTH flight condition 2 and flight condition 9** — Table IX-3 | **recorded nowhere in the code** | **the code assumes, without stating, that the aerodynamic moment reference, the inertia reference and the accelerometer location are one point. CR-2144 supplies the number that makes this checkable, and both conditions use the same CG, so the assumption is benign — but it is undeclared.** |
| Sensor station | `LXP(FT) = 86.0`, `LZP(FT) = −10.0` (Table IX-3, both conditions) | absent | the arm needed to bound `ASSUMPTIONS` B4, which declares itself unboundable |

## 7. Recorded conflicts — not resolved, recorded

### 7.1 Table IX-2 states no axis system
Tables IX-4, IX-5, IX-6 and IX-8 each carry "(BODY AXIS SYSTEM)" in their
headers. **Table IX-2 does not**, and neither does Table IX-1. Meanwhile the
Section IX derivative *plots* for the lateral set (printed pp. 224–228) are
labelled **"Stability Axis"**. `aircraft._boeing_747_approach` takes Table IX-2's
lateral derivatives verbatim with no rotation, i.e. assumes body axes.
**Status: `unverified`.** The assumption is probably right — a
stability-to-body rotation by α₀ = 5.7° would move `Clb` and `Cnb` by ~2% and
the approach case already agrees with an independent implementation to ~1% —
but the source does not state it and the code does not flag it.

### 7.2 CR-2144's `X_w`, `Z_w`, `M_w` carry a `W₀` term the code drops
Appendix A (printed A-16, A-17), verbatim:

```
X_w = (ρSU₀/2m)[ −C_Xα − 2(W₀/U₀)(C_X + (M/2)C_XM) ]
Z_w = (ρSU₀/2m)[ −C_Nα − 2(W₀/U₀)(C_N + (M/2)C_NM) ]
M_w = (ρScU₀/2I_y)[  C_mα + (2W₀/U₀)(C_m + (M/2)C_mM) ]
```

`aircraft.py` inverts these as if the bracket were the first term alone, and as
if `U₀ = V_To`. Both the `W₀` term and the `U₀`/`V_To` distinction are dropped.
Consequence quantified in `AUDIT.md`. The relations the code uses for
`C_mq`, `C_mδe`, `C_Lδe`, `C_Lq` and **every lateral derivative** carry no `W₀`
term in Appendix A either, so those are exact.

### 7.3 `C_Lα̇`: CR-2144 disagrees with itself
- Table IX-2 (approach) prints **`C_Lα̇ = −6.7 /rad`** and `C_mα̇ = −3.2 /rad`.
  Table IX-1 (landing) independently prints **`C_Lα̇ = −6.7 /rad`** and
  `C_mα̇ = −3.3 /rad`, so the negative sign is consistent across both
  non-dimensional tables and is not a single typo.
- Table IX-4 flight condition 2 prints **`ZWD = +.0338`** and `MWD = −.000240`.
- Appendix A: `Z_ẇ = −(ρSc/4m)(U₀/V_To)C_Nα̇`, `M_ẇ = (ρSc²/4I_y)(U₀/V_To)C_mα̇`.

Putting Table IX-2's `C_mα̇ = −3.2` through Appendix A gives `M_ẇ = −0.000241`
against Table IX-4's `−0.000240` — **consistent to 0.4%**. Putting Table IX-2's
`C_Lα̇ = −6.7` through Appendix A gives `Z_ẇ = −0.03407`, but Table IX-4 prints
**`+0.0338`** — **same magnitude to 0.8%, opposite sign**.

Caughey (the independent implementation the project compares against) publishes
`Z_ẇ = −0.0341`, i.e. he follows the Table IX-2 route. For a conventional
aft-tail aircraft `C_Lα̇` is physically positive, which would make Table IX-4's
`+0.0338` the expected sign and Table IX-2's printed `−6.7` a sign typo.

**Status: recorded conflict, not resolved.** It does not touch the engine, which
excludes α̇ entirely — but `ASSUMPTIONS.md` C2 calls the omission's cost "exact
rather than attributed" on the strength of a reconstruction that silently picks
one side of this conflict, and describes `C_Lα̇ = 6.7` as "tabulated" when the
table prints `−6.7`.

### 7.4 FD2e eq. 3.4-49 has a sign error
FD2e §3.4 (printed p. 216) gives the body-axis wind-shear/rate equivalences:

```
∂w/∂y = +p   (3.4-48)      ∂v/∂z = +p   (3.4-49)
∂w/∂x = −q   (3.4-50)      ∂u/∂z = +q   (3.4-51)
∂v/∂x = +r   (3.4-52)      ∂u/∂y = −r   (3.4-53)
```

A rigid rotation gives `v = rx − pz`, hence `∂v/∂z = −p`. Equation 3.4-49 is
inconsistent with 3.4-48 within the book's own framing. **Confirmed: the book is
wrong.** The project's `docs/superpowers/specs/2026-08-14-...` records this
independently; this audit re-derived it rather than transcribing the claim.

### 7.5 FD2e eq. 3.4-55 is wrong by a factor of −2
FD2e (printed p. 217): `ΔM = −(ΔM_q,wing + ΔM_q,ht)(w_wx − u_wz)`.
For an air mass in rigid rotation at rate `q_a`, `w_wx = −q_a` and `u_wz = +q_a`,
so the bracket is `−2q_a` and the book gives `ΔM = +2 M_q q_a`. The correct
increment is `ΔM = −M_q q_a`. **Ratio −2. Confirmed: the book is wrong**, and the
code correctly does not use this equation.

### 7.6 `wind.py`'s citation of FD2e on strip theory overstates the source
`wind.py`'s strip-integration header says FD2e "p. 217 is explicit that below
rotor scale the rotary derivatives stop being adequate and strip theory or CFD
is required". The actual passage (printed p. 218) says stability derivatives
"can be used to estimate disturbance effects for **wind rotors**, but they have
limited value for **wake vortices**, where fine-scale estimates … based on strip
theory or computational fluid dynamics are required." The book names wake
vortices specifically and explicitly endorses rotary derivatives for wind
rotors. A Parks Kelvin–Helmholtz vortex with r₀ = 183 m is a wind rotor, not a
wake vortex. **Status: the citation does not say what the code claims.** This
weakens the stated justification for the strip path; it does not make the strip
path wrong.

---

## 8. Conversions actually required, in one list

| From → to | Factor | Where applied | Verified |
|---|---|---|---|
| ft → m | `FT2M = 0.3048` exact | `aircraft.py` at definition | yes |
| ft² → m² | `FT2M²` | `aircraft.py` | yes |
| lb → kg | `LB2KG = 0.45359237` exact | `aircraft.py` | yes |
| slug·ft² → kg·m² | `SLUG2KG·FT2M²`, `SLUG2KG = LBF2N/FT2M` | `aircraft.py` | yes |
| kt → m/s | `1852/3600` exact | `aircraft.py` | yes |
| deg → rad | `π/180` | `aircraft.py` | yes |
| lb (weight) → slug (mass) | `/32.174` | `aircraft._B747_G` | 2.4e-6 slip vs `g₀ = 32.17405` |
| dimensional → non-dimensional | CR-2144 Appendix A §5 | `aircraft.from_dimensional_*` and inline | **partially — see §7.2** |
| primed → unprimed lateral | Appendix A A-18 | `aircraft._unprime` | yes, exactly |
| body → stability axes | rotation by α₀ | `validation.to_stability_axes` | yes |
| SI → imperial plant matrix | per-element | `validation.to_imperial_matrix` | yes |

**Not required anywhere, and correctly absent:** per-degree → per-radian for the
747 (Table IX-2 prints "/rad"); chord ↔ half-chord (all three sources use
`c/2V`); stability → body for the 747 cruise set (Tables IX-4 and IX-8 print
"BODY AXIS SYSTEM").
