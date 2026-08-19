# Agent F — Blind Reproduction of the B-747 Power-Approach Longitudinal Linearisation

**Scope constraint honoured.** I read *only* the two permitted documents:

- `./refs/NASA-CR-2144.pdf` — Heffley & Jewell, *Aircraft Handling Qualities Data*, NASA CR-2144, Dec. 1972.
- `./Flight_Dynamics_-_Second_Edition.pdf` — Stengel, *Flight Dynamics*, 2nd ed., Princeton UP, 2022.

I did **not** open anything under `flightsim/`, `docs/PROJECT.md`, `docs/ASSUMPTIONS.md`, `notebooks/`, or
`scripts/`. I also did not open `AUDIT_PROMPT.md`, `analysis-ui-investigation-prompt.md`, `runs/`, or
`audit_evidence/G-archaeology.md`. The only engine numbers I used are the ones supplied in my task prompt.

Page references are given as **printed page** (the number on the page) with the PDF page in parentheses.
Every number below is traceable to a table, figure, or equation I actually rendered and read. Anything I
could not verify is explicitly marked `unverified`.

---

## 1. Source data as read

### 1.1 Table IX-2 — Power Approach Configuration Non-Dimensional Derivatives (printed p. 217, PDF p. 223)

Header: `h = sea level`, `V_T0 = 165 KTAS`, `α₀ = 5.7°`, `δ_s = −2.1°`

| Coefficient | Value | | Coefficient | Value |
|---|---|---|---|---|
| C_L      | 1.11        | | C_Lq   | 5.4 /rad    |
| C_D      | 0.102       | | C_mq   | −20.8 /rad  |
| C_Lα     | 5.70 /rad   | | C_LM   | −0.81       |
| C_Dα     | 0.66 /rad   | | C_mM   | 0.27        |
| C_mα     | −1.26 /rad  | | C_Lδe  | 0.338 /rad  |
| C_Lα̇     | −6.7 /rad   | | C_mδe  | −1.34 /rad  |
| C_mα̇     | −3.2 /rad   | |        |             |

**C_DM is not tabulated.** From the figure on printed p. 222 (PDF p. 228), the C_DM curve is identically
zero for Mach below ≈ 0.78 at SL, 20 000 ft and 40 000 ft. At M = 0.249 I therefore take **C_DM = 0**.
C_Dδe is not tabulated either; it does not enter the plant matrix.

### 1.2 Table IX-3 — Dimensional, Mass and Flight Condition Parameters (printed p. 229, PDF p. 235)

Header: `S = 5500 sq ft`, `b = 195.68 ft`, `c̄ = 27.31 ft`

Flight condition **2** column:

| Row | Value |
|---|---|
| H(FT)          | SL |
| M(−)           | 0.249 |
| VTO(FPS)       | 278. |
| VTO(KTAS)      | 165. |
| VTO(KCAS)      | 165. |
| W(LBS)         | 564032. |
| C.G.(MGC)      | 0.250 |
| IX (slug-ft²)  | .142E+8 |
| **IY (slug-ft²)** | **.323E+8** |
| IZ (slug-ft²)  | .454E+8 |
| IXZ (slug-ft²) | 870050. |
| EPSILON(DEG)   | −1.60 |
| **Q(PSF)**     | **92.2** |
| QC(PSF)        | 93.6 |
| **ALPHA(DEG)** | **5.70** |
| **GAMMA(DEG)** | **0.** |
| LXP(FT)        | 86.0 |
| LZP(FT)        | −10.0 |
| ITH(DEG)       | 2.50 |
| XI(DEG)        | 2.50 |
| LTH(FT)        | 10.0 |

Flight condition 2 header matches Table IX-2 exactly (SL, 165 KTAS, α = 5.70°), so IX-2 and IX-3 col. 2
describe the same case.

### 1.3 Derived flight-condition scalars

`VTO(FPS)` is printed as `278.`, i.e. rounded. 165 KTAS = 165 × 1.68781 = **278.489 ft/s**. Checking against
the tabulated dynamic pressure: with standard sea-level ρ = 2.3769 × 10⁻³ slug/ft³,
q̄ = ½ρV² = ½(2.3769e-3)(278.489)² = **92.17 psf**, versus the tabulated `Q(PSF) = 92.2`. This pins both
V_T0 and ρ. (Using the rounded `278.` instead would need ρ = 2 × 92.2/278² = 2.3860e-3 to hit 92.2; the
choice changes the dimensional derivatives by ≲ 0.4 %, which I show explicitly in §3.)

- V_T0 = U₀ = 278.489 ft/s
- ρ = 2.3769 × 10⁻³ slug/ft³
- q̄ = 92.1713 psf
- m = W/g = 564032/32.174 = **17530.68 slug**
- I_y = 3.23 × 10⁷ slug-ft²
- g = 32.174 ft/s²

Consistency of the trim state: W/(q̄S) = 564032/(92.1713 × 5500) = **1.11262**, versus the tabulated
C_L = 1.11. So the tabulated trim is L = W at γ = 0, as `GAMMA(DEG) = 0.` states.

Convenient groups:

```
ρ S V_T0 / m            = 0.2076741   1/s
q̄ S / m                 = 28.91744    ft/s²
ρ S c̄ V_T0 / (4m)       = 1.417895    ft/s
ρ S c̄ V_T0 / I_y        = 3.0782244e-3
```

---

## 2. Definitions quoted from the source

### 2.1 Non-dimensional definitions — CR-2144 App. A §4 (printed p. A-14, PDF p. 336)

> **a) Longitudinal Body Axis**
> C_N = N / (q̄ S), positive up
> C_X = − X / (q̄ S), positive aft
> C_M = M / (q̄ S c)
> C_Nα = ∂C_N/∂α ; C_Nα̇ = (2V_T0/c) ∂C_N/∂α̇ ; C_NM = ∂C_N/∂M ; C_Nδ = ∂C_N/∂δ
> C_Xα = ∂C_X/∂α ; C_XM = ∂C_X/∂M ; C_Xδ = ∂C_X/∂δ
> C_Mα = ∂C_M/∂α ; C_Mα̇ = (2V_T0/c) ∂C_M/∂α̇ ; C_MM = ∂C_M/∂M ; C_Mq = (2V_T0/c) ∂C_M/∂q
>
> **b) Longitudinal Stability Axis**
> C_L = L/(q̄ S), positive up ; C_D = D/(q̄ S), positive aft
> C_Lα = ∂C_L/∂α ; C_Lα̇ = (2V_T0/c) ∂C_L/∂α̇ ; C_LM = ∂C_L/∂M ; C_Lδ = ∂C_L/∂δ
> C_Dα = ∂C_D/∂α ; C_DM = ∂C_D/∂M ; C_Dδ = ∂C_D/∂δ
> *"Pitching moment derivatives are identical to those for body axis"*

Two things follow that matter later: `C_X` is defined **positive aft** (so C_X = +C_D in stability axes,
not −C_D), and the rate derivatives are non-dimensionalised on `c/(2V_T0)`.

### 2.2 Axis transformation — CR-2144 App. B (printed p. B-1, PDF p. 341)

> U₀ = V_T0 cos α₀   ;   W₀ = V_T0 sin α₀
>
> **LONGITUDINAL — Body Axis**
> C_N   = C_L cos α₀ + C_D sin α₀
> C_X   = C_D cos α₀ − C_L sin α₀
> C_Nα  = C_Lα cos α₀ − C_L sin α₀ + C_Dα sin α₀ + C_D cos α₀
> C_Nα̇  = C_Lα̇ cos α₀
> C_Nq  = C_Lq cos α₀
> C_NM  = C_LM cos α₀ + C_DM sin α₀
> C_Nδ  = C_Lδ cos α₀ + C_Dδ sin α₀
> C_Xα  = C_Dα cos α₀ − C_D sin α₀ − C_Lα sin α₀ − C_L cos α₀
> C_Xα̇  = − C_Lα̇ sin α₀
> C_Xq  = − C_Lq sin α₀
> C_XM  = C_DM cos α₀ − C_LM sin α₀
> C_Xδ  = C_Dδ cos α₀ − C_Lδ sin α₀
> C_m, C_mα, C_mα̇, C_mq, C_mM, C_mδ — UNCHANGED

**This is the definition of "stability axis" I use.** Setting α₀ → 0 in the above recovers the stability-axis
non-dimensional set, because stability axes are precisely the body axes rotated so that x is along V_T0:

| Stability-axis coefficient | = | Value |
|---|---|---|
| C_N | C_L                    | 1.11 |
| C_X | C_D                    | 0.102 |
| C_Nα | **C_Lα + C_D**        | 5.70 + 0.102 = **5.8020** |
| C_Xα | **C_Dα − C_L**        | 0.66 − 1.11 = **−0.4500** |
| C_Nα̇ | C_Lα̇                  | −6.7 |
| C_Xα̇ | 0                     | 0 |
| C_Nq | C_Lq                   | 5.4 |
| C_Xq | 0                      | 0 |
| C_NM | C_LM                   | −0.81 |
| C_XM | C_DM                   | 0 |

The two bolded rows are the **α-rotation terms**: at a perturbed α, lift and drag rotate with the
instantaneous relative wind, so a fraction C_D of drag appears as normal force and a fraction C_L of lift
appears as axial force. CR-2144's own transformation contains them (the `+ C_D cos α₀` and `− C_L cos α₀`
terms survive at α₀ = 0). They are numerically large here: C_L = 1.11 flips the sign of C_Xα from
+0.66 to −0.45. This turns out to be decisive (§6.4).

Note also that **X_q ≡ 0 and X_ẇ ≡ 0 in stability axes** for this data set: C_Xq = −C_Lq sin α₀ → 0 and
C_Xα̇ = −C_Lα̇ sin α₀ → 0, and CR-2144 tabulates no C_Dq and no C_Dα̇.

### 2.3 Dimensional definitions — CR-2144 App. A §5 (printed pp. A-16/A-17, PDF pp. 338/339)

> **5. DIMENSIONAL STABILITY DERIVATIVE DEFINITIONS**
> *"The same symbols are used for body- and stability-axis dimensional derivatives. Care should be
> exercised so that a consistent set of quantities are used."*
>
> **a) Longitudinal Body Axis**
> X_u* = X_u + T_u cos ξ₀                                                              [1/sec]
> X_u  = (ρSU₀/m) ( − (M/2) C_XM − C_X + (W₀/2U₀) C_Xα )                               [1/sec]
> X_w  = (ρSU₀/2m) [ − C_Xα − 2 (W₀/U₀) ( C_X + (M/2) C_XM ) ]                         [1/sec]
> X_δe = − (ρSV_T0²/2m) C_Xδe                                                          [ft/sec²-rad]
> Z_u* = Z_u − T_u sin ξ₀                                                              [1/sec]
> Z_u  = (ρSU₀/m) ( − (M/2) C_NM − C_N + (W₀/2U₀) C_Nα )                               [1/sec]
> Z_w  = (ρSU₀/2m) [ − C_Nα − 2 (W₀/U₀) ( C_N + (M/2) C_NM ) ]                         [1/sec]
> Z_ẇ  = − (ρSc/4m) (U₀/V_T0) C_Nα̇
> Z_δe = − (ρSV_T0²/2m) C_Nδe                                                          [ft/sec²-rad]
> M_u* = M_u + (ℓ_th/I_y) T_u                                                          [1/sec-ft]
> M_u  = (ρScU₀/I_y) [ (M/2) C_mM + C_m − (W₀/2U₀) C_mα ]                              [1/sec-ft]
> M_w  = (ρScU₀/2I_y) [ C_mα + (2W₀/U₀) ( C_m + (M/2) C_mM ) ]                         [1/sec-ft]
> M_ẇ  = (ρSc²/4I_y) (U₀/V_T0) C_mα̇
> M_α  = U₀ M_w   ;   M_α̇ = U₀ M_ẇ
> M_q  = (ρSc²V_T0/4I_y) C_mq                                                          [1/sec]
> M_δe = (ρScV_T0²/2I_y) C_mδe                                                         [1/sec²]
> T_u  = (1/(a m)) ∂T/∂M                                                               [1/sec]

**Gap in the source: Z_q is never defined.** App. A §5a lists X_u, X_w, X_δe, Z_u, Z_w, Z_ẇ, Z_δe, M_u,
M_w, M_ẇ, M_α, M_α̇, M_q, M_δe, T_u — there is no Z_q and no X_q. Yet the mnemonic table on printed
p. A-9 (PDF p. 331) lists `ZQ → Z_q [1/sec]` and Table IX-4 prints a value for it. I therefore **inferred**
the definition by analogy with M_q (both are `c/(2V_T0)`-normalised rate derivatives) and with the sign
convention of Z_w:

```
Z_q = − (ρ S c V_T0 / 4m) C_Nq                                   [INFERRED, not printed]
```

I verified this inference numerically against Table IX-4: it reproduces `ZQ = −7.58` to 0.5 % at F/C 2 and
`ZQ = −6.22` to 1.1 % at F/C 1 (§3). I flag it as inferred rather than quoted.

Two unit misprints in the source, noted but harmless: p. A-9 gives `ZWD → Z_ẇ [1/sec²]` (the A-16 formula
is dimensionless) and `MWD → M_ẇ [1/sec-ft]` (the A-17 formula is 1/ft).

### 2.4 Equations of motion — Stengel

CR-2144 never writes its equations of motion. I take the structure from Stengel and from the derivative
definitions themselves.

Stengel eq. 5.1-27 (printed p. 397, PDF p. 415), repeated as eq. 5.4-5 (printed p. 434, PDF p. 452), gives a
*hybrid-axis* fourth-order longitudinal model on the state Δx_Lon = [ΔV Δγ Δq Δα]ᵀ:

> F_Lon =
> ```
> [   −D_V            −g cos θ_o        −D_q                       −D_α                ]
> [   L_V/(V_o+L_α̇)    g sin γ_o / V_o   L_q/(V_o+L_α̇)              L_α/(V_o+L_α̇)      ]
> [   M_V − M_α̇L_V/(V_o+L_α̇)   0        M_q − M_α̇(L_q−V_o)/(V_o+L_α̇)  M_α − M_α̇L_α/(V_o+L_α̇) ]
> [  −L_V/(V_o+L_α̇)   −g sin γ_o / V_o  (V_o−L_q)/(V_o+L_α̇)        −L_α/(V_o+L_α̇)     ]
> ```
> (eq. 5.4-5, printed p. 434)

That is a different state vector from the one I was asked for. I therefore assemble the [u, w, q, θ] form
directly from CR-2144's own derivative definitions, and use Stengel's formulation only as an
*independent structural cross-check* (§5.3).

In stability axes with Θ₀ = 0 (which follows from `GAMMA(DEG) = 0.`: with γ₀ = 0 the stability x-axis is
along the horizontal velocity vector, so the stability-axis pitch attitude is zero) and W₀ = 0, the
perturbation equations are

```
Δu̇ = X_u Δu + X_w Δw + X_q Δq − g cos Θ₀ Δθ
Δẇ = Z_u Δu + Z_w Δw + Z_ẇ Δẇ + (U₀ + Z_q) Δq − g sin Θ₀ Δθ
Δq̇ = M_u Δu + M_w Δw + M_ẇ Δẇ + M_q Δq
Δθ̇ = Δq
```

`U₀ Δq` in the ẇ row is the kinematic (Coriolis) term −(−U₀q) from the body-frame acceleration; it is not
an aerodynamic derivative. Solving the ẇ row and substituting into the q̇ row:

```
Δẇ = [ Z_u Δu + Z_w Δw + (U₀ + Z_q) Δq ] / (1 − Z_ẇ)
```

giving, with X_q = 0 and Θ₀ = 0,

```
        ⎡ X_u                          X_w                          0                              −g ⎤
A   =   ⎢ Z_u/(1−Z_ẇ)                  Z_w/(1−Z_ẇ)                  (U₀+Z_q)/(1−Z_ẇ)                0 ⎥
        ⎢ M_u + M_ẇZ_u/(1−Z_ẇ)         M_w + M_ẇZ_w/(1−Z_ẇ)         M_q + M_ẇ(U₀+Z_q)/(1−Z_ẇ)       0 ⎥
        ⎣ 0                            0                            1                               0 ⎦
```

---

## 3. Validation of my formula chain against CR-2144's own output

Before computing anything new I checked that I can reproduce CR-2144's *printed* body-axis dimensional
derivatives from CR-2144's non-dimensional tables. This is the control experiment: if it fails, nothing
downstream is trustworthy.

**Table IX-4 — B-747 Longitudinal Dimensional Derivatives (BODY AXIS SYSTEM), printed p. 230 (PDF p. 236).**
F/C 2 column: `XU* −.0108, ZU* −.150, MU* .000181, XW .106, ZW −.613, MW −.00193, ZWD .0338, ZQ −7.58,
MWD −.000240, MQ −.437, XDE .971, ZDE −9.73, MDE −.574`.

Using α₀ = 5.70°, V_T0 = 278.489 ft/s, ρ = 2.3769e-3, the App. B transformation and the App. A §5a formulas:

### F/C 2 — Power Approach (SL, M = 0.249)

| Derivative | Mine | Table IX-4 | Ratio |
|---|---|---|---|
| X_u  | −0.0108226   | −0.0108    | 1.002 |
| Z_u  | −0.150522    | −0.150     | 1.003 |
| **M_u** | **+0.000295572** | **+0.000181** | **1.633** |
| X_w  | +0.105780    | +0.106     | 0.998 |
| Z_w  | −0.612823    | −0.613     | 1.000 |
| M_w  | −0.00191942  | −0.00193   | 0.995 |
| Z_ẇ  | +0.0337758   | +0.0338    | 0.999 |
| Z_q  | −7.61878     | −7.58      | 1.005 |
| M_ẇ  | −0.000240299 | −0.000240  | 1.001 |
| M_q  | −0.437145    | −0.437     | 1.000 |

### F/C 1 — Landing (SL, M = 0.198), independent replication

Using Table IX-1 (printed p. 216, PDF p. 222: C_L 1.76, C_D .263, C_Lα 5.67, C_Dα 1.13, C_mα −1.45,
C_Lα̇ −6.7, C_mα̇ −3.3, C_Lq 5.65, C_mq −21.4, C_LM −1.1, C_mM .36, α₀ = 8.5°) and Table IX-3 col. 1
(131 KTAS, W 564032, I_y .323E+8):

| Derivative | Mine | Table IX-4 | Ratio |
|---|---|---|---|
| X_u  | −0.020898    | −0.0209    | 1.000 |
| Z_u  | −0.202258    | −0.202     | 1.001 |
| **M_u** | **+0.00034804** | **+0.000117** | **2.975** |
| X_w  | +0.121913    | +0.122     | 0.999 |
| Z_w  | −0.511584    | −0.512     | 0.999 |
| M_w  | −0.00173951  | −0.00177   | 0.983 |
| Z_ẇ  | +0.0333671   | +0.0334    | 0.999 |
| Z_q  | −6.29047     | −6.22      | 1.011 |
| M_ẇ  | −0.000246304 | −0.000246  | 1.001 |
| M_q  | −0.357078    | −0.357     | 1.000 |

**Conclusion: 9 of 10 derivatives reproduce to ≤ 1.1 % at both sea-level flight conditions.** My reading of
Table IX-2, of the App. B transformation, of the App. A §5a formulas, of ρ and V_T0, and my inferred Z_q
are all confirmed. Sensitivity to the V_T0 rounding: using `278.` and ρ = 2.3860e-3 instead moves every
entry by ≤ 0.4 % and does not change any conclusion.

### 3.1 An internal inconsistency in CR-2144: M_u

M_u is the single exception, and the residual is not a constant factor (1.633 at F/C 2, 2.975 at F/C 1), so
it is not a units or scaling slip on my side. It cannot be the printed thrust term either:

- `M_u* = M_u + (ℓ_th/I_y) T_u` with ℓ_th = `LTH(FT)` = 10.0 would need T_u = −0.370 s⁻¹ at F/C 2 and
  T_u = −0.746 s⁻¹ at F/C 1 to close the gap.
- But `X_u* = X_u + T_u cos ξ₀` with ξ₀ = `XI(DEG)` = 2.50° reproduces `XU*` to 0.2 % with **T_u ≈ 0**.
  A T_u of −0.37 s⁻¹ would make X_u* ≈ −0.39, not −0.0108.

The two columns cannot both be satisfied by any single T_u. Inverting the A-17 M_u formula for the value
of C_mM that *would* reproduce the printed MU* gives C_mM = **−0.0304** at F/C 2 and C_mM = **−0.6055** at
F/C 1, against the tabulated +0.27 and +0.36.

I report this without attributing a cause. It is a discrepancy **internal to CR-2144**, present before the
engine enters the picture, and any audit conclusion that leans on C_mM inherits it. Because of it I carry
two variants forward:

- **M_u (A)** = +1.03475e-4 s⁻¹ft⁻¹ — from the A-17 definition with the tabulated C_mM = 0.27.
- **M_u (A′)** = −1.16666e-5 s⁻¹ft⁻¹ — back-solved from Table IX-4's printed `MU* = .000181` through the
  App. B dimensional transform, i.e. what CR-2144's own computer run actually used.

---

## 4. My stability-axis dimensional derivatives

Setting α₀ → 0 (U₀ = V_T0 = 278.489, W₀ = 0) in the App. A §5a formulas with the stability-axis
coefficients of §2.2. Worked algebra for each:

```
X_u  = (ρSV/m)( −(M/2)C_DM − C_D )
     = 0.2076741 × ( −0 − 0.102 )                                 = −0.02118276  s⁻¹

X_w  = (ρSV/2m)( −C_Xα ) = (ρSV/2m)( C_L − C_Dα )
     = 0.10383705 × ( 1.11 − 0.66 ) = 0.10383705 × 0.45           = +0.04672668  s⁻¹

X_q  = 0   (C_Xq = −C_Lq sin α₀ → 0; no C_Dq tabulated)           =  0

Z_u  = (ρSV/m)( −(M/2)C_LM − C_L )
     = 0.2076741 × ( −0.1245×(−0.81) − 1.11 )
     = 0.2076741 × ( +0.100845 − 1.11 )                           = −0.20957539  s⁻¹
      [without the Mach term: 0.2076741 × (−1.11)                 = −0.23051829  s⁻¹]

Z_w  = (ρSV/2m)( −C_Nα ) = (ρSV/2m)( −(C_Lα + C_D) )
     = 0.10383705 × ( −5.8020 )                                   = −0.60246267  s⁻¹

Z_q  = −(ρScV/4m) C_Lq = −1.417895 × 5.4                          = −7.6566339   ft/s

Z_ẇ  = −(ρSc/4m) C_Lα̇ = −(1.417895/278.489) × (−6.7)              = +0.03411233  (dimensionless)

M_u  = (ρScV/I_y)( (M/2) C_mM + C_m )
     = 3.0782244e-3 × ( 0.1245 × 0.27 + 0 )                       = +1.0347e-4   s⁻¹ft⁻¹

M_w  = (ρScV/2I_y) C_mα = 1.5391122e-3 × (−1.26)                  = −1.93928e-3  s⁻¹ft⁻¹

M_q  = (ρSc²V/4I_y) C_mq = (1.417895×27.31/3.23e7×...)            = −0.4371448   s⁻¹
       [ = (ρSc²V/4I_y) = 0.02101657 ; × (−20.8) ]

M_ẇ  = (ρSc²/4I_y) C_mα̇ = (0.02101657/278.489) × (−3.2)           = −2.41493e-4  ft⁻¹
```

Summary:

| | X | Z | M |
|---|---|---|---|
| **u** | −0.02118276 | −0.20957539 (−0.23051829 w/o Mach) | +1.03475e-4 (A) / −1.16666e-5 (A′) |
| **w** | +0.04672668 | −0.60246267 | −1.93928e-3 |
| **q** | 0 | −7.6566339 | −0.4371448 |
| **ẇ** | 0 | +0.03411233 | −2.41493e-4 |

---

## 5. Plant matrices and modes

### 5.1 Variant A′ — full CR-2144 derivative set (the reference reproduction)

Uses everything: Mach derivatives (C_LM), α̇ derivatives (C_Lα̇, C_mα̇), Z_q, the α-rotation terms, and
M_u taken from Table IX-4.

```
        ⎡ −2.1182762e-02   4.6726681e-02   0.0000000e+00  −3.2174000e+01 ⎤
A_A′ =  ⎢ −2.1697698e-01  −6.2373989e-01   2.8039701e+02   0.0000000e+00 ⎥
        ⎢  4.0731787e-05  −1.7886526e-03  −5.0485871e-01   0.0000000e+00 ⎥
        ⎣  0.0000000e+00   0.0000000e+00   1.0000000e+00   0.0000000e+00 ⎦
```

Eigenvalues: `−0.571989 ± 0.707528j`, `−0.002901 ± 0.126743j`

| Mode | ω_n (rad/s) | ζ |
|---|---|---|
| Phugoid      | **0.12678** | **0.02289** |
| Short-period | **0.90982** | **0.62869** |

**Cross-check against CR-2144's own answer.** Table IX-5, "B-747 Elevator Transfer Function Factors, Bare
Airframe (BODY AXIS SYSTEM)", printed p. 231 (PDF p. 237), F/C 2 DENOMINATOR block:

```
Z(DET)1 = .0228      W(DET)1 = .127        <- phugoid
Z(DET)2 = .629       W(DET)2 = .910        <- short period
```

| | Mine (A′) | CR-2144 IX-5 | Δ |
|---|---|---|---|
| phugoid ω_n      | 0.12678 | 0.127  | −0.17 % |
| phugoid ζ        | 0.02289 | 0.0228 | +0.39 % |
| short-period ω_n | 0.90982 | 0.910  | −0.02 % |
| short-period ζ   | 0.62869 | 0.629  | −0.05 % |

**Agreement to four significant figures on all four numbers.** Eigenvalues are invariant under the
stability↔body rotation (it is a similarity transform on [u,w,q,θ]), so this is a legitimate comparison.
This validates the equations of motion, the axis handling, the Θ₀ = 0 choice, the Z_ẇ / M_ẇ implicit-mass
handling, and my inferred Z_q — end to end, against the source's own computed output.

For completeness, Variant **A** (same but with M_u from the tabulated C_mM = 0.27) gives phugoid
0.13746 / 0.01649 and short-period 0.90906 / 0.62991 — the short period is unaffected but the phugoid
moves off CR-2144's printed answer, which is the §3.1 inconsistency showing up in the modes.

### 5.2 Variant B — the engine-comparable model

Dropping the terms the engine's stated aerodynamic form does not contain (no α̇ terms ⇒ Z_ẇ = M_ẇ = 0; no
Mach derivatives ⇒ the C_LM term in Z_u and the C_mM term in M_u both vanish, so M_u = 0 exactly). C_Lq,
C_mq, and the α-rotation terms are retained, because the engine's stated form does contain CLq/Cmq and a
drag polar.

```
        ⎡ −2.1182762e-02   4.6726681e-02   0.0000000e+00  −3.2174000e+01 ⎤
A_B  =  ⎢ −2.3051829e-01  −6.0246267e-01   2.7083202e+02   0.0000000e+00 ⎥
        ⎢  0.0000000e+00  −1.9392814e-03  −4.3714480e-01   0.0000000e+00 ⎥
        ⎣  0.0000000e+00   0.0000000e+00   1.0000000e+00   0.0000000e+00 ⎦
```

Eigenvalues: `−0.528895 ± 0.721456j`, `−0.001500 ± 0.134058j`

| Mode | ω_n (rad/s) | ζ | Period |
|---|---|---|---|
| Phugoid      | **0.134066** | **0.011188** | 46.87 s |
| Short-period | **0.894555** | **0.591238** | 8.71 s |

### 5.3 Independent structural cross-check via Stengel

I rebuilt Variant B in Stengel's hybrid-axis formulation (eq. 5.1-27, printed p. 397 / eq. 5.4-5, printed
p. 434), state [ΔV, Δγ, Δq, Δα], γ₀ = 0, with the mapping D_V = (ρSV/m)C_D, D_α = (q̄S/m)C_Dα,
L_V = (ρSV/m)C_L, L_α = (q̄S/m)C_Lα, L_q = (ρScV/4m)C_Lq, M_V = M_u, M_α = M_w V₀, M_q as above, D_q = 0.
Note that Stengel's D_α and L_α are the *bare* drag and lift derivatives — the α-rotation terms that appear
inside stability-axis X_w and Z_w are carried instead by his separate γ̇ equation, so the mapping is
D_α = −X_w V₀ − (q̄S/m)C_L and L_α = −Z_w V₀ − (q̄S/m)C_D.

Result: phugoid 0.13448 / 0.01124, short-period 0.89181 / 0.58710, versus my [u,w,q,θ] Variant B of
0.13407 / 0.01119 and 0.89456 / 0.59124. Agreement to 0.3 %, 0.4 %, 0.3 %, 0.7 %. The residual is the
difference between the two formulations themselves (Stengel's hybrid-axis α equation carries
(V₀ − L_q)/(V₀ + L_α̇) where mine carries Z_q exactly), not an error in either. Two independent EOM
formulations therefore give the same answer.

---

## 6. Comparison with the engine

### 6.1 Difference table

Engine values as supplied to me (I did not read them from the repo):

| Quantity | Mine (Variant B) | Engine | Difference | Relative |
|---|---|---|---|---|
| A[0,0] = X_u              | −0.021183 | −0.020940 | +0.000243 | **+1.15 %** |
| A[0,1] = X_w              | +0.046727 | +0.046320 | −0.000407 | **−0.87 %** |
| A[1,0] = Z_u              | −0.230518 | −0.228510 | +0.002008 | **+0.87 %** |
| A[0,3] = −g cos Θ₀        | −32.174000 | −32.174000 | 0 | **exact** |
| A[1,3] = −g sin Θ₀        | 0.0 | 0.0 | 0 | **exact** |
| phugoid ω_n               | 0.134066 | 0.133400 | −0.000666 | **−0.50 %** |
| phugoid ζ                 | 0.011188 | 0.012890 | +0.001702 | **+15.2 %** |
| short-period ω_n          | 0.894555 | 0.896100 | +0.001545 | **+0.17 %** |
| short-period ζ            | 0.591238 | 0.591100 | −0.000138 | **−0.02 %** |
| trim α                    | 5.70° (Table IX-2/IX-3) | 5.57° | −0.13° | **−2.3 %** |

And against the source's own published answer:

| Quantity | CR-2144 Table IX-5 | Engine | Relative |
|---|---|---|---|
| phugoid ω_n      | 0.127  | 0.1334  | **+5.0 %** |
| phugoid ζ        | 0.0228 | 0.01289 | **−43 %** |
| short-period ω_n | 0.910  | 0.8961  | **−1.5 %** |
| short-period ζ   | 0.629  | 0.5911  | **−6.0 %** |

### 6.2 A[0,3] and A[1,3] — exact

A[0,3] = −32.174 and A[1,3] = 0 match my derivation exactly. This confirms Θ₀ = 0 in the engine's
stability-axis linearisation and g = 32.174 ft/s², consistent with `GAMMA(DEG) = 0.` in Table IX-3.

### 6.3 The three aerodynamic elements: a common ~0.9 % scale plus a consistent trim shift

The three supplied aerodynamic elements over-determine the trim aerodynamic coefficients. Inverting them
against the table's ρ, S, V_T0, m:

```
C_L,eng  = −A[1,0] / (ρSV/m) = 0.228510 / 0.2076741         = 1.100330   (table 1.11 ; −0.871 %)
C_D,eng  = −A[0,0] / (ρSV/m) = 0.020940 / 0.2076741         = 0.100831   (table 0.102; −1.146 %)
C_Dα,eng = C_L,eng − 2A[0,1]/(ρSV/m) = 1.100330 − 0.446036  = 0.654246   (table 0.66 ; −0.872 %)
```

Two structural facts fall out of these three numbers:

1. **The engine's C_Dα/C_L equals the table's to five decimals.**
   engine 0.594591 vs. table 0.66/1.11 = 0.594595.
   For a parabolic polar C_D = C_D0 + C_L²/(πeAR), C_Dα = κ C_L C_Lα with κ ≡ 2/(πeAR). Calibrating κ on
   the table point, κ = 0.66/(1.11 × 5.70) = 0.104315. Then κ × C_L,eng × 5.70 = **0.654250** versus the
   engine-implied 0.654246 — a ratio of 1.000006. The engine's induced-drag model and the table's C_Dα are
   the same number evaluated at a slightly different C_L.

2. **Both C_L and C_D imply the same trim angle of attack.**
   Linearising off the table point with C_Lα = 5.70 and C_Dα = 0.66:
   - from C_L: α = 5.70° − (1.11 − 1.100330)/5.70 × 180/π = **5.6028°**
   - from C_D: α = 5.70° − (0.102 − 0.100831)/0.66 × 180/π = **5.5985°**
   These agree to 0.004°, and bracket the engine's reported trim α of 5.57°.

So the ~0.9–1.1 % offsets in A[0,0], A[0,1], A[1,0] are one effect, not three: the engine is evaluating
the *same* aerodynamic data at a trim α of about 5.60° instead of the source's 5.70°. The residual
0.03° between the implied 5.60° and the reported 5.57° I could not resolve from the supplied elements.

Candidate explanations for the trim-α offset, and what would distinguish them:
- **(a)** The engine solves L = W at a slightly different ρ, V, S or W than Table IX-3. Discriminator:
  check the engine's ρ, V_T0, S, W against 2.3769e-3, 278.489, 5500, 564032; and check whether its
  q̄ is 92.17.
- **(b)** The engine's C_L0 / C_Lα are fitted such that C_L(5.70°) ≠ 1.11. Discriminator: evaluate the
  engine's CL0 + CLa·(5.70°·π/180) and compare with 1.11.
- **(c)** The engine trims with a nonzero elevator contribution C_Lδe δe (Table IX-2 gives
  C_Lδe = 0.338/rad, and Table IX-2 lists δ_s = −2.1° stabiliser). Discriminator: check whether the
  engine's trim δe is nonzero and whether its trim C_L includes CLde·de.

**Important negative result:** A[1,0] = −0.228510 is *not* consistent with a pure −2g/V₀ at
V₀ = 278.489 ft/s, which is −0.231061. The airspeed that would give −0.228510 from −2g/V is 281.60 ft/s
(166.84 kt). Discriminator between "different trim α" and "different airspeed": the airspeed hypothesis
also rescales A[0,0] and A[0,1] by the same factor, which would leave C_D/C_L unchanged at 0.091892 — but
the engine's C_D/C_L is 0.091637, 0.28 % lower. So a pure airspeed difference does not explain all three
elements; a trim-α shift does.

### 6.4 What the engine demonstrably *does* include

Dropping the α-rotation terms (i.e. using C_Nα = C_Lα and C_Xα = C_Dα instead of C_Lα + C_D and
C_Dα − C_L) gives:

```
X_w = (ρSV/2m)(−C_Dα) = −0.06853      phugoid ω_n 0.13537, ζ = −0.04599   (UNSTABLE)
Z_w = (ρSV/2m)(−C_Lα) = −0.59187      short-period 0.88592 / 0.59974
```

The engine's A[0,1] is **+0.04632**, positive, and its phugoid is stable. The engine therefore
unambiguously includes the α-rotation terms. This is worth recording explicitly because it is the single
most common way to get a stability-axis linearisation wrong, and the engine gets it right.

### 6.5 The one substantive residual: phugoid damping

Every quantity above agrees to ≤ 1.2 % except phugoid ζ: mine 0.011188, engine 0.012890, a difference of
**+0.001702 absolute (+15.2 % relative)**.

To separate "my row 0 differs from the engine's" from "something else differs", I built a hybrid: the
engine's supplied A[0,0], A[0,1], A[1,0] with my Z_w, Z_q, M_w, M_q:

| | Hybrid | Engine | Δ |
|---|---|---|---|
| phugoid ω_n      | 0.13349 | 0.13340 | −0.07 % |
| phugoid ζ        | 0.01073 | 0.01289 | **+20.1 %** |
| short-period ω_n | 0.89448 | 0.89610 | +0.18 % |
| short-period ζ   | 0.59123 | 0.59110 | −0.02 % |

So with the engine's own row 0 in place, three of the four modal numbers land within 0.2 %, and the
phugoid damping gap *widens*. The residual therefore lives in a matrix element I was not given
(A[0,2], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2]), not in row 0.

Perturbations that individually close the gap, and their side effects:

| Perturbation | Value needed | Resulting phugoid | Resulting short period |
|---|---|---|---|
| A[0,2] = X_q  | **−1.049 s⁻¹**  | 0.13354 / 0.01289 | 0.89413 / 0.59114 |
| A[2,0] = M_u  | **−5.523e-5**   | 0.12831 / 0.01289 | 0.89497 / 0.59065 |
| A[2,2] = M_q  | ≈ −0.47         | 0.13185 / 0.01367 | 0.90560 / 0.60170 |

Only the X_q hypothesis leaves phugoid ω_n and both short-period parameters where the engine reports them;
M_u ≈ −5.5e-5 pulls phugoid ω_n down to 0.1283 against the engine's 0.1334, and M_q ≈ −0.47 pushes
short-period ω_n up to 0.9056 against 0.8961.

An arithmetic observation relevant to distinguishing these, stated without asserting causation: the engine's
drag model is C_D = C_D0 + C_L²/(πeAR), and its C_L contains a CLq·q̂ term. A parabolic polar therefore
carries an induced-drag sensitivity to pitch rate, C_Dq = κ C_L C_Lq. With κ = 0.104315 (calibrated in §6.3)
and the engine's C_L = 1.100330, C_Dq = 0.104315 × 1.100330 × 5.4 = 0.61984, giving
X_q = −(ρScV/4m)C_Dq = −1.417895 × 0.61984 = **−0.8788 s⁻¹**. CR-2144 tabulates no C_Dq at all, so in my
reproduction X_q = 0 identically.

**Discriminator:** read the engine's A[0,2]. If it is ≈ −0.88 to −1.05, the residual is a pitch-rate axial
force absent from CR-2144's tabulated derivative set. If A[0,2] = 0, the residual is in row 1 or row 2 and
the candidates above are ruled out. Either way this is a small effect: it moves phugoid ζ by 0.0017 and
nothing else by more than 0.2 %.

---

## 7. Terms included and dropped, with the numerical cost of each

Baseline for the sensitivity study is Variant A′ (which reproduces CR-2144 Table IX-5 to four figures).
Each row drops exactly one thing.

| Model | phugoid ω_n | phugoid ζ | s.p. ω_n | s.p. ζ |
|---|---|---|---|---|
| **A′ — all terms (= CR-2144 IX-5)** | 0.12678 | 0.02289 | 0.90982 | 0.62869 |
| drop C_Lα̇ only (Z_ẇ = 0)           | 0.12677 | 0.02158 | 0.89419 | 0.62667 |
| drop C_mα̇ only (M_ẇ = 0)           | 0.12680 | 0.01996 | 0.90968 | 0.59197 |
| drop both α̇ terms                  | 0.12679 | 0.01865 | 0.89405 | 0.59060 |
| drop C_LM only (Mach term in Z_u)  | 0.13297 | 0.01609 | 0.91044 | 0.62909 |
| drop C_Lq only (Z_q = 0)           | 0.12565 | 0.02569 | 0.91799 | 0.62377 |
| **B — drop α̇ + Mach (engine-like)** | **0.13407** | **0.01119** | **0.89456** | **0.59124** |
| B, also drop α-rotation terms      | 0.13537 | −0.04599 | 0.88592 | 0.59974 |
| **Engine (supplied)**              | **0.13340** | **0.01289** | **0.89610** | **0.59110** |

Reading:

- **C_mα̇ = −3.2** is what sets short-period damping: dropping it takes ζ_sp from 0.629 to 0.592, a −5.8 %
  change. This alone accounts for essentially the whole 6.0 % gap between CR-2144's published ζ_sp = 0.629
  and the engine's 0.5911.
- **C_Lα̇ = −6.7** is what sets short-period frequency: dropping it takes ω_n,sp from 0.910 to 0.894,
  a −1.7 % change. This accounts for the 1.5 % gap between CR-2144's 0.910 and the engine's 0.8961.
- **C_LM = −0.81** is what sets phugoid frequency: dropping it takes ω_n,ph from 0.1268 to 0.1330, a
  +4.9 % change. This accounts for the 5.0 % gap between CR-2144's 0.127 and the engine's 0.1334.
- **C_Lq = 5.4** (via Z_q) matters more than it looks: dropping it moves ω_n,sp by +0.9 % and ζ_ph by
  +12 %. It is retained in both my Variant B and, evidently, the engine.
- **The α-rotation terms are load-bearing.** Without them the phugoid is unstable.

**Terms I included:** C_L, C_D, C_Lα, C_Dα, C_mα, C_Lq (as Z_q), C_mq, the α-rotation terms C_D and −C_L,
gravity at Θ₀ = 0, and the U₀q kinematic term. In Variant A′ additionally C_Lα̇, C_mα̇, C_LM, and M_u.

**Terms I dropped, and why:**

| Dropped | Reason |
|---|---|
| C_DM | Read as identically 0 below M ≈ 0.78 from the figure on printed p. 222; M = 0.249 here. |
| C_m (trim) | Trimmed flight, C_m = 0. Consistent with Table IX-3 `GAMMA(DEG) = 0.` and W/(q̄S) = 1.1126 ≈ C_L = 1.11. |
| T_u, thrust terms | ∂T/∂M is not given anywhere in CR-2144. Constrained to ≈ 0 by the `XU*` column (§3.1). |
| X_q, X_ẇ | Identically zero in stability axes for this data set: C_Xq = −C_Lq sin α₀ → 0, C_Xα̇ = −C_Lα̇ sin α₀ → 0, and no C_Dq / C_Dα̇ is tabulated. |
| C_Lδe, C_mδe | Control derivatives; they populate B, not A. |
| Lateral-directional | Out of scope. |
| Aeroelastic / flexible effects | Figures in Section IX are labelled "Flexible", but no separate rigid data is given and no modal data is provided. Not separable. |
| Altitude / range states | Stengel, printed p. 395: *"Altitude variation has small effect on dynamic characteristics, and range perturbations have no dynamic effect in the flat-earth model."* |

---

## 8. Arithmetic I could not complete, and why

1. **T_u and the thrust contribution to X_u* and M_u*.** CR-2144 defines T_u = (1/(am)) ∂T/∂M (printed
   p. A-17) but tabulates neither ∂T/∂M nor the speed of sound `a`. I could only bound T_u ≈ 0 from the
   `XU*` column. Marked `unverified`.

2. **CR-2144's M_u.** I cannot reproduce the printed `MU*` from CR-2144's own definitions and tables at
   either sea-level flight condition (§3.1), and no thrust term can reconcile the `XU*` and `MU*` columns
   simultaneously. I carried both variants rather than choosing.

3. **Z_q.** No definition appears in App. A. I inferred `Z_q = −(ρScV_T0/4m)C_Nq` by analogy with M_q and
   confirmed it numerically to 0.5 % / 1.1 % against Table IX-4 at two flight conditions. Marked
   `INFERRED`.

4. **The exact engine trim α.** From the three supplied matrix elements I get 5.598°–5.603°; the engine
   reports 5.57°. The remaining 0.03° cannot be resolved without matrix elements I was not given.

5. **The phugoid-ζ residual.** I could bound what would explain it (§6.5) but not identify it, because
   A[0,2], A[1,1], A[1,2], A[2,0], A[2,1], A[2,2] were not supplied.

6. **C_DM at M = 0.249 to better than "zero".** The figure on printed p. 222 has no plotted curve below
   M ≈ 0.78 and its vertical resolution is ≈ 0.02. I take C_DM = 0; anything up to |C_DM| ≈ 0.02 would be
   invisible on that figure and would change X_u by ≤ 0.5 %.

7. **Aeroelasticity.** The Section IX figures are labelled "Flexible" and Table IX-2's derivatives are
   presumably flexible-airframe values, but no rigid-airframe counterpart is given, so I could not
   quantify the difference.

---

## 9. Assessment

**The engine reproduces the CR-2144 power-approach linearisation correctly, given its own stated
aerodynamic form.** Independently derived from the two source documents:

- Both gravity elements are exact.
- The three supplied aerodynamic elements agree to 0.87–1.15 %, and that spread is a single coherent
  effect (trim at α ≈ 5.60° rather than 5.70°), not three independent errors. The engine's C_D/C_L and
  C_Dα/C_L ratios match Table IX-2 to 0.28 % and 0.0007 % respectively.
- Short-period ω_n agrees to 0.17 % and ζ to 0.02 %.
- Phugoid ω_n agrees to 0.50 %.
- Phugoid ζ differs by +0.0017 absolute (+15.2 %), traceable to a matrix element I was not given.

**The larger differences are against CR-2144's *published* modes, and they are fully accounted for by the
two derivative families the engine's stated form omits.** Dropping C_mα̇ = −3.2 explains the −6.0 % in
ζ_sp; dropping C_Lα̇ = −6.7 explains the −1.5 % in ω_n,sp; dropping C_LM = −0.81 explains the +5.0 % in
ω_n,ph. These are consequences of the model form, quantified in §7, not arithmetic errors.

**Findings that belong to the source, not the engine:**

- CR-2144's Table IX-4 `MU*` column is not reproducible from CR-2144's own App. A §5a definition together
  with the C_mM values of Tables IX-1/IX-2, at either sea-level flight condition, and no single T_u
  reconciles the `XU*` and `MU*` columns. The value CR-2144's own run used corresponds to C_mM ≈ −0.03
  (F/C 2) and ≈ −0.61 (F/C 1) rather than the tabulated +0.27 and +0.36.
- App. A §5 defines no Z_q and no X_q, though `ZQ` is printed in Table IX-4 and listed on p. A-9.
- App. B printed p. B-2 gives `(M_u)_b = M_w cos α₀ − M_u sin α₀`, which transposes M_u and M_w relative
  to the companion line `(M_w)_b = M_w cos α₀ + M_u sin α₀`; only the latter form is consistent with
  Table IX-4. I used the consistent form.
- Unit misprints on p. A-9 for `ZWD` and `MWD`.

**Reproducibility.** All arithmetic in this document is deterministic and derives from the tables quoted in
§1 and the definitions quoted in §2. The chain was validated three ways: against CR-2144's Table IX-4
body-axis derivatives at two flight conditions (§3), against CR-2144's Table IX-5 modes (§5.1, four
significant figures), and against Stengel's independent hybrid-axis formulation (§5.3).
