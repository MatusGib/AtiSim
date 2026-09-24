# Physics and Assumptions

This manual gives the mathematical model of AtiSim, the assumptions in the model, and the limits
of its use. It also gives the evidence that the model is correct inside those limits.

The content follows the documentation practice of NASA-STD-7009, *Standard for Models and
Simulations*. That standard asks for the intended use, the assumptions, the limits of operation,
and the verification and validation of a model.

## 1 Use of the model

### 1.1 Intended use

AtiSim is a model of the **longitudinal** gust response of a rigid transport aircraft at
cruise. Use it to compare turbulence encounters, and to find the mechanism of a response.

AtiSim can answer these questions:

- Which of two encounters gives the larger load?
- How does the response change with airspeed, altitude or aircraft?
- Which frequencies does the airframe amplify?

AtiSim cannot give the absolute load that a real aircraft feels in an encounter.

:::{caution}
Do not use AtiSim to calculate a design load or a certification load. AtiSim does not predict
absolute loads. An incorrect load can cause an unsafe design.
:::

### 1.2 Limits of operation

The validation in section 10 applies only inside these limits. Outside one of these limits, the
results have no validation.

| Parameter | Validated range | Reason for the limit |
|---|---|---|
| Aircraft | `boeing747` and `boeing747_jsbsim` only | Only these aircraft have a measured validation. |
| Mach number | 0.70 to 0.90 | The range of the validation data. |
| Altitude | 35,000 to 45,000 ft for `boeing747`, 35,000 to 41,000 ft for `boeing747_jsbsim` | The range of the validation data. For both aircraft together, use the smaller range. |
| Angle of attack | less than 10°. Use 10° to 12° with caution. | The lift model has no stall (assumption C1). |
| Gust length | more than 3 wingspans | The model samples the gust at one point (assumption E2). |
| Axis | longitudinal only | No recorded lateral response is available for validation. |

`aircraft.valid_mach` and `aircraft.valid_altitude` hold these ranges in the code. The
`alpha_band` run check shows when a run goes out of the angle-of-attack limit.

## 2 Frames, units and state

### 2.1 Units

All quantities are in SI units: meters, seconds, kilograms and radians. The module
`atisim.units` holds the conversion factors.

### 2.2 Frames of reference

| Frame | Axes | Use |
|---|---|---|
| NED | x north, y east, z down | the inertial frame, flat and non-rotating |
| Body | x forward, y right, z down, origin at the center of gravity | forces, moments, rates |

The quaternion $q = [q_w, q_x, q_y, q_z]$ gives the attitude. The rotation matrix $C(q)$ changes
a body vector into an NED vector. Its transpose $C^T$ changes an NED vector into a body
vector.

### 2.3 State and controls

| Symbol | Code name | Content |
|---|---|---|
| $\mathbf{p}$ | `State.pos_ned` | position in NED, m |
| $\mathbf{v} = [u, v, w]$ | `State.vel_body` | inertial velocity in body axes, m/s |
| $q$ | `State.quat` | attitude quaternion, body to NED |
| $\boldsymbol{\omega} = [p, q, r]$ | `State.omega` | body rates, rad/s |
| $\delta_e, \delta_a, \delta_r$ | `Controls.elevator`, `aileron`, `rudder` | control deflections, rad |
| $\delta_t$ | `Controls.throttle` | throttle, 0 to 1 |

## 3 Equations of motion

The function `dynamics.derivatives` calculates the time derivative of the state. The model is a
rigid body with six degrees of freedom.

### 3.1 Translation

$$
\dot{\mathbf{v}} = \frac{\mathbf{F}_a + \mathbf{F}_t}{m} + C^T \mathbf{g} - \boldsymbol{\omega} \times \mathbf{v}
$$

$\mathbf{F}_a$ is the aerodynamic force, $\mathbf{F}_t$ is the thrust, $m$ is the mass, and
$\mathbf{g} = [0, 0, g(h)]$ is gravity in NED.

### 3.2 Rotation

$$
\dot{\boldsymbol{\omega}} = I^{-1} \left( \mathbf{M} - \boldsymbol{\omega} \times I \boldsymbol{\omega} \right)
$$

$I$ is the inertia tensor, with the product of inertia $I_{xz}$. $\mathbf{M}$ is the sum of the
aerodynamic moment and the thrust moment.

### 3.3 Kinematics

$$
\dot{\mathbf{p}} = C \mathbf{v}, \qquad
\dot{q} = \tfrac{1}{2} \, q \otimes [0, \boldsymbol{\omega}]
$$

### 3.4 Load factor

The load factor is the body-normal specific force, as an accelerometer at the center of gravity
measures it:

$$
n_z = -\frac{f_z}{g_0}, \qquad
\mathbf{f} = \dot{\mathbf{v}} - C^T \mathbf{g} + \boldsymbol{\omega} \times \mathbf{v}
$$

In level flight, $n_z$ is about 1. The function `dynamics.load_factor` calculates it.

## 4 Earth, gravity and atmosphere

### 4.1 Earth

The Earth is flat and does not turn (assumption A1).

### 4.2 Gravity

Gravity changes with altitude $h$ by the inverse-square law:

$$
g(h) = g_0 \left( \frac{R}{R + h} \right)^2, \qquad g_0 = 9.80665 \ \mathrm{m/s^2}, \quad R = 6\,371\,000 \ \mathrm{m}
$$

Gravity does not change with latitude.

### 4.3 Atmosphere

The atmosphere is the International Standard Atmosphere (ISA), with two layers to 20 km. The
model changes the geometric altitude $z$ into the geopotential altitude $H$:

$$
H = \frac{R_0 \, z}{R_0 + z}, \qquad R_0 = 6\,356\,766 \ \mathrm{m}
$$

At sea level, the temperature is 288.15 kelvin. Below 11,000 m, the temperature decreases by
0.0065 kelvin for each meter. Above 11,000 m, the temperature stays at 216.65 kelvin. Pressure
comes from hydrostatic balance. Density comes from the gas law with
$R = 287.05 \ \mathrm{J/(kg\,K)}$. The speed of sound is $a = \sqrt{\gamma R T}$, with
$\gamma = 1.4$.

## 5 Aerodynamics

### 5.1 Air-relative velocity

The aerodynamic model uses only the velocity and the rates of the aircraft relative to the air:

$$
\mathbf{v}_r = \mathbf{v} - C^T \mathbf{w}, \qquad
\boldsymbol{\omega}_r = \boldsymbol{\omega} - \boldsymbol{\omega}_g
$$

$\mathbf{w}$ is the wind in NED, and $\boldsymbol{\omega}_g$ is the gust rate in body axes
(section 7.2). From $\mathbf{v}_r = [u_r, v_r, w_r]$:

$$
V = |\mathbf{v}_r|, \qquad \alpha = \operatorname{atan2}(w_r, u_r), \qquad \beta = \arcsin(v_r / V)
$$

The dimensionless rates are $\hat{p} = p_r b / 2V$, $\hat{q} = q_r \bar{c} / 2V$,
$\hat{r} = r_r b / 2V$ and $\hat{\dot\alpha} = \dot\alpha \bar{c} / 2V$. Here, $b$ is the span
and $\bar{c}$ is the mean chord.

### 5.2 Longitudinal coefficients

$$
C_L = C_{L,\alpha} + C_{L_q} \hat{q} + C_{L_{\delta e}} \delta_e + C_{L_{\dot\alpha}} \hat{\dot\alpha} + C_{L_M} \Delta M
$$

$$
C_m = C_{m_0} + C_{m_\alpha} \alpha + C_{m_q} \hat{q} + C_{m_{\delta e}} \delta_e + C_{m_{\dot\alpha}} \hat{\dot\alpha} + C_{m_M} \Delta M
$$

$$
C_D = C_{D_0} + \frac{C_L^2}{\pi e A} + C_{D,\mathrm{wave}} + C_{D_\beta} \beta^2 + C_{D_\alpha} \alpha + C_{D_M} \Delta M
$$

- The lift term $C_{L,\alpha}$ is $C_{L_0} + C_{L_\alpha} \alpha$. For the 737 and `boeing747_jsbsim`, it is an interpolation in a table of $C_L$ against $\alpha$ instead.
- The drag uses the **total** lift coefficient, with the rate and control terms (assumption C7).
- The Mach term $\Delta M = M - M_{\mathrm{ref}}$ applies only to `boeing747`. This aircraft has the Mach derivatives of CR-2144 at $M_{\mathrm{ref}} = 0.80$.
- The 737 has tables of $C_{m_{\delta e}}$ and $C_{l_{\delta a}}$ against Mach number.

The model can also apply a Prandtl–Glauert factor to the lift slope and to the rate and control
derivatives. An aircraft turns it on with `pg_mach_ref`. No aircraft in the registry uses it.

### 5.3 Wave drag

The wave drag uses the Korn equation for the drag-divergence Mach number. The inputs are the
sweep angle $\Lambda$, the thickness ratio $t/c$ and the airfoil factor $\kappa$:

$$
M_{DD} = \frac{\kappa}{\cos\Lambda} - \frac{t/c}{\cos^2\Lambda} - \frac{C_L}{10\cos^3\Lambda}, \qquad
M_{\mathrm{crit}} = M_{DD} - \left(\frac{0.1}{80}\right)^{1/3}
$$

$$
C_{D,\mathrm{wave}} = 20 \, \max(M - M_{\mathrm{crit}}, 0)^4
$$

### 5.4 Lateral coefficients

$$
C_Y = C_{Y_\beta} \beta + C_{Y_p} \hat{p} + C_{Y_r} \hat{r} + C_{Y_{\delta r}} \delta_r
$$

$$
C_l = C_{l_\beta} \beta + C_{l_p} \hat{p} + C_{l_r} \hat{r} + C_{l_{\delta a}} \delta_a + C_{l_{\delta r}} \delta_r
$$

$$
C_n = C_{n_\beta} \beta + C_{n_p} \hat{p} + C_{n_r} \hat{r} + C_{n_{\delta a}} \delta_a + C_{n_{\delta r}} \delta_r
$$

The 737 has a yaw damper, which adds $k \, r$ to the rudder deflection.

### 5.5 Forces and moments

The dynamic pressure is $\bar{q} = \tfrac{1}{2} \rho V^2$, and $S$ is the wing area. The lift is
$L = \bar{q} S C_L$, the drag is $D = \bar{q} S C_D$, and the side force is $Y = \bar{q} S C_Y$.
In body axes:

$$
\mathbf{F}_a =
\begin{bmatrix}
-D \cos\alpha \cos\beta + L \sin\alpha \\
-D \sin\beta + Y \\
-D \sin\alpha \cos\beta - L \cos\alpha
\end{bmatrix},
\qquad
\mathbf{M}_a = \bar{q} S \begin{bmatrix} b \, C_l \\ \bar{c} \, C_m \\ b \, C_n \end{bmatrix}
$$

### 5.6 Rate of change of angle of attack

The terms in $\dot\alpha$ need the rate of change of the relative velocity:

$$
\dot\alpha = \frac{u_r \dot{w}_r - w_r \dot{u}_r}{u_r^2 + w_r^2}
$$

The model adds two parts:

- The part from the aircraft. The model calculates the acceleration once without this part, and then uses that acceleration to get $\dot\alpha$.
- The part from the wind. This part has two terms: the gradient of the field along the flight path, and the transport term $\boldsymbol{\omega} \times C^T \mathbf{w}$. Without the transport term, a uniform wind can change the attitude.

## 6 Thrust

$$
T = \delta_t \, T_{\max} \left( \frac{\rho}{\rho_0} \right)^{n} \left( 1 + k_M M^2 \right), \qquad \rho_0 = 1.225 \ \mathrm{kg/m^3}
$$

The thrust acts along the thrust line, at the incidence $i_T$ to the x-axis of the body. For
`boeing747`, the thrust line is 1.737 m (5.70 ft) below the center of gravity, at 2.5°. Thus the
thrust gives a pitching moment. For the other aircraft, the thrust line is on the x-axis of the
body, through the center of gravity.
The engines have no dynamics (assumption C5).

## 7 Wind and turbulence

### 7.1 Wind coupling

The wind changes only the air-relative velocity and rates (section 5.1). The Coriolis, gyroscopic
and kinematic terms use the **inertial** velocity and rates. Thus a gust changes the flow on the
wings, but not the velocity of the airframe over the ground.

This design prevents two frequent errors in gust models:

- An air-relative velocity in the Coriolis term. This error breaks Galilean invariance.
- An added $-m \, d\mathbf{w}/dt$ term. This error counts the gust two times.

A wind model is a function with this interface:

```python
wind_model(wind_state, state, key, dt) -> (wind_ned, omega_gust, wind_state, key)
```

`wind.field_model(field)` makes a wind model from a function of position. The model samples the
wind at the center of gravity (assumption E2).

### 7.2 Gust rates

The gust rates come from the gradient of the wind field, in body axes, with $w_g$ positive down:

$$
p_g = \frac{\partial w_g}{\partial y}, \qquad
q_g = -\frac{\partial w_g}{\partial x}, \qquad
r_g = \frac{\partial v_g}{\partial x}
$$

`wind.gust_rates` uses the analytic gradient at the center of gravity. `wind.sampled_rates` fits
the slope across the airframe instead. The two methods give the same result for a field that is
linear across the aircraft.

### 7.3 Rankine vortex array

Parks et al. (1985) and Mehta (1987) model clear-air turbulence as a row of horizontal vortices.
Each vortex has a core radius $r_0$ and a peak speed $V_0$. The aircraft is at the distance $l$
past a core and the height $d$ above it, and $r^2 = l^2 + d^2$. The wind components are:

$$
\text{inside the core } (r < r_0): \quad
w_x = V_0 \frac{d}{r_0}, \quad w_{\mathrm{up}} = -V_0 \frac{l}{r_0}
$$

$$
\text{outside the core}: \quad
w_x = V_0 \frac{r_0 \, d}{r^2}, \quad w_{\mathrm{up}} = -V_0 \frac{r_0 \, l}{r^2}
$$

The array is the sum of its vortices. `wind.line_vortex_wind` uses the same equations for vortex
lines at an angle to the flight path, so that the field changes across the span.
`wind.lamb_oseen_wind` has a smooth core.

### 7.4 Updraft, lee wave and microburst

| Field | Vertical wind | Source |
|---|---|---|
| Updraft column | $w_{\mathrm{up}} = w_0 \exp\left(-(r/R)^s\right)$, with the distance $r$ from the axis | Wingrove and Bach (1994), magnitudes |
| Lee wave | $w_{\mathrm{up}} = -w_0 \cos\left(2\pi x / \lambda\right)$ | Doyle et al. (2011), amplitudes |
| Microburst | axisymmetric stagnation flow, Equations 5 and 6 of the source | Oseguera and Bowles (1988) |

The sharpness $s$ of the updraft is a declared value. The source does not give it.

### 7.5 Dryden turbulence

The Dryden spectra come from MIL-F-8785C. The scale length $L$ is 1750 ft:

$$
\Phi_w(\Omega) = \sigma^2 \frac{L}{\pi} \, \frac{1 + 3 (L\Omega)^2}{\left(1 + (L\Omega)^2\right)^2},
\qquad
\Phi_u(\Omega) = \sigma^2 \frac{2L}{\pi} \, \frac{1}{1 + (L\Omega)^2}
$$

The lateral component uses the same form as the vertical component. `wind.dryden_field` makes
each component as a sum of 400 cosines. The wavelengths are between 20 m and 40 km, with a
logarithmic spacing and a random phase. The seed sets the phases. Thus the field is frozen, and
the three components are independent.

### 7.6 F-factor

The F-factor is the hazard index of Proctor et al. (2000):

$$
F = \frac{\dot{U}_x}{g} - \frac{w_{\mathrm{up}}}{V}
$$

$U_x$ is the horizontal wind along the ground track. `dynamics.f_factor` and
`dynamics.average_f_factor` calculate it. `dynamics.thrust_authority` gives the value of $F$
that the thrust of the aircraft can hold.

## 8 Numerical methods

### 8.1 Integration

`integrate.step` does one step of the classical fourth-order Runge–Kutta method, with a fixed
time step. The model samples the wind one time for each step, and uses that wind in all four
stages (assumption E4). After each step, the model sets the quaternion norm to 1.

`integrate.rollout` does the steps in `jax.lax.scan`. JAX compiles the rollout again for each new
number of steps, and for each new wind model.

### 8.2 Precision

All calculations use 64-bit floating-point numbers. The Newton trim to 1e-10 and the quaternion
norm over 100,000 steps are not stable with 32-bit numbers.

### 8.3 Trim

`trim.trim` finds the angle of attack, the elevator and the throttle for steady level flight. It
uses Newton iteration on $[\dot{u}, \dot{w}, \dot{q}] = 0$, with the Jacobian from `jax.jacfwd`.
The trim is in still air.

### 8.4 Linear modes

`validation.longitudinal_matrix` linearizes the dynamics about a trim point with `jax.jacfwd`.
The eigenvalues of the matrix give the phugoid and short-period modes.
`validation.lateral_modes` gives the Dutch roll, roll and spiral modes.

### 8.5 Sensitivity

`atisim.sensitivity` calculates the derivative of a result with respect to a coefficient of the
aircraft. It gives the elasticity $(\partial Q / \partial p)(p / Q)$, which has no units. For the
trim, it uses the implicit function theorem. For the modes, it uses first-order eigenvalue
perturbation. This method is not correct for two eigenvalues that are almost equal.

## 9 Assumptions

The code refers to each assumption by its identifier. Each row gives the effect of the
assumption, and a measured bound when one is available.

### 9.1 Frames and Earth

| ID | Assumption | Effect |
|---|---|---|
| A1 | The Earth is flat and does not turn. NED is an inertial frame. | At 747 cruise, the Coriolis term is 0.0035 g at most, and the transport term is 0.0009 g at most. In 20 s, the position error is 6.9 m at most. Gravity does not change with latitude. This gives an error of 0.53% or less. |
| A4 | There is no ground. | A run can go below altitude 0. Stop the run at the clearance that you need, as `scripts/microburst.py` does. |

### 9.2 Mass and structure

| ID | Assumption | Effect |
|---|---|---|
| B1 | The airframe is rigid. | The CR-2144 derivatives are for a flexible airframe. At 2.48 times the dynamic pressure, all derivatives are less stiff, as aeroelastic relief predicts. This is the largest open assumption. Do not use AtiSim for structural response. |
| B2 | The mass and inertia are constant. | The fuel flow at 747 cruise is about 2.9 kg/s. This is small for an encounter. |
| B3 | The aircraft is symmetric: $I_{xy} = I_{yz} = 0$. | The sources give only $I_{xz}$. The model uses $I_{xz}$. |
| B4 | The accelerometer is at the center of gravity. | A flight recorder accelerometer also measures rotational terms. Thus use recorded loads only to find which encounter is worse. Do not compare the values. |
| B5 | The slug mass of the 747 uses $g_0 = 32.174$ ft/s². | The relative error is 1.5e-6. |

### 9.3 Aerodynamics

| ID | Assumption | Effect |
|---|---|---|
| C1 | Lift is linear in $\alpha$, except for aircraft with a lift table. | There is no stall. Results above about 10° are not valid. |
| C2 | The aerodynamics are quasi-steady, with no unsteady lag. | The model uses the $\dot\alpha$ derivatives of the source where the source gives them. Without $C_{m_{\dot\alpha}}$, the short-period damping is 12% too low. |
| C3 | The stability derivatives are constant in the flight envelope. | The 747 has the Mach derivatives of CR-2144. All other changes with $\alpha$ and altitude are frozen. Give the excursion with each result. |
| C4 | The drag is a parabolic polar with a Korn wave-drag rise. | The error in $C_D$ is 0.004 near the fit point, and up to 0.014 below Mach 0.75. |
| C5 | The engines have no dynamics. | The thrust changes immediately. Do not use results that need a powered recovery. |
| C6 | The control surfaces move immediately. | A 747 elevator needs about 0.24 s for a full maneuver deflection. This is 3.7% of the short period. |
| C7 | The drag uses the total $C_L$, with the rate and control terms. | This gives an $X_q$ term that CR-2144 does not have. The value agrees with the drag polar to 0.9%. |
| C8 | The wave drag uses the total $C_L$. | This gives a small, measured increase in drag at cruise. |
| C9 | All lift increments act at the relative wind of the center of gravity. | This breaks the energy balance at very high pitch rates, from 84 °/s to 201 °/s. No state inside the limits of operation shows it. |
| C10 | The aileron deflection is one angle for a compound control. | This agrees with the definition in the source derivatives. Do not compare `aileron_limit` with the travel of one surface. |
| C11 | The two estimates of the tail arm do not agree. | The difference is 1.4% to 2.9% on the 747, and 2.0 to 2.9 times on the light aircraft. Do not change a derivative to make them agree. |

### 9.4 Atmosphere

| ID | Assumption | Effect |
|---|---|---|
| D1 | The atmosphere is the ISA exactly, and the air is dry. | Each wind field comes from a source that gives its own conditions. |
| D2 | The atmosphere has no top and no bottom. | Above 11 km, the temperature stays at 216.65 kelvin. Below sea level, the lapse rate continues. |

### 9.5 Wind and turbulence

| ID | Assumption | Effect |
|---|---|---|
| E1 | The aircraft does not change the air. | This is correct for atmospheric fields. It is not correct for a wake vortex of a different aircraft. |
| E2 | The aircraft is a point for the gust, with first-order gust rates. | This is correct for gusts larger than 3 wingspans. For the Hannibal cores (2.3 to 3.1 spans), a sample across the airframe changes the load by 4.4% or less. |
| E3 | The fields are frozen. The wind is a function of position, not time. | This is correct for vortices, updrafts, microbursts and lee waves, which change slowly during an encounter. |
| E4 | The wind is constant during one integration step. | A random field stays the same when the time step changes. But in a field that changes in space, the method is first order. At the usual time step, this changes the pitch angle in a vortex core by 0.82%. |
| E5 | The sum of two fields is exact, except for the ground condition of the microburst. | A field with wind at the ground, added to a microburst, puts wind through the ground. |
| E6 | The model of the microburst has no outflow depth. | The downdraft on the axis increases with altitude to a limit. Use the microburst only at low altitude. |
| E7 | The along-track shear needs the turn rate. | The turn term gives $\Delta F = 0.14$ in a standard-rate turn. In straight flight, it is zero. |
| E8 | The longitudinal stations are all aft of the center of gravity. | The sampled pitch rate is a backward difference. It has a bias on the upstream half of a vortex core. |
| E9 | At the Rankine core boundary, the model uses the outer equation. | The two equations agree to 1e-13 at the boundary. No value shows the difference. |
| E10 | Only the line-vortex field changes across the span. | Other fields cannot roll the aircraft. No validation of the lateral response is available. |
| E11 | A response spectrum needs a stationary record. | With fixed controls, the airspeed can change by 13% in 100 s. Give this drift with each spectrum. |
| E12 | The published descriptions of the Hannibal vortex do not agree. | The model uses each set of core radius, speed and spacing as its paper gives it. A mixed set changes the peak-to-peak load by about 4%. |

### 9.6 Numerical methods

| ID | Assumption | Effect |
|---|---|---|
| F1 | The integrator is fixed-step RK4. | The measured order of accuracy is 3.99982 on a closed-form problem, and 3.989 on the full dynamics. |
| F2 | The integrator sets the quaternion norm to 1 after each step. | The norm error is 1.1e-16 after 3,000 steps. |
| F3 | All calculations use 64-bit numbers. | 32-bit numbers are not accurate enough for the trim and the quaternion norm. |
| F4 | Round-off limits the difference between two trajectories. | At 40,000 ft, the limit is about 7e-11 m. A time step smaller than 1/128 s makes the result less accurate. |
| F5 | The strip integral uses 9 stations across the span. | It gives 82.6% of its calibration value. About 56 stations give 1%. This affects only the rolling moment. |
| F6 | The shape of the span loading is a global value. | This is correct in one thread. Do not use the strip loads from more than one thread. |
| F7 | A control with zero effect makes the Newton solution NaN. | This occurs only in a trim that uses the rudder as an unknown. No such trim is in AtiSim. |

## 10 Verification and validation

Verification shows that the code solves the equations correctly. Validation shows that the
equations agree with the real aircraft and with the published data. The notebooks
`notebooks/solver-validation.ipynb` and `notebooks/validation-ladder.ipynb` do the checks again.

### 10.1 Verification

- The RK4 integrator has an observed order of accuracy of 3.99982 on a closed-form problem.
- The angular momentum changes by 5.7e-13 in 600 s of torque-free rotation.
- The torque-free rotation agrees with its analytic solution.
- A uniform wind moves the trajectory but does not change the attitude.
- A wind that is uniform but changes with time gives no body force.

### 10.2 Validation against the source data

This table compares the linear modes of `boeing747` with Table IX-5 of NASA CR-2144, at Mach 0.8
and 40,000 ft.

| Mode | Parameter | Error of the model |
|---|---|---|
| Short period | natural frequency | −1.32% |
| Short period | damping | +0.57% |
| Phugoid | natural frequency | +1.68% |
| Phugoid | damping | +1.38% |

Three independent implementations of the same data agree with the model:

- The worked example of Caughey agrees with each element of the plant matrix that the model contains.
- JSBSim, with the same coefficients, gives the same trim and modes. The phugoid frequency of the 737 is in 0.45% of the JSBSim value.
- The code of Yoshimura et al. gives the same eigenvalues, in 0.01% for the short period and 0.20% for the phugoid.

### 10.3 Validation against a recorded encounter

On 3 April 1981, a DC-10 flew through a row of clear-air-turbulence vortices at 37,000 ft near
Hannibal, Missouri. Mehta (1987) found the wind field from the flight recorder. NASA TM-102186
gives the load that the aircraft measured.

The 747 model flies through this field on the path of the DC-10. The model gives **70.2% of the
recorded peak-to-peak load**: 1.90 g, against 2.70 g.

A match is not expected, for these reasons:

- The recorded aircraft was a DC-10-10. No DC-10 derivative set is available.
- The weight of the aircraft is not known.

The 747 derivatives come only from their source, and not from their effect on this result.

### 10.4 Validation against published orders and mechanisms

- **TM-102186 Figure 8.** Three aircraft with different speeds fly through the same vortices. The model gives the same order of loads as the paper. It also agrees with the mechanism in the paper in 6 of 6 cases.
- **Wingrove and Bach discriminator.** The pitch angle in a vortex, an updraft and a pilot maneuver increases in that order, as the paper gives.
- **Frequency response.** The load spectrum of the Hannibal run has its peak in one frequency bin of the short period. In a Dryden ensemble, the response at the short period is more than 3 times the response at 0.05 Hz. But the input has more energy at 0.05 Hz.

### 10.5 Predictions before the result

`atisim.predictions` holds predictions with a digest and a commit, from before the run that
gave the result. Three predictions have a result. Two are correct, and one is incorrect on one
panel of four. One prediction has no result yet.

### 10.6 Uncertainty

These values give the size of the known uncertainties in the headline load:

| Source of uncertainty | Effect on the load |
|---|---|
| Gust sampled at one point (E2) | 4.4% or less |
| Wind constant during one step (E4) | 0.82% of the pitch change in a core |
| Different published vortex parameters (E12) | about 4% |
| Different aircraft (DC-10 against 747) | not known |

All comparisons with a recorded encounter use summary values, such as peaks, spacings and
orders. No comparison uses a full time history.

## 11 Known limitations

- The strip load path counts the rolling moment of the gust two times, when you set `vortex_viz.fly(strip=True)`. This affects only lateral results.
- The Earth is flat and does not turn.
- The lift model has no stall.
- The validation of AtiSim does not include absolute loads.

The references for this manual are in {doc}`references`.
