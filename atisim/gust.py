"""Closed-form gust response theory, and the verification it makes possible.

`wind.py` builds gust FIELDS and `response.py` analyses the time series a run
produces. Nothing between them ever answered the question this module exists
for: **what does the aeroplane's own linearisation say the load should be?**

`test_verification.py` checks the integrator against a closed-form solution
and measures RK4's observed order at 3.99982. This module does the equivalent
for the response to a gust: it compares a flown run against an exact answer
rather than against a document.

Four things live here:

  `gust_transfer`      H(Omega): vertical gust to n_z, from the aircraft's own
                       linearised plant plus the frozen-field geometry that
                       turns one spatial sinusoid into three input channels.
  `mean_square_ratio`  Abar^2 = integral |H|^2 Phi dOmega -- the PSD identity
                       that turns H into an rms, and the factor-of-two trap it
                       is easy to fall into.
  `sears` / `kussner_attenuation`
                       The unsteady lag on the GUST's arrival, which is a
                       different effect from `Cmadot` and is bounded nowhere
                       else in this project.
  `pratt_walker`       The discrete-gust design formula, for the comparison
                       that has a predicted SIGN rather than a predicted
                       agreement.

**WHAT IS AND IS NOT INDEPENDENT HERE, because the value of a verification is
exactly its independence.** `gust_transfer` differentiates the SAME
`dynamics.derivatives` the simulator integrates, so it is not an independent
aerodynamic model and must never be quoted as one. What it is independent of is
the whole TIME DOMAIN path: trim, the RK4 rollout, `wind.field_model`'s
composition of the three gust channels, `vortex_viz`'s measurement of n_z, and
every sign in between. Those are what `measure_gust_transfer` tests, and they
carry every turbulence number the model produces.

The gust channel gains in `frozen_sinusoid_channels` ARE derived here from
scratch, on paper, and are checked against `wind.gust_rates` and
`wind.gust_alphadot` rather than taken from them -- so a sign error in either
is a failure and not a shared assumption.
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import hankel2, jv

from atisim import dynamics, trim, wind
from atisim.aircraft import Aircraft
from atisim.state import Controls, State, euler_to_quat


# ---------------------------------------------------------------------------
# V1 -- the transfer function from a frozen vertical gust to n_z
# ---------------------------------------------------------------------------
#
# THE THREE CHANNELS, AND WHY THERE ARE THREE. A frozen field is a function of
# POSITION, so an aircraft flying through it at ground speed Vg meets a
# sinusoid in time. It arrives at `dynamics.derivatives` through three separate
# arguments, not one:
#
#   wind_ned          the gust velocity itself
#   omega_gust        its gradient across the airframe -- a pitching input
#   alphadot_gust     its rate of change along the path -- a plunging input
#
# A linearisation that used only the first would be wrong by whatever the other
# two are worth, and would look exactly like a correct one. Both of the others
# carry a factor of the spatial frequency Omega, so their weight grows with
# frequency and vanishes at the low-frequency end -- which is why an error in
# them is invisible in a sweep that stops at the phugoid.
#
# DERIVATION. Let the vertical gust be w_up(N) = Re{a e^{i Omega N}}, positive
# UP, N the along-track (north) coordinate. NED z is DOWN, so
#
#     wind_ned = [0, 0, -w_up]                       => channel gain -1
#
# Its Jacobian J = d(wind_ned)/d(pos_ned) has the single element
#
#     J[2,0] = d(wind_ned[2])/dN = -i Omega a e^{i Omega N} == G
#
# `wind.gust_rates` forms grad_body = C^T J C with C the body->NED rotation and
# returns q_gust = -grad_body[2,0]. For a wings-level pitch attitude theta,
# C[0,0] = cos(theta) and C[2,2] = cos(theta), and J's single element picks out
#
#     grad_body[2,0] = C[2,2] J[2,0] C[0,0] = G cos^2(theta)
#     q_gust = -G cos^2(theta)               => channel gain +i Omega cos^2(theta)
#
# `wind.gust_alphadot` forms wind_rate_ned = J @ vel_ned = [0, 0, G Vg] and
# then alphadot = (u_rel wdot_rel - w_rel udot_rel)/(u_rel^2 + w_rel^2) with
# rel_rate_body = -C^T wind_rate_ned. Working it through at the trim state,
# where u_rel = V cos(alpha), w_rel = V sin(alpha) and theta = alpha exactly
# (`trim.trimmed_state`),
#
#     alphadot = -G (Vg/V) cos(theta - alpha) = -G   => channel gain +i Omega
#
# The cos^2(theta) on the pitching channel and the bare 1 on the plunging one
# are NOT the same number -- 0.9961 against 1 at this trim -- and writing one
# for the other is the kind of error this file exists to make visible.
# `test_gust.py` asserts all three against `wind`'s own evaluation of a real
# sinusoidal field, so the paper above is checked rather than believed.


def frozen_sinusoid_channels(theta: float) -> np.ndarray:
    """Per-unit-Omega gust channel gains for a frozen vertical sinusoid.

    Returns the complex triple (wind_ned[2], omega_gust[1], alphadot_gust) that
    a unit-amplitude UPWARD gust w_up(N) = e^{i Omega N} presents, written as
    `c[0] + 1j * Omega * c[1]` per channel -- i.e. the first column is the
    in-phase part and the second the part that scales with spatial frequency.

    `theta` is the trim pitch attitude in radians. It is an argument rather
    than being recovered from a state because the derivation above is only
    valid AT the trim it linearises about, and passing it makes that explicit.
    """
    return np.array([
        [-1.0, 0.0],                      # wind_ned[2] = -w_up
        [0.0, np.cos(theta) ** 2],        # q_gust      = +i Omega cos^2(theta)
        [0.0, 1.0],                       # alphadot    = +i Omega
    ])


def _linearise(ac: Aircraft, alpha: float, elevator: float, throttle: float,
               V: float, H: float):
    """(A, B, C, D) for [u, w, q, theta, z] against the three gust channels.

    **FIVE STATES, NOT THE TEXTBOOK FOUR, AND THAT IS A MEASURED CHOICE.**
    `validation.longitudinal_matrix` carries [u, w, q, theta] and holds the
    altitude fixed, which is the standard longitudinal model and is the right
    one for comparing against CR-2144's own published matrices -- they are
    constant-density too. It is NOT the system this simulator integrates:
    `dynamics.derivatives` reads `density(altitude)` and `gravity(altitude)`
    afresh at every call, so height is a genuine fifth state with genuine
    feedback into the vertical force.

    Phase V1 measured what leaving it out costs, by flying the free response to
    a 1 m/s plunge in still air and comparing it against `expm(A t) x0`:

        four states   6% to 26% disagreement over 20 s
        five states   0.1% to 1.4% over the same record

    and the phugoid root moves from 0.0699 to 0.0788 rad/s -- **13%** -- between
    the two. The four-state phugoid is not the one this simulator exhibits, and
    `docs/PROJECT.md` §6(i) records that separately because it is a statement
    about `validation.py` rather than about this module.

    `z` is `pos_ned[2]`, i.e. DOWN-positive, so it is the negative of altitude.
    Taking the state straight off `State` rather than flipping the sign here
    keeps the Jacobian something that can be read against `dynamics.py`.

    **D'S THIRD ENTRY IS STRUCTURALLY ZERO, AND THAT IS A PROPERTY OF THE
    MEASUREMENT PATH RATHER THAN A SIMPLIFICATION.** `dynamics.load_factor`
    takes no `alphadot_gust` argument at all, so the n_z this project measures
    cannot see that channel and the derivative with respect to it is zero by
    construction. It is HARMLESS for every aircraft in the registry because
    each declares `CLadot = 0`, so alphadot reaches C_m and never C_L; with a
    non-zero `CLadot` the measurement would be missing a real term.
    `test_gust.py::test_the_load_factor_path_cannot_see_alphadot` pins that
    condition rather than leaving it as a comment.
    """
    u0, w0 = V * np.cos(alpha), V * np.sin(alpha)
    controls = Controls(
        elevator=jnp.array(elevator), aileron=jnp.array(0.0),
        rudder=jnp.array(0.0), throttle=jnp.array(throttle),
    )
    x0 = jnp.array([u0, w0, 0.0, alpha, -H])
    g0 = jnp.zeros(3)

    def state_of(x):
        u, w, q, theta, z = x
        return State(
            pos_ned=jnp.array([0.0, 0.0, z]),
            vel_body=jnp.array([u, 0.0, w]),
            quat=euler_to_quat(jnp.array(0.0), theta, jnp.array(0.0)),
            omega=jnp.array([0.0, q, 0.0]),
        )

    def f(x, g):
        d = dynamics.derivatives(
            state_of(x), controls, ac,
            jnp.array([0.0, 0.0, g[0]]), jnp.array([0.0, g[1], 0.0]),
            None, g[2],
        )
        return jnp.array([
            d.vel_body[0], d.vel_body[2], d.omega[1], x[2], d.pos_ned[2],
        ])

    def n_of(x, g):
        return dynamics.load_factor(
            state_of(x), controls, ac,
            jnp.array([0.0, 0.0, g[0]]), jnp.array([0.0, g[1], 0.0]),
        )

    A = np.asarray(jax.jacfwd(f, 0)(x0, g0))
    B = np.asarray(jax.jacfwd(f, 1)(x0, g0))
    C = np.asarray(jax.jacfwd(n_of, 0)(x0, g0))
    D = np.asarray(jax.jacfwd(n_of, 1)(x0, g0))
    return A, B, C, D


def gust_transfer(ac: Aircraft, V: float, H: float, spatial_frequency,
                  ground_speed: float | None = None,
                  q_rolloff_span: float | None = None,
                  q_rolloff_phase: str = "minimum"):
    """H(Omega): complex n_z per unit UPWARD gust velocity, in g per (m/s).

    `spatial_frequency` is Omega in rad/m -- the axis `wind.dryden_spectrum` is
    written on, so `mean_square_ratio` can integrate the two together without a
    conversion step that could carry a factor of 2 pi.

    `ground_speed` defaults to `V`. It is separable because the Taylor
    hypothesis maps space to time through the GROUND speed and the aerodynamics
    through the AIRSPEED, and in still-air trim those are equal only because
    there is no mean wind. A measured run can hand its own mean ground speed
    here rather than having one assumed for it.

    Sign convention: POSITIVE Omega, positive gust UP, and H -> the static
    gust-lift value as Omega -> 0 rather than to zero. **That static limit is
    NOT what a sustained updraft does to a free aeroplane** -- given time it
    climbs with the air and the load returns to trim -- and the difference is
    the phugoid, which lives in A and is therefore already in H. At Omega
    exactly zero the two disagree and H's value there is the frozen-attitude
    one; nothing in this project evaluates it there.

    `q_rolloff_span` multiplies the pitching
    gust channel by a first-order roll-off whose magnitude is MIL-F-8785C's,
    1/|1 + i(4b/pi)Omega| with b the span in metres -- section 3.7.5, printed
    p. 58, whose Phi_q is that magnitude squared times Omega^2 Phi_w. The
    specification fixes only the magnitude, so the phase is a DECLARED choice,
    `q_rolloff_phase`. "minimum", the default, is the causal first-order lag
    1/(1 + i(4b/pi)Omega), the conventional Dryden pitching-gust filter;
    "zero" applies the magnitude alone. They differ materially in sigma_nz,
    because the lag enters through its cross term with the w channel. The
    alphadot_gust channel is left unfiltered, also a DECLARED choice: p. 58
    prints -alphadot_g = q_g = dw_g/dx, but gives alphadot_g no spectrum and
    treats w_g as uniform over the aircraft, and alphadot reaches n_z here
    only through Cmalphadot. None, the default span, is the field as flown:
    the pure gradient, no roll-off.
    """
    if q_rolloff_phase not in ("minimum", "zero"):
        raise ValueError(f"unknown q_rolloff_phase {q_rolloff_phase!r}")
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha, elevator, throttle = float(x[0]), float(x[1]), float(x[2])
    A, B, C, D = _linearise(ac, alpha, elevator, throttle, V, H)
    channels = frozen_sinusoid_channels(alpha)  # theta == alpha at trim
    Vg = V if ground_speed is None else ground_speed

    Omega = np.atleast_1d(np.asarray(spatial_frequency, dtype=float))
    out = np.empty(Omega.shape, dtype=complex)
    eye = np.eye(A.shape[0])
    for i, om in enumerate(Omega):
        gains = channels[:, 0] + 1j * om * channels[:, 1]
        if q_rolloff_span is not None:
            corner = 1.0 + 1j * (4.0 * q_rolloff_span / np.pi) * om
            gains[1] = gains[1] / (corner if q_rolloff_phase == "minimum"
                                   else abs(corner))
        omega = om * Vg  # rad/s -- the Taylor hypothesis, and the only place it enters
        resolvent = np.linalg.solve(1j * omega * eye - A, B @ gains)
        out[i] = C @ resolvent + D @ gains
    return out if np.ndim(spatial_frequency) else out[0]


# ---------------------------------------------------------------------------
# V2 -- the PSD identity
# ---------------------------------------------------------------------------


def mean_square_ratio(ac: Aircraft, V: float, H: float, spectrum,
                      *, n_points: int = 4000,
                      wavelength_min: float = 20.0,
                      wavelength_max: float = 40_000.0,
                      q_rolloff_span: float | None = None,
                      q_rolloff_phase: str = "minimum") -> float:
    """Abar^2 = integral |H(Omega)|^2 Phi(Omega) dOmega / sigma^2, per (m/s)^2.

    `spectrum` is called as `spectrum(Omega, 1.0)` -- a UNIT-sigma spectrum, so
    the ratio returned is independent of intensity and `sigma_nz = Abar *
    sigma_w` is exact for a linear system. That is the whole content of the
    identity and it is why this returns a ratio rather than an rms.

    **THE FACTOR OF TWO.** `wind.dryden_spectrum` is ONE-SIDED: its integral
    over the half line 0..inf is sigma^2, not 2 sigma^2, and
    `test_cat_validation.py::test_the_dryden_spectrum_is_one_sided` measures
    that. So the integral here runs over positive Omega only and carries NO
    factor of 2. Writing the two-sided form would inflate sigma_nz by sqrt(2) --
    41%, which is larger than every discrepancy this project is trying to
    explain and would be attributed to physics.

    **THE INTEGRATION LIMITS ARE THE REALISATION'S, NOT INFINITY**, and that is
    deliberate rather than lazy. `wind.dryden_vertical_field` sums components
    between these same two wavelengths, so a realisation cannot contain the
    tails and an identity integrated over the whole line would be comparing
    against variance the field does not have. `test_gust.py` measures what the
    omitted tails are worth so the choice is priced rather than assumed.

    Log-spaced, for the reason `dryden_vertical_field` is: the Dryden spectrum
    falls as Omega^-2 over three decades and a linear grid spends almost all of
    itself on the flat tail.
    """
    lo = 2.0 * np.pi / wavelength_max
    hi = 2.0 * np.pi / wavelength_min
    Omega = np.geomspace(lo, hi, n_points)
    Phi = np.asarray(spectrum(jnp.asarray(Omega), 1.0), dtype=float)
    Hmag2 = np.abs(gust_transfer(ac, V, H, Omega, q_rolloff_span=q_rolloff_span,
                                 q_rolloff_phase=q_rolloff_phase)) ** 2
    return float(np.trapezoid(Hmag2 * Phi, Omega))


def realisation_mean_square_ratio(ac: Aircraft, V: float, H: float,
                                  sigma_w: float, seed: int, **kwargs):
    """The EXACT response variance of one Shinozuka realisation, and its own.

    Returns `(sigma_nz_squared, sigma_w_squared)` in g^2 and (m/s)^2.

    `wind.dryden_vertical_field` is a sum of sinusoids with FIXED amplitudes
    A_k and random phases. Driven through a linear H, its response over a long
    record has variance `sum |H(omega_k)|^2 A_k^2 / 2` EXACTLY -- there is no
    integral being approximated here, so nothing in this number can be blamed
    on discretisation.

    **THAT IS WHY BOTH HALVES ARE RETURNED.** The realisation's own variance
    `sum A_k^2 / 2` is NOT sigma_w^2: the 400-component log grid is a Riemann
    sum of the integral that defines it and comes out 1.63% light at the
    default settings. Comparing a flown sigma_nz against `Abar * sigma_w`
    therefore mixes two errors -- the model's and the grid's -- while comparing
    the RATIO sigma_nz/sigma_w_realised against this one cancels the grid
    exactly. Report both: it is the only way to see which of the two any
    disagreement belongs to.
    """
    omega, amplitude, _ = wind.dryden_vertical_components(sigma_w, seed, **kwargs)
    omega = np.asarray(omega, dtype=float)
    power = np.asarray(amplitude, dtype=float) ** 2 / 2.0
    Hmag2 = np.abs(gust_transfer(ac, V, H, omega)) ** 2
    return float(np.sum(Hmag2 * power)), float(np.sum(power))


# ---------------------------------------------------------------------------
# V4 -- the unsteady lag on the GUST's arrival
# ---------------------------------------------------------------------------
#
# WHY THIS IS NOT `Cmadot`, which is the confusion the whole phase exists to
# end. `docs/ASSUMPTIONS.md` C2 is titled "Aerodynamics are quasi-steady: no
# alphadot or unsteady lag" and calls itself the best-characterised assumption
# in the project -- but every number in its bound is about alphadot, the lag on
# the aircraft's OWN motion, which `Cmadot` carries. The lag on the GUST's
# arrival is Sears' problem, not Theodorsen's, and it was bounded nowhere.
#
# Sears, "Some Aspects of Non-Stationary Airfoil Theory and Its Practical
# Application", J. Aeronautical Sciences 8(3), 1941, 104-108. The sinusoidal
# gust response of a thin aerofoil is S(k) times the quasi-steady value, with
#
#     S(k) = [J0(k) - i J1(k)] C(k) + i J1(k)
#
# and C(k) the Theodorsen function. k = omega c / (2 V) is the SEMICHORD
# reduced frequency, which is the convention both Sears and Theodorsen are
# written in; a chord-based k would halve every number below.


def theodorsen(k):
    """C(k) = H1(k) / (H1(k) + i H0(k)), Hankel functions of the second kind."""
    k = np.asarray(k, dtype=float)
    h1, h0 = hankel2(1, k), hankel2(0, k)
    return h1 / (h1 + 1j * h0)


def sears(k):
    """Sears' function S(k). |S| < 1 is attenuation; arg S < 0 is lag."""
    k = np.asarray(k, dtype=float)
    return (jv(0, k) - 1j * jv(1, k)) * theodorsen(k) + 1j * jv(1, k)


def reduced_frequency(omega: float, chord: float, V: float) -> float:
    """k = omega c / (2 V). Semichord, per Sears and Theodorsen."""
    return omega * chord / (2.0 * V)


def kussner_attenuation(omega: float, chord: float, V: float):
    """(fractional loss of lift, phase lag in degrees) at a gust frequency.

    Named for Kussner because the effect bounded here is the Kussner gust lag;
    the frequency-domain statement of that lag IS Sears' function, Kussner's
    indicial response being its transform. One effect, two authors, and taking
    them for two would double-count it.

    **THIS IS THE LOSS OF LIFT MAGNITUDE, NOT THE EFFECT ON THE LOAD.** The lag
    also shifts the phase, and the load depends on both. Modelled with its
    phase (Jones's approximation to Kussner's function), the lag changes
    sigma_nz over the Dryden band by only -0.6% (747) to -1.0% (737), and near
    the short period it RAISES the load by about 1%. Applying |S| to |H| alone
    gives -6.78% and overstates it. Thin-aerofoil, incompressible and unswept,
    so approximate at the Mach numbers and sweeps this model flies.
    """
    k = reduced_frequency(omega, chord, V)
    S = sears(k)
    return float(1.0 - abs(S)), float(np.degrees(np.angle(S)))


# ---------------------------------------------------------------------------
# V3 -- Pratt & Walker's discrete gust formula
# ---------------------------------------------------------------------------
#
# K. G. Pratt and W. G. Walker, "A Revised Gust-Load Formula and a Re-Evaluation
# of V-G Data Taken on Civil Transport Airplanes From 1933 to 1950", NACA
# Report 1206, 1953. The formula below is its Eq. (5) with the alleviation
# factor of Eq. (7), and it is the form that became the certification standard.
#
# IT IS AN EMPIRICAL FIT, NOT A DERIVATION, and that is what makes the
# comparison worth making. K_g was fitted so that a RIGID, PITCH-RESTRAINED
# aeroplane penetrating a 1-cosine gust of 12.5 chords gradient reproduces
# measured V-G records. A free-to-pitch simulation sheds load the fit never
# saw. So the expected result is a DISCREPANCY WITH A SIGN, and agreement
# would be the surprise.


def mass_ratio(ac: Aircraft, rho: float, g: float) -> float:
    """mu = 2 W / (rho g c CLa S). Pratt & Walker Eq. (6), non-dimensional."""
    W = float(ac.mass) * g
    return 2.0 * W / (rho * g * float(ac.c) * float(ac.CLa) * float(ac.S))


def alleviation_factor(mu: float) -> float:
    """K_g = 0.88 mu / (5.3 + mu). Pratt & Walker Eq. (7)."""
    return 0.88 * mu / (5.3 + mu)


def pratt_walker(ac: Aircraft, rho: float, V: float, U_de: float,
                 g: float) -> float:
    """Peak load increment dn, in g, for a derived gust velocity U_de.

    dn = K_g U_de V CLa rho S / (2 W). Pratt & Walker Eq. (5).

    `U_de` is the DERIVED gust velocity, which is a reference gust intensity
    and not the peak of any particular gust -- the formula's own alleviation
    factor is what converts one to the other. Handing it a measured peak gust
    and reading the answer as a prediction is the standard misuse and would
    understate the load by 1/K_g.
    """
    W = float(ac.mass) * g
    mu = mass_ratio(ac, rho, g)
    return alleviation_factor(mu) * U_de * V * float(ac.CLa) * rho * float(ac.S) / (2.0 * W)


# ---------------------------------------------------------------------------
# V1's other half -- measuring the same transfer function through the simulator
# ---------------------------------------------------------------------------
#
# THE PROBLEM A PLAIN SINE FIT DOES NOT SOLVE. The aeroplane is dropped into
# the field at t = 0 out of equilibrium with it, so the record is a driven
# sinusoid PLUS the free response of four modes. The phugoid's damping ratio
# here is 0.038 -- it decays by a factor e in 380 s -- so over any record short
# enough to keep the altitude excursion small, it is still there. Projecting
# the record onto cos and sin alone folds whatever the phugoid is doing at the
# window edges into the answer, and at 0.5% gates that is not a rounding error.
#
# WHAT IS DONE INSTEAD. The free response of a linear system is known in form
# once A's eigenvalues are: it is a sum of exp(lambda t). So the fit carries
# the drive quadratures AND a decaying quadrature pair per oscillatory mode,
# with the eigenvalues taken from the SAME linearisation the answer is being
# compared against. The fit is then exact for the linearised system, and its
# residual is the nonlinearity plus the integrator -- both of which are
# reported rather than assumed small.
#
# This is not circular. The eigenvalues fix only the SHAPE of the transient
# basis; the drive amplitude and phase are free parameters, and a wrong H
# cannot be hidden by any combination of decaying exponentials at frequencies
# the drive does not sit on.


def fit_at_frequency(t, y, omega: float, eigenvalues=()):
    """Complex amplitude of `y` at `omega`, transients regressed out.

    Returns `(Y, residual_rms, cond)` where the record is modelled as
    `Re{Y e^{i omega t}}` plus a constant plus one decaying quadrature pair per
    entry of `eigenvalues` with positive imaginary part.

    `cond` is the design matrix's condition number and is RETURNED RATHER THAN
    CHECKED, because the frequency at which it goes bad is exactly the
    interesting one: a drive sitting on the phugoid is nearly collinear with
    the phugoid's own free response over a short record, and the honest
    response to that is to report it and lengthen the record, not to hide it
    behind a tolerance.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    columns = [np.ones_like(t), np.cos(omega * t), np.sin(omega * t)]
    for lam in eigenvalues:
        if lam.imag <= 0:
            continue
        envelope = np.exp(lam.real * t)
        columns.append(envelope * np.cos(lam.imag * t))
        columns.append(envelope * np.sin(lam.imag * t))
    M = np.column_stack(columns)
    coeffs, *_ = np.linalg.lstsq(M, y, rcond=None)
    residual = y - M @ coeffs
    return (
        complex(coeffs[1], -coeffs[2]),
        float(np.sqrt(np.mean(residual ** 2))),
        float(np.linalg.cond(M)),
    )


def plant_eigenvalues(ac: Aircraft, V: float, H: float):
    """Eigenvalues of the linearised longitudinal plant, for `fit_at_frequency`."""
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    A, _, _, _ = _linearise(ac, float(x[0]), float(x[1]), float(x[2]), V, H)
    return np.linalg.eigvals(A)


def steady_state_seed(ac: Aircraft, V: float, H: float, wavelength: float,
                      amplitude: float):
    """The state a driven sinusoid settles into, for starting a run already in it.

    Returns `(state, controls)` for `vortex_viz.fly_from_state`, trimmed and then
    displaced by `Re{(i omega I - A)^-1 B a}` -- the linearised system's own
    steady response to the gust the run is about to fly through, at the phase the
    field has at north = 0.

    **WHY, AND WHY IT IS NOT CIRCULAR.** An aeroplane dropped into a gust field
    out of equilibrium with it rings: the record is the driven response PLUS the
    free response of five modes, and this aircraft's phugoid decays by only a
    factor e in 587 s. Regressing the modes out works (`fit_at_frequency` does
    it anyway, and it is what makes this optional rather than required) but
    leaves a residual 16x larger, and at the low-frequency end the drive sits
    near enough to the phugoid that the two are hard to separate over any
    affordable record.

    It cannot manufacture agreement. The seed sets the INITIAL CONDITION; the
    amplitude and phase the fit reads back are free parameters of a run that is
    thereafter driven by the real field through the real integrator. A wrong H
    shows up as a transient -- the seed would be the wrong steady state, and the
    aircraft would ring its way to the right one -- so a seeded run that does
    NOT ring is itself evidence. `measure_gust_transfer` reports the transient
    content as `load_residual` for exactly that reason.
    """
    x, _ = trim.trim(jnp.array(V), jnp.array(H), ac)
    alpha, elevator, throttle = float(x[0]), float(x[1]), float(x[2])
    A, B, _, _ = _linearise(ac, alpha, elevator, throttle, V, H)
    Omega = 2.0 * np.pi / wavelength
    gains = frozen_sinusoid_channels(alpha) @ np.array([1.0, 1j * Omega])
    d = np.linalg.solve(1j * Omega * V * np.eye(A.shape[0]) - A,
                        B @ gains).real * amplitude

    state = trim.trimmed_state(jnp.array(alpha), jnp.array(V), jnp.array(H))
    state = state._replace(
        pos_ned=jnp.array([0.0, 0.0, -H + d[4]]),
        vel_body=jnp.array([V * np.cos(alpha) + d[0], 0.0, V * np.sin(alpha) + d[1]]),
        omega=jnp.array([0.0, d[2], 0.0]),
        quat=euler_to_quat(jnp.array(0.0), jnp.array(alpha + d[3]), jnp.array(0.0)),
    )
    return state, trim.trimmed_controls(jnp.array(elevator), jnp.array(throttle))


def measure_gust_transfer(ac: Aircraft, V: float, H: float, wavelength: float,
                          *, amplitude: float = 0.5, periods: float = 12.0,
                          min_seconds: float = 400.0,
                          samples_per_period: float = 200.0,
                          dt_max: float = 0.05, dt_min: float = 0.002,
                          stage_sampled: bool = True, seed_steady: bool = True):
    """Fly one single-frequency gust and read H off the flown record.

    Returns a dict carrying the measured complex H in g per (m/s), the fitted
    input amplitude, both fits' residuals and condition numbers, the mean
    ground speed the Taylor hypothesis was evaluated at, and the altitude and
    incidence excursions -- which is what says whether the run stayed inside
    the linear regime it is verifying.

    `amplitude` is small on purpose, and what it buys was MEASURED rather than
    assumed. Swept over 2.0, 0.5, 0.125 and 0.03125 m/s at the short period the
    amplitude error reads -0.00089%, -0.00005%, -0.00000% and +0.00000%: it
    falls as the SQUARE of the gust, which is the signature of the aerodynamic
    nonlinearity and of nothing else, and by 0.5 m/s it is five parts in ten
    million. 0.5 m/s reaches 0.3745 deg of peak-to-peak incidence there, well
    inside `panel.ALPHA_LINEAR_DEG`.

    **`stage_sampled` DEFAULTS TO TRUE HERE AND TO FALSE EVERYWHERE ELSE**, and
    the asymmetry is the point. Every other run is compared against a document
    and keeps the frozen first-order default so its baseline does not move;
    this one is compared against an exact answer, so the integrator must not be
    what it measures. With the hold instead (assumption E4), the held run at the
    short period reads +3.20%, +1.62%, +0.81% and +0.41% as dt is halved from
    0.119 s, while the stage-sampled run sits at 0.0002% and does not move at
    all. Halving with dt is the wind hold's signature and not the response's.
    """
    from atisim import vortex_viz  # local: viz pulls matplotlib, gust.py must not

    Omega = 2.0 * np.pi / wavelength
    seconds = max(periods * wavelength / V, min_seconds)
    dt = float(np.clip(wavelength / (V * samples_per_period), dt_min, dt_max))

    field = wind.sinusoidal_vertical_field(amplitude, wavelength)
    label = f"lambda={wavelength:.0f}"
    if seed_steady:
        state, controls = steady_state_seed(ac, V, H, wavelength, amplitude)
        enc = vortex_viz.fly_from_state(
            ac, field, state, controls, label=label, seconds=seconds, dt=dt,
            window=(-np.inf, np.inf), window_name="whole run",
            stage_sampled=stage_sampled,
        )
    else:
        enc = vortex_viz.fly(
            ac, field, V, H, label=label,
            start_north=0.0, seconds=seconds, dt=dt,
            window=(-np.inf, np.inf), window_name="whole run",
            stage_sampled=stage_sampled,
        )

    ground_speed = float((enc.north[-1] - enc.north[0]) / (enc.t[-1] - enc.t[0]))
    omega = Omega * ground_speed
    eig = plant_eigenvalues(ac, V, H)

    w_fit, w_res, w_cond = fit_at_frequency(enc.t, enc.w_up, omega)
    n_fit, n_res, n_cond = fit_at_frequency(enc.t, enc.n_z, omega, eig)

    return {
        "wavelength": wavelength,
        "spatial_frequency": Omega,
        "frequency_hz": omega / (2.0 * np.pi),
        "ground_speed": ground_speed,
        "seconds": seconds,
        "dt": dt,
        "stage_sampled": stage_sampled,
        "seed_steady": seed_steady,
        "H": n_fit / w_fit,
        "input_amplitude": abs(w_fit),
        "input_residual": w_res,
        "load_residual": n_res,
        "load_amplitude": abs(n_fit),
        "input_cond": w_cond,
        "load_cond": n_cond,
        "altitude_excursion": float(np.ptp(enc.altitude)),
        "alpha_excursion_deg": float(np.degrees(np.ptp(enc.alpha_air))),
    }
