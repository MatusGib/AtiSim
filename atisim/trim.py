"""Steady level flight trim, on a rotating Earth.

SIX unknowns -- angle of attack, elevator, throttle, bank, aileron, rudder --
against the SIX residuals [udot, vdot, wdot, pdot, qdot, rdot], at a target true
airspeed and altitude with gamma = 0 and no body rates relative to ECEF.

Wings-level level flight stopped being an equilibrium when the Earth started
turning: Coriolis puts lateral acceleration into `vdot`, which the previous
three-unknown solver neither saw nor could cancel. Every run in this project
starts from trim, so a residual `vdot` would have put a slow lateral drift under
the whole evidence ledger.

HOW MUCH LATERAL ACCELERATION, MEASURED. The design estimated 3.5e-3 g, which is
`2 Omega V / g` -- IT OMITS sin(latitude). The horizontal Coriolis acceleration
is `2 Omega V sin(lat)`, so at the 747's cruise and 47N it is 2.58e-3 g, and the
bank that balances it is 0.1480 deg rather than the design's 0.2010. That factor
is not a rounding difference: it goes to zero on the equator and to `2 Omega V`
only at the pole.

THE COUNT IS EASY TO GET WRONG, and the design got it wrong once. Six residuals
against alpha, beta, phi, elevator, aileron, rudder and throttle is SEVEN
freedoms for six equations -- a one-parameter family, not a solution. It is
closed by imposing `beta = 0`, the coordinated condition every trim in this
project already assumed implicitly. That leaves six and six, square, which is
what lets Newton reach machine precision instead of wandering along the family.

`aileron` is in the set and is NOT optional, though NOT FOR THE REASON THE DESIGN
GAVE. The design said the rudder's `Cldr` makes `Cl` non-zero and the aileron is
there to cancel it. Measured on the 747 at 47N heading 030, the roll residual the
aileron has to remove is -1.053e-9 rad/s^2, of which `Cldr` supplies +1.161e-10 --
11%. The other 89% is the gyroscopic row `-I^-1 (Omega_b x I Omega_b)`, which is
O(Omega^2) and matches `-1.169044e-9` to every digit printed. Both are real and
neither vanishes, so the conclusion stands and the mechanism does not.

A RUDDERLESS AIRCRAFT CANNOT BE TRIMMED BY THIS SOLVER, and it fails as NaN, not
as a bad answer. `rudder` is an unknown, so an aircraft with `CYdr = Cldr =
Cndr = 0` -- which `aircraft.py` gives the Cessna 172 deliberately, see its
"*** THE RUDDER IS ABSENT ***" note -- has an identically zero sixth Jacobian
column. The system is rank 5 of 6 and `jnp.linalg.solve` returns NaN.
`test_every_real_aircraft_trims_to_a_physical_solution` pins that rank rather
than skipping past it. The fix, if such an aircraft ever needs a trim, is to
swap the unknown rather than to relax the solve: trade `rudder` for `beta` and
the system is square again through `Clb` and `Cnb`.

Newton with a forward-mode Jacobian. Pure JAX, so it is jittable and vmappable
over a grid of (V, h) if that is ever wanted. Every run starts from here: an
untrimmed start accelerates or climbs away for no reason and makes autopilot
tuning meaningless.
"""

import math
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array

from atisim import earth
from atisim.aero import wave_drag
from atisim.aircraft import Aircraft
from atisim.atmosphere import G0, density, speed_of_sound
from atisim.dynamics import derivatives
from atisim.state import Controls, State, euler_to_quat, state_from_ned


def trimmed_state(
    alpha: Array,
    phi: Array,
    airspeed: Array,
    altitude: Array,
    anchor: earth.Anchor,
    heading: Array,
) -> State:
    """Level flight at the given alpha and bank, with beta = 0.

    The level-flight constraint is NOT theta = alpha any more. With beta = 0 and
    gamma = 0 it is tan(theta) = cos(phi) tan(alpha), which reduces to the old
    form at phi = 0. At the bank this trim actually produces the two differ by
    well under a microradian, but the exact form is used because nothing
    downstream would reveal it if it were wrong.

    THE AIRCRAFT IS PLACED AT THE ANCHOR, so the trim altitude is `anchor.h` and
    `altitude` is carried only to keep the (alpha, V, h) call shape the rest of
    the module uses. Passing an `altitude` that differs from `anchor.h` does not
    move the aircraft. Every caller here builds the anchor at the same altitude
    it trims for.
    """
    theta = jnp.arctan(jnp.cos(phi) * jnp.tan(alpha))
    quat_ned = euler_to_quat(phi, theta, heading)
    vel_body = airspeed * jnp.array([jnp.cos(alpha), 0.0, jnp.sin(alpha)])
    return state_from_ned(
        jnp.array([0.0, 0.0, 0.0]), vel_body, quat_ned, jnp.zeros(3), anchor
    )


def trimmed_controls(elevator: Array, throttle: Array) -> Controls:
    return Controls(
        elevator=elevator,
        aileron=jnp.array(0.0),
        rudder=jnp.array(0.0),
        throttle=throttle,
    )


def residual(
    x: Array,
    airspeed: Array,
    altitude: Array,
    ac: Aircraft,
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
    heading: Array,
) -> Array:
    """All six: [udot, vdot, wdot, pdot, qdot, rdot].

    `vdot` and the two lateral moment rows are the ones the three-unknown solver
    could not see. They are not decoration: on a rotating Earth they are the
    equations the bank, aileron and rudder exist to satisfy.
    """
    alpha, elevator, throttle, phi, aileron, rudder = x
    d = derivatives(
        trimmed_state(alpha, phi, airspeed, altitude, anchor, heading),
        Controls(elevator=elevator, aileron=aileron, rudder=rudder, throttle=throttle),
        ac,
        jnp.zeros(3),
        jnp.zeros(3),
        anchor,
        earth_model,
    )
    return jnp.concatenate([d.vel_body, d.omega])


def minimum_drag_speed(
    ac: Aircraft, altitude: Array, low: float = 20.0, high: float = 400.0, n: int = 4000
) -> Array:
    """Level-flight speed of minimum drag, swept from the real drag model.

    This is the boundary of the autopilot's loop pairing. Below it the drag
    curve slopes the wrong way -- slowing down increases drag, which slows the
    aircraft further -- so throttle-to-airspeed and elevator-to-altitude stop
    being the right assignment. No gain set repairs that; it is the shape of the
    drag curve.

    The textbook closed form assumes a parabolic polar and is wrong wherever
    wave drag is active: it puts the 747's V_md 32 m/s ABOVE its own cruise
    speed. Sweeping the actual coefficients costs nothing here and is right for
    every aircraft in the registry.
    """
    rho = density(altitude)
    a_sound = speed_of_sound(altitude)
    speeds = jnp.linspace(low, high, n)

    def drag(V):
        qS = 0.5 * rho * V**2 * ac.S
        CL = ac.mass * G0 / qS
        CD = ac.CD0 + CL**2 / (jnp.pi * ac.e * ac.AR) + wave_drag(V / a_sound, CL, ac)
        return qS * CD

    return speeds[jnp.argmin(jax.vmap(drag)(speeds))]


# The Newton start point. A module constant rather than a literal inside `trim`
# so that verification.py measures the convergence of the actual solver instead
# of a hand-copied guess that could drift away from it.
INITIAL_GUESS = jnp.array([0.05, 0.0, 0.5, 0.0, 0.0, 0.0])


@partial(jax.jit, static_argnames=("iterations", "earth_model"))
def trim(
    airspeed: Array,
    altitude: Array,
    ac: Aircraft,
    anchor: earth.Anchor,
    earth_model: earth.EarthModel,
    heading: Array = 0.0,
    guess: Array = None,
    iterations: int = 40,
) -> tuple[Array, Array]:
    """Solve for [alpha, elevator, throttle, phi, aileron, rudder].

    Six unknowns, six residuals, square. Returns (solution, final residual).

    `anchor` and `earth_model` have no defaults, for the same reason a default
    anchor does not exist in `earth.py`: latitude changes the answer.

    `heading` DOES change the answer, but far less than the design expected, and
    the measurement is worth carrying because the expectation was wrong. Swept
    over 24 headings at 47N the trimmed bank fits

        phi(psi) = -0.14804001 - 0.00105617 sin(psi) deg

    to 1.3e-6 deg. The cos(psi) coefficient is -2.2e-7 deg, 1.5e-6 of the
    constant -- zero. So the bank is heading-INDEPENDENT to 0.7%, which is right
    for `2 Omega x V`: the local-vertical component of Omega is `Omega sin(lat)`
    and a rotation about the vertical gives a horizontal Coriolis acceleration of
    magnitude `2 Omega V sin(lat)` perpendicular to the track WHATEVER the track's
    azimuth. The cos(psi) terms that appear separately in `Omega_x v_z` and
    `Omega_z v_x` cancel identically once theta = alpha, which is exactly what the
    absent cos coefficient measures. The 0.71% sin(psi) residue is the east/west
    asymmetry -- the bank's own coupling back into `Omega_b` accounts for 0.24 of
    those points (predicted `2 Omega V phi cos(lat)`, measured 6.043e-5 m/s^2
    against 6.06e-5) and the vertical Coriolis channel for the rest.
    """
    x0 = INITIAL_GUESS if guess is None else guess

    def step(x, _):
        r = residual(x, airspeed, altitude, ac, anchor, earth_model, heading)
        jacobian = jax.jacfwd(residual)(
            x, airspeed, altitude, ac, anchor, earth_model, heading
        )
        # LSTSQ, NOT SOLVE, AND THIS CLOSES ASSUMPTIONS.md F7.
        #
        # F7 recorded that a control channel with zero authority makes a Newton
        # solve return NaN in silence, and called it "latent for the shipped
        # solvers" because "`trim` does not carry rudder as an unknown, so
        # nothing in the package hits it". Carrying rudder as an unknown is
        # exactly what this function now does, so the latent case went live: the
        # Cessna's CYdr = Cldr = Cndr = 0 makes the rudder column identically
        # zero, and `solve` returned [nan, nan, nan, nan, nan, nan].
        #
        # `lstsq` takes the minimum-norm step instead. Measured across the
        # registry at 47N: every aircraft converges, the Cessna's rudder comes
        # out at exactly 0.0 -- the null direction, correctly left alone -- and
        # all six of its residuals reach 1e-13. Its yaw is trimmed by the
        # aileron through Cnda, which is why the system is solvable at all
        # despite the missing column.
        #
        # It does NOT trade a loud failure for a silent one, which was the thing
        # to check before adopting it. Stripping the 747 of elevator authority
        # as well leaves qdot at 9.3e-5 -- eight orders above a converged solve
        # -- in the residual this function already returns. A caller that checks
        # the residual sees an unsolvable request; a NaN it would have had to
        # catch separately, and could not, since JAX cannot raise under jit.
        #
        # rcond is left at the default: measured 1.090e-12 on the 747 at rcond
        # 1e-6, 1e-10, 1e-14 and None alike, so the residual here is set by
        # Newton convergence and not by singular-value truncation.
        return x - jnp.linalg.lstsq(jacobian, r)[0], None

    x, _ = jax.lax.scan(step, x0, None, length=iterations)
    return x, residual(x, airspeed, altitude, ac, anchor, earth_model, heading)


# PROJECT.md section 7 puts the linear-aero ceiling at |alpha| ~ 10-12 deg and says
# a run outside it "is not evidence of anything". 15 deg is a little beyond the
# amber band, so a legitimate trim is never rejected -- every aircraft in the
# registry trims at 3-6 deg at its own cruise condition, which test_trim.py
# asserts as the positive control.
ALPHA_LIMIT = math.radians(15.0)

# A steady-flight trim on a rotating Earth banks by a fraction of a degree --
# 0.148 deg for the 747 at 47N, and `2 Omega V sin(lat) / g` puts the worst case
# at the pole, still under 0.21 deg for anything in this registry. Five degrees is
# far beyond that and far below a turn, so it separates "trimmed" from "solved
# into a banked turn" without rejecting any legitimate solution.
BANK_LIMIT = math.radians(5.0)


def is_physical(x: Array, ac: Aircraft) -> bool:
    """Is a trim solution a flight condition, as opposed to merely converged?

    `CL = CL0 + CLa*alpha` is linear, so a large alpha compensates a small CLa
    and `trim` reaches roots that satisfy the residual to machine precision at
    hundreds of degrees of incidence. A residual check detects non-convergence;
    it cannot detect nonsense, and the two are different questions.

    ALL SIX UNKNOWNS ARE CHECKED, not just alpha. The three that arrived with the
    rotating Earth carry their own stops -- an aileron or rudder past the
    hardware limit is no more a flight condition than a throttle of 567 was --
    and `BANK_LIMIT` separates a trimmed fraction of a degree from a solution
    that has wandered into a banked turn. Until the remediation pass
    this read the angle of attack alone, and a plain (V, h) sweep at the shipped
    initial guess produced solutions it endorsed while they demanded throttle
    outside [0, 1] or elevator past the stops: 319/640 for the 747, 22/640 for
    the 747 approach, 284/640 for the Cherokee, 364/640 for the Cessna. The
    sharpest was the 747 at V = 471.8 m/s, endorsed at alpha = -0.57 deg on a
    throttle of 567. A trim requiring 567 times full thrust is not a flight
    condition under any reading of the word, and the limits that say so were
    already in the `Aircraft` pytree -- which is why this now takes one.

    THE RESIDUAL IS DELIBERATELY NOT AN ARGUMENT. Convergence is the caller's
    other question and every caller that sweeps already asks it separately
    (`validation.sweep`, `verification.newton_residual_history`). Folding it in
    here would make one function answer two questions and would bind this
    module's notion of "converged" to whatever tolerance the caller had in mind.

    NOT folded into `trim` itself, deliberately. `trim` is jitted and vmapped
    (see `minimum_drag_speed`), so it cannot raise, and returning a flag would
    churn every call site for a case that has never arisen with real aircraft
    data. This is a separate question, asked by the callers that sweep.
    """
    alpha, elevator, throttle, phi, aileron, rudder = (float(v) for v in x)
    return bool(
        abs(alpha) <= ALPHA_LIMIT
        and abs(elevator) <= float(ac.elevator_limit)
        and 0.0 <= throttle <= 1.0
        and abs(phi) <= BANK_LIMIT
        and abs(aileron) <= float(ac.aileron_limit)
        and abs(rudder) <= float(ac.rudder_limit)
    )
