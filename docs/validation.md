# Validation

## What AtiSim can be used for

> **AtiSim is validated as a comparative and mechanistic tool for the longitudinal gust response
> of a rigid transport aircraft at cruise.** Within the envelope below, it reproduces published
> response orderings, the mechanisms behind them, and the linear modes of its own source data.
> **It is not a validated absolute-load predictor.**

The model can answer questions such as "which encounter is worse?", "how does the response scale
with airspeed?", "what happens to the load if the aircraft is slower?" and "does the ordering
hold?". It cannot answer "what will the load be?".

### The envelope

Outside any one of these limits, the claim does not carry over.

| | Validated range | Why |
|---|---|---|
| Aircraft | 747-class transports: `boeing747` and `boeing747_jsbsim` | the only entries with a measured recovery band |
| Mach | 0.70–0.90 | `aircraft.valid_mach` |
| Altitude | 35,000–45,000 ft (`boeing747`); 35,000–41,000 ft (`boeing747_jsbsim`) | `aircraft.valid_altitude`. For a statement about both, use the narrower range |
| Angle of attack | \|α\| < 10°, with 10–12° as a caution band | the lift model has no stall |
| Gust length scale | more than about three wingspans | gusts are sampled at a point (assumption E2) |
| Axis | longitudinal only | no recorded lateral turbulence response is available to validate against |

## The evidence

The evidence is built in four steps, each resting on the one below. The executed notebook
`notebooks/validation-ladder.ipynb` re-runs all four.

### 1. The numerical core

These checks compare the model against exact mathematics rather than published data:

- The RK4 integrator shows an observed order of accuracy of **3.99982** on a problem with a
  closed-form solution, and **3.989** through the full 6-DOF dynamics.
- Angular momentum drifts by **5.7 × 10⁻¹³** over 600 s, and the quaternion norm stays within
  1.1 × 10⁻¹⁶ of unity.
- Torque-free rotation matches its analytic trajectory, not only its invariants.
- A uniform wind only translates the trajectory, and a uniform wind that varies in time adds no
  body force. These results rule out the two classic gust-modelling errors.

`notebooks/solver-validation.ipynb` covers this step.

### 2. The source data, and independent implementations

The table compares the Boeing 747's linear modes with NASA CR-2144 Table IX-5, at flight
condition 9 (Mach 0.8, 40,000 ft):

| | Model error |
|---|---|
| Short period, natural frequency | −1.32% |
| Short period, damping | +0.57% |
| Phugoid, natural frequency | +1.68% |
| Phugoid, damping | +1.38% |

Three independent implementations of the same equations back up the chain from derivatives
to behaviour:

- **Caughey's worked example** of the CR-2144 power-approach case. The model matches every
  plant-matrix element it carries.
- **JSBSim**, fed the same coefficients. The two engines reach the same trim and modes, with
  the 737's phugoid frequency within 0.45% of JSBSim's own linearisation. The two engines are
  also compared through vortex encounters.
- **The simulation code of Yoshimura et al.** The model reproduces its eigenvalues to 0.01% on
  the short period and 0.20% on the phugoid.

The 747's speed derivatives are a hand digitisation of CR-2144 pp. 220–222. An independent
automated trace of the same pages, scored against Table IX-4, supports that reading.

### 3. A recorded encounter: Hannibal, Missouri, 3 April 1981

A DC-10 at 37,000 ft flew through a row of clear-air-turbulence vortices. NASA identified the
wind field from its flight recorder, and Mehta (1987) fitted five vortices to it. NASA TM-102186
records the normal load the aircraft pulled.

Flown through Mehta's field along the path on which the field was identified, the model's 747
reaches **70.2% of the recorded peak-to-peak load** (1.90 g against 2.70 g).

The model is not expected to match the record exactly. The recorded aircraft was a DC-10-10 at
an unrecorded weight, and no DC-10 derivative set is published. The shortfall is a result, not
something to tune away. Adding sourced derivatives to the 747 has moved this figure **away**
from the record twice while improving agreement with the model's own source. The model does not
choose derivatives by their effect on this figure.

### 4. Published orderings and mechanisms

- **TM-102186 Fig. 8.** Three aircraft of very different speed fly the same reconstructed
  vortices. The model reproduces the load ordering the paper reports, including its
  counter-intuitive half, and agrees with the paper's stated mechanism in six cases out of six.
- **Wingrove & Bach's discriminator.** A vortex, an updraft and a pilot's manoeuvre can be told
  apart by how much the aircraft pitches while pulling the load. The model's pitch excursions
  fall in the published order.
- **Frequency response.** The load-factor spectrum of the Hannibal run peaks within one
  frequency bin of the aircraft's short period, and more than a bin and a half from the rate at
  which it crosses the vortex cores. Over a Dryden ensemble, the response carries more than three
  times as much energy at the short period as at 0.05 Hz, even though the input carries more
  energy at 0.05 Hz. So the airframe selects which frequencies it responds to.

### Predictions made before the run

`atisim.predictions` holds claims that were committed before the run that decided them. Three
have been settled: two were right, and one was wrong on one panel of four. The wrong one is
recorded as wrong. A fourth claim, that a DC-10 derivative set would not close the Hannibal
gap, is still open.

## Limitations

- **Absolute loads.** These are not validated, for the reasons given under step 3.
- **Lateral response.** The capability exists, but no available source records a lateral
  turbulence response to compare against.
- **Past the linear range.** The lift model has no stall, so it will fly to large angles of
  attack and report plausible nonsense. Check `panel.ALPHA_LINEAR_DEG` and the `alpha_band` run
  check.
- **A rigid airframe.** The CR-2144 derivatives describe a flexible airframe, but the model
  integrates a rigid body (assumption B1). Do not use it to make claims about structural loads.
- **Derivatives held away from trim.** Except for the 747's speed derivatives, stability
  derivatives are constant across the envelope (assumption C3). Quote the excursion with every
  result.
- **Summary numbers only.** Every comparison against a real encounter uses summary numbers:
  peaks, spacings and orderings. None compares a time history against a time history.

## Known issues

- When the strip load path is enabled (`vortex_viz.fly(strip=True)`), it counts the gust's
  rolling moment twice. This affects lateral results only, and no longitudinal result uses this
  path.
- The Earth is flat and non-rotating (assumption A1). At the 747's cruise condition, Coriolis
  acceleration is about 0.0035 g.

{doc}`model` lists every modelling assumption, with a measured bound for each where one exists.
