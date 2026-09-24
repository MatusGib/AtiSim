"""Sealed predictions: what the model says BEFORE the answer is available.

A comparison made with the source open beside the model is retrodictive. A
model earns the word "predictive" by stating a result before anyone can check
it, and then being checked. Each entry here is such a statement, with the
observation that would falsify it.

The seal is the git history. Each entry records the commit its author could see
(`sealed_at`) and a hash of its own claim (`digest`). `test_predictions.py`
recomputes the digest, so an accidental edit fails the build; a deliberate one
shows up as a diff on a line whose only purpose is to be stable. The repository
history was rewritten once to remove third-party documents, so `sealed_at`
values name commits from before that rewrite.

The rules:

  1. A SEALED entry is never edited -- not to fix a number, widen an interval,
     or add a caveat.
  2. Settling one means adding `outcome` and `settled_by`, and changing
     `status` -- nothing else. A wrong prediction stays wrong, with its
     original numbers.
  3. A prediction that cannot be wrong is not a prediction. Every entry names
     the observation that would falsify it.
  4. Nothing here may be quoted as evidence FOR the model. Until an entry is
     settled it is a bet; after it is settled it is one data point.
"""

import hashlib
from typing import NamedTuple


class Prediction(NamedTuple):
    name: str
    claim: str          # what is predicted, in numbers
    falsified_if: str   # the observation that would kill it
    reasoning: str      # why the model says this
    settled_by: str     # the source or run that will decide
    sealed_at: str      # commit the author could see when sealing
    digest: str         # sha256 of the four fields above it, first 16 hex
    status: str = "SEALED"
    outcome: str = ""   # filled in ONLY when settling


def digest_of(p: Prediction) -> str:
    """Hash of the claim as sealed. Deliberately excludes status and outcome,
    which are the only two fields settling is allowed to touch."""
    payload = "\x1f".join(
        (p.name, p.claim, p.falsified_if, p.reasoning, p.settled_by, p.sealed_at)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Sealed session 23d, against tree 0c72200.
# ---------------------------------------------------------------------------

PREDICTIONS: tuple[Prediction, ...] = (
    Prediction(
        name="dc10_does_not_close_the_hannibal_gap",
        claim=(
            "Flown on wind.mehta_hannibal_array at 37,000 ft with fixed "
            "controls, a DC-10 built from a published derivative set will "
            "reach a peak-to-peak n_z within +/-15% of the 747's 1.8385 g -- "
            "that is, between 1.563 and 2.114 g -- and will NOT reach the "
            "2.70 g the DFDR recorded."
        ),
        falsified_if=(
            "the DC-10 run lands outside [1.563, 2.114] g, and emphatically if "
            "it reaches 2.70 g, which would mean aircraft TYPE was the missing "
            "32% all along and this project's section 5 was right for the "
            "wrong reason."
        ),
        reasoning=(
            "Section 4 measured the relevant null directly: quadrupling airframe "
            "mass moved the load answer by 0.6%, because dn = dCL/CL_trim and "
            "CL_trim = W/qS, so wing loading cancels. If that null is general, "
            "swapping a 747 for a DC-10 cannot close a 32% gap, and the "
            "surviving explanation is not aircraft type but the missing "
            "nonlinearity -- buffet, which CR-2144 does not tabulate. This "
            "prediction is therefore UNCOMFORTABLE for the project's current "
            "story, which names a DC-10 set as the highest-value acquisition. "
            "That is precisely why it is worth sealing."
        ),
        settled_by=(
            "any published DC-10-10/-30 longitudinal derivative set at cruise, "
            "flown through scripts/cat_uncertainty.py's harness"
        ),
        sealed_at="0c72200",
        digest="9ba2a78079f06a1c",
    ),
    Prediction(
        name="mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling",
        claim=(
            "When MIL-F-8785C's sigma_w chart is digitised at 37,000 ft, the "
            "severe-turbulence curve (10^-5 exceedance) will give sigma_w "
            "greater than 4.46 m/s -- the ceiling wind.mehta_residual_ceiling "
            "derives from Mehta's own fit residual."
        ),
        falsified_if=(
            "the digitised severe curve gives sigma_w <= 4.46 m/s at that "
            "altitude, which would mean the intensity needed to close the "
            "Hannibal load gap is not available even at the specification's "
            "own severe level, and the Dryden explanation dies outright."
        ),
        reasoning=(
            "Mehta's residual is an RMS over an entire flight record, most of "
            "which is not the encounter, so it should sit well below a value "
            "defined by an exceedance probability at the severe end. If the "
            "spec's severe value did NOT exceed it, the two sources would be "
            "in direct conflict about the same air. Note what this does and "
            "does not buy: it makes the ~4-5 m/s closure PHYSICALLY available "
            "without making it what Hannibal contained -- section 4 keeps "
            "those separate and settling this must not merge them."
        ),
        settled_by="MIL-F-8785C Figure 7, digitised at 37,000 ft",
        sealed_at="0c72200",
        digest="3fb6e8c238194d1d",
        status="SETTLED",
        outcome=(
            "RIGHT, with a margin of 0.34 m/s. Figure 7 digitised from "
            "refs/MIL-F-8785C.pdf p. 49 by scripts/digitise_mil_f_8785c_fig7.py: "
            "at 37,000 ft the severe stroke reads 15.7 ft/s = 4.80 m/s against "
            "the predicted > 4.46. The reading is a BAND, not a point, and the "
            "reason is in the figure rather than in the method: SEVERE and the "
            "10^-5 exceedance curve are drawn as ONE stroke of ink at that "
            "altitude, so the value is 4.80 +/- ~0.12 m/s depending which member "
            "the stroke belongs to. The whole band clears 4.46, so the merge "
            "cannot change the verdict. Cross-checked against JSBSim's "
            "FGWinds.cpp -- an independent transcription of the same figure, "
            "which neither this reading nor its author set -- whose severe row "
            "interpolates to 4.822 m/s at 37 kft, 0.5% away and inside the merge "
            "band. An earlier pass quoted 0.4% by reading the merged stroke as "
            "though it were SEVERE alone; that was too good and is not what is "
            "claimed here. WHAT THIS DOES NOT BUY, per the reasoning's own "
            "reservation: it makes the ~4-5 m/s closure PHYSICALLY AVAILABLE at "
            "the specification's severe level. It is NOT evidence that Hannibal "
            "contained it. Section 4 keeps those separate and this outcome does "
            "not merge them."
        ),
    ),
    # -----------------------------------------------------------------------
    # Sealed session 25 (phase 2), against tree 2837ddd, BEFORE any response
    # spectrum had been computed anywhere in this project. `atisim/response.py`
    # and its unit tests existed; nothing had been flown through them.
    # -----------------------------------------------------------------------
    Prediction(
        name="the_dryden_response_peaks_at_the_short_period",
        claim=(
            "The ensemble-averaged n_z response spectrum of boeing747 flown "
            "through wind.dryden_field at wind.MEHTA_HANNIBAL_ALTITUDE, "
            "searched above response.PHUGOID_FLOOR_HZ, will peak within "
            "+/-20% of the aircraft's own short-period frequency at that "
            "condition -- 0.1640 Hz, from validation.longitudinal_modes -- "
            "that is, between 0.131 and 0.197 Hz."
        ),
        falsified_if=(
            "the ensemble-averaged peak lands outside [0.131, 0.197] Hz, and "
            "emphatically if it lands at the 0.05 Hz search floor, which would "
            "mean the load follows the input's own spectrum and the airframe's "
            "resonance does not organise the response at all."
        ),
        reasoning=(
            "The gust-to-load transfer is band-pass at both ends and the "
            "Dryden input supplies no peak of its own. At zero frequency a "
            "sustained updraft is a new equilibrium -- the aircraft climbs "
            "with the air and dn returns to zero -- so the transfer function "
            "vanishes there; at the short period, zeta = 0.3645 gives a "
            "resonant gain near 1/(2 zeta). The input is flat below "
            "Omega = 1/L_w (0.070 Hz at this speed) and falls as Omega^-2 "
            "above it, which is a factor 2.4 against the short period between "
            "those two frequencies -- against a factor near 8 of resonant "
            "gain in its favour. The product must therefore peak near the "
            "airframe's frequency and not at the input's flat end. This is "
            "the first test in this project of the COUPLING rather than the "
            "amplitude, and PROJECT.md section 5 already records that "
            "Yoshimura 2022's own resonance argument is wrong by a factor of "
            "2*pi -- so the corrected version deserves to be bet on before it "
            "is run."
        ),
        settled_by="scripts/cat_spectra.py, the Dryden ensemble limb",
        sealed_at="2837ddd",
        digest="60843c1f7237db6d",
        status="SETTLED",
        outcome=(
            "RIGHT, at both intensities and not comfortably. 32 flights of "
            "100 s each: the ensemble-averaged peak is 0.1400 Hz at "
            "sigma = 2.108 m/s and 0.1700 Hz at sigma = 4.459, against a "
            "predicted [0.131, 0.197]. The lower one sits 14.7% below the "
            "short period and 0.009 Hz -- less than one bin -- inside the "
            "band's lower edge, which is the falling Dryden input pulling the "
            "peak below the resonance exactly as the reasoning said it would "
            "compete. A tighter band would have been wrong. The robust form of "
            "the result is not the argmax at all: the input has strictly MORE "
            "energy at 0.05 Hz than at the short period and the response has "
            "more than three times LESS, and that reversal is a ratio of two "
            "numbers rather than a peak location. Pinned in "
            "test_cat_spectra.py."
        ),
    ),
    Prediction(
        name="cr2144_two_readings_agree_on_the_good_panels",
        claim=(
            "On the four CR-2144 pp. 220-222 panels the automated trace rates "
            "'good' (CL_alpha, CD_alpha, Cm_alpha, Cm_M), session 30's hand "
            "reading and the automated trace agree to within 2.0% of the "
            "panel's full scale -- the median over the hand-placed points -- "
            "at every altitude, and each hand curve sits closest to the "
            "automated curve of its OWN altitude."
        ),
        falsified_if=(
            "Any of those twelve curves shows a median disagreement above 2.0% "
            "of full scale, or a hand curve sits closer to a different "
            "altitude's automated curve."
        ),
        reasoning=(
            "2% of full scale is the automated trace's own stated accuracy "
            "(Reference_papers/CR-2144/README.md: 'good to about 1-2% of full "
            "scale', the printed line width dominating). The hand reading was "
            "checked against Table IX-4 at eight conditions in session 30. Two "
            "readings each good to that level of the same ink should agree to "
            "it. Sealed before the comparison code existed."
        ),
        settled_by=(
            "scripts/cr2144_digitisation_crosscheck.py, which prints "
            "atisim.cr2144_mach.crosscheck()"
        ),
        sealed_at="1091fc7",
        digest="72262a936af4ab7e",
        status="SETTLED",
        outcome=(
            "WRONG, on one panel of four. Ten of the twelve good-panel curves "
            "agree to 0.22-1.32% of full scale (CL_alpha 0.23/0.29/0.74, "
            "CD_alpha 0.30/0.52/0.55, Cm_alpha 0.22/0.27/0.56, Cm_M SL 1.32), "
            "but Cm_M reads 2.40% at 20K and 2.02% at 40K, biased the same way "
            "at all three altitudes. The altitude half of the claim holds: "
            "every good curve sits on its own altitude. Scored against Table "
            "IX-4 at the circled conditions, the AUTOMATED trace is the one "
            "that is off -- Cm_M RMS 0.0146 against the hand reading's 0.0063 "
            "-- so the hand reading the shipped 747 declares its speed "
            "derivatives from is the better-anchored of the two. Run: "
            "scripts/cr2144_digitisation_crosscheck.py, session 32."
        ),
    ),
)

# The digests above are LITERALS on purpose. An earlier draft computed them at
# import from the same fields, which makes the check vacuous -- edit a claim and
# the digest follows it, and the test can never fail. Written by hand, editing a
# claim breaks the build until the digest is edited too, and that second edit is
# a diff on a line whose only job is to be stable.

BY_NAME = {p.name: p for p in PREDICTIONS}
