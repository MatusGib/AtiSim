"""The seal on `atisim.predictions`, and the rules that make it worth having.

These tests do not check whether a prediction is RIGHT. They check that the
register behaves like a register: claims are falsifiable, digests match the text
they were computed from, and settling an entry cannot quietly rewrite what was
predicted.

*** THIS FILE USED TO SAY "none of them can be settled from anything this
project holds, which is the point of sealing them". Session 25 sealed one that
CAN be, deliberately, and the last test in this file records what changed about
the rule and what did not. ***
"""

import re

import pytest

from atisim import predictions
from atisim.predictions import PREDICTIONS, Prediction, digest_of


def test_every_sealed_claim_still_hashes_to_its_recorded_digest():
    """The seal itself.

    The digests in predictions.py are LITERALS, so editing a claim without
    editing its digest fails here. Editing both passes -- and shows up as a diff
    on a line whose only purpose is to be stable, next to the SHA that dates the
    claim. That is what this buys: tampering made visible, not impossible, and
    predictions.py's own docstring says so rather than overselling it.
    """
    for p in PREDICTIONS:
        assert p.digest == digest_of(p), (
            f"{p.name}: the sealed claim has changed since it was written. If "
            f"this is a deliberate re-seal, say so in the commit message -- the "
            f"rules in predictions.py say a SEALED entry is never edited."
        )


def test_a_computed_digest_would_have_made_this_check_vacuous():
    """The negative control on the test above, and on a real drafting mistake.

    The first draft of predictions.py computed each digest at import from the
    same fields it hashes, which cannot ever fail: change the claim and the
    digest follows it. This asserts the digest is genuinely a function of the
    text, by mutating the text and watching it stop matching.
    """
    p = PREDICTIONS[0]
    tampered = p._replace(claim=p.claim + " (and something else)")
    assert digest_of(tampered) != p.digest


def test_no_prediction_can_be_unfalsifiable():
    """Rule 3: a prediction that cannot be wrong is not a prediction.

    Checked as more than a non-empty string -- the falsifier has to name an
    observation, so it must carry a number or a comparison. A restatement of
    the claim in the negative would pass a mere length check and would tell a
    later reader nothing about what to go and measure.
    """
    for p in PREDICTIONS:
        assert p.falsified_if.strip(), p.name
        assert len(p.falsified_if) > 60, p.name
        assert re.search(r"\d", p.falsified_if), (
            f"{p.name}: the falsifier names no quantity"
        )
        # And it must say where the answer comes from, or nobody can settle it.
        assert len(p.settled_by) > 20, p.name


def test_every_claim_carries_numbers_and_a_dated_seal():
    """A claim without an interval is an opinion; a seal without a SHA is not
    dated. Both are cheap to get wrong and expensive to notice later."""
    for p in PREDICTIONS:
        assert re.search(r"\d", p.claim), p.name
        assert re.fullmatch(r"[0-9a-f]{7,40}", p.sealed_at), p.name
        assert re.fullmatch(r"[0-9a-f]{16}", p.digest), p.name
        assert p.status in ("SEALED", "SETTLED"), p.name


def test_names_are_unique_and_the_index_agrees_with_the_tuple():
    names = [p.name for p in PREDICTIONS]
    assert len(names) == len(set(names))
    assert set(predictions.BY_NAME) == set(names)


def test_an_unsettled_prediction_carries_no_outcome():
    """Rule 4 in mechanical form: a SEALED entry is a bet, not evidence.

    An outcome on a SEALED row would read as a result to anyone skimming, which
    is exactly the confusion the register exists to prevent.
    """
    for p in PREDICTIONS:
        if p.status == "SEALED":
            assert p.outcome == "", p.name
        else:
            assert p.outcome, p.name


def test_settling_cannot_rewrite_what_was_predicted():
    """Rule 2, demonstrated rather than asserted.

    `digest_of` deliberately excludes `status` and `outcome`, so the legitimate
    act of settling leaves the seal intact -- while any edit to the claim in the
    same breath breaks it. This is what stops a settled entry from being quietly
    reshaped to match its own answer.
    """
    p = PREDICTIONS[0]
    settled = p._replace(status="SETTLED", outcome="it did the other thing")
    assert digest_of(settled) == p.digest

    reshaped = settled._replace(claim="something the answer happens to match")
    assert digest_of(reshaped) != p.digest


SOURCE_GATED = {
    # Settled only by a document PROJECT.md section 5 names as not held here.
    "dc10_does_not_close_the_hannibal_gap",
}
RUN_GATED = {
    # Settled by a run in this repository. Admissible ONLY because the seal
    # commit precedes the run -- see the test below for why that is the rule.
    "the_dryden_response_peaks_at_the_short_period": "scripts/cat_spectra.py",
    # MOVED from SOURCE_GATED in session 27, and the move is a fact about the
    # shelf rather than a convenience. This entry was sealed (0c72200) when
    # MIL-F-8785C was not held; session 25 fetched it to `refs/MIL-F-8785C.pdf`,
    # which makes it settleable by a run here exactly as the Dryden entry is.
    # The seal still precedes the run by many commits, so the rule the class
    # exists to protect is untouched. Note the script is named in `outcome`
    # rather than `settled_by`: `settled_by` is inside the digest and a SEALED
    # entry's claim fields are never edited, so the test below looks in both.
    "mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling": (
        "scripts/digitise_mil_f_8785c_fig7.py"
    ),
}


def test_every_open_prediction_is_declared_and_classified():
    """A tripwire on scope, not on content -- and it fired once, as designed.

    ITS ORIGINAL FORM asserted that both sealed claims were blocked on a source
    section 5 names as unobtainable, on the reasoning that a claim checkable
    from what is already held "is not a prediction -- it is a run someone has
    not done yet, and this test is where that gets noticed". Session 25 added
    exactly that kind of entry and this test went red. That was the notice.

    WHAT CHANGED AND WHAT DID NOT. The objection is right about the failure it
    fears: a claim that could have been checked in five minutes, sealed and
    settled in one breath, manufactures a cheap win and is worth nothing. But
    an unobtainable source was never what defeated that -- what defeats it is
    the SEAL PRECEDING THE RUN, which is a fact about the git history and is
    the only thing predictions.py has ever claimed to rest on. An unobtainable
    source merely makes the ordering obvious for free.

    So the rule is now two admissible classes, and adding an entry means
    classifying it here rather than adding one to a count. A RUN_GATED entry
    additionally has to name a script that exists, because "settled by a run"
    with no run named is the same cheap win wearing a different hat.
    """
    from pathlib import Path

    assert set(predictions.BY_NAME) == SOURCE_GATED | set(RUN_GATED)
    root = Path(__file__).resolve().parents[2]
    for p in PREDICTIONS:
        if p.name in RUN_GATED:
            # A run-gated entry may already be settled -- the run is in this
            # repository, so settling it needs nobody to send a document.
            assert p.status in ("SEALED", "SETTLED"), p.name
            script = RUN_GATED[p.name]
            # The run must be NAMED somewhere in the entry, which is the whole
            # point of the check -- "settled by a run" with no run named is the
            # cheap win this class exists to forbid. It may be named in
            # `outcome` rather than `settled_by` for an entry sealed before the
            # script existed, because `settled_by` is inside the digest and a
            # sealed claim is never edited (predictions.py rule 1).
            assert script in p.settled_by or script in p.outcome, p.name
            assert (root / script).is_file(), (
                f"{p.name} is settled by {script}, which does not exist"
            )
        else:
            # A source-gated entry cannot be settled here by definition, so a
            # SETTLED one means a document arrived and section 3 should say so.
            assert p.status == "SEALED", p.name
            assert p.sealed_at == "0c72200", p.name


def test_the_dc10_claim_contradicts_the_projects_own_current_story():
    """Why this entry is worth having: it bets AGAINST the running conclusion.

    PROJECT.md section 5 now names aircraft type as the surviving explanation
    for the 32% load shortfall and a DC-10 derivative set as the highest-value
    acquisition. This prediction says that acquisition will NOT close the gap,
    on the strength of section 4's measured mass null. One of the two is wrong,
    and sealing it is how that gets settled rather than argued.
    """
    p = predictions.BY_NAME["dc10_does_not_close_the_hannibal_gap"]
    assert "will NOT reach" in p.claim
    assert "2.70" in p.claim          # the recorded value it says is out of reach
    assert "0.6%" in p.reasoning      # the measured null it rests on
    with pytest.raises(AssertionError):
        # Guard the guard: if the claim ever stops being the uncomfortable one,
        # this test should fail rather than keep vouching for it.
        assert "will reach 2.70" in p.claim
