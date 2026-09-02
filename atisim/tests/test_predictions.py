"""The seal on `atisim.predictions`, and the rules that make it worth having.

These tests do not check whether a prediction is RIGHT -- none of them can be
settled from anything this project holds, which is the point of sealing them.
They check that the register behaves like a register: claims are falsifiable,
digests match the text they were computed from, and settling an entry cannot
quietly rewrite what was predicted.
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


def test_the_two_open_predictions_are_the_ones_the_project_says_matter():
    """A tripwire on scope, not on content.

    Both sealed claims are blocked on a source PROJECT.md section 5 already
    names as unobtainable here. If a future session seals something checkable
    from what is already held, that is not a prediction -- it is a run someone
    has not done yet -- and this test is where that gets noticed.
    """
    assert len(PREDICTIONS) == 2
    assert set(predictions.BY_NAME) == {
        "dc10_does_not_close_the_hannibal_gap",
        "mil_f_8785c_sigma_w_exceeds_the_mehta_ceiling",
    }
    for p in PREDICTIONS:
        assert p.sealed_at == "0c72200"
        assert p.status == "SEALED"


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
