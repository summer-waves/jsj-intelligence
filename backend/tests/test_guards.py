from app.safety import guards
from app.safety.profiles import DEMO, PRIVATE


def test_citation_numbers_and_invalid():
    used, invalid = guards.check_citations("A fact [1]. Another [2, 9].", 5)
    assert used == [1, 2, 9]
    assert invalid == [9]


def test_no_citation_becomes_refusal():
    r = guards.apply_guards("Yoga is relaxing.", 5, DEMO.claim_pattern)
    assert r.refused and r.withheld == guards.WITHHELD_NO_CITATIONS
    assert r.answer == guards.REFUSAL_TEXT


def test_invalid_citation_becomes_refusal():
    r = guards.apply_guards("Yoga is relaxing [7].", 5, DEMO.claim_pattern)
    assert r.refused


def test_cure_claim_is_withheld():
    r = guards.apply_guards("JSJ cures diabetes [1].", 5, PRIVATE.claim_pattern)
    assert r.withheld == guards.WITHHELD_CLAIM
    assert r.answer == guards.UNSAFE_TEXT


def test_negated_claim_is_allowed():
    r = guards.apply_guards("The library does not show it treats anything [1].", 5, PRIVATE.claim_pattern)
    assert r.withheld is None and r.used == [1]


def test_model_refusal_passes_through():
    r = guards.apply_guards("I could not find this in the JSJ library.", 5, DEMO.claim_pattern)
    assert r.refused and r.withheld is None


def test_uncited_sentence_detected_and_citation_after_stop():
    used, invalid, uncited, refused = guards.analyze("Good fact. [1] Bare sentence.", 5)
    assert used == [1] and uncited == ["Bare sentence."]


def test_question_flags():
    assert guards.question_flags("I have chest pain") == (True, True)
    assert guards.question_flags("What is yoga?") == (False, False)
