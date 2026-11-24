def test_cited_answer_has_sources(make_service):
    svc = make_service("According to the NCCIH pages in the library, yoga is generally safe [1].")
    out = svc.answer("Is yoga safe?")
    assert not out.refused and out.citations_used == [1]
    assert out.sources and out.sources[0].cited
    assert len(out.sources[0].excerpt.split()) <= 41  # 40 words plus the "..." marker


def test_refusal_hides_sources_and_rewords(make_service):
    out = make_service("I could not find this in the JSJ library.").answer("Tell me about yoga")
    assert out.refused and out.sources == []
    assert "demo library" in out.answer


def test_emergency_notice(make_service):
    out = make_service("According to the NCCIH pages in the library, yoga helps [1].").answer(
        "I have chest pain, is yoga ok?")
    assert out.notices.emergency and out.notices.health


def test_uncited_answer_withheld(make_service):
    out = make_service("Yoga is nice.").answer("Is yoga nice?")
    assert out.refused and out.withheld_reason == "no valid citations"
