import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app  # noqa: E402


def test_health_and_ask(make_service):
    svc = make_service("According to the NCCIH pages in the library, yoga is generally safe [1].")
    with TestClient(create_app(svc)) as client:
        h = client.get("/health").json()
        assert h["status"] == "ok" and h["library_mode"] == "demo"
        r = client.post("/ask", json={"question": "Is yoga safe?"})
        assert r.status_code == 200 and r.json()["citations_used"] == [1]


def test_short_question_rejected(make_service):
    with TestClient(create_app(make_service("x"))) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
