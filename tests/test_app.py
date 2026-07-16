import pytest

import main


@pytest.fixture
def client():
    main.app.config["TESTING"] = True
    return main.app.test_client()


def test_healthz(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.get_json() == {"status": "ok"}


def test_query_ok_strips_suffix(client, monkeypatch):
    # Mock out the Kubernetes + OpenAI calls so the endpoint is testable offline.
    monkeypatch.setattr(main, "gather_kubernetes_data", lambda: {"pods": []})
    monkeypatch.setattr(main, "query_llm", lambda data, q: "frontend-6b5f4cf68c-6g5lt")

    resp = client.post("/query", json={"query": "name a running pod"})
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["query"] == "name a running pod"
    assert body["answer"] == "frontend"  # generated suffix stripped


def test_query_blank_returns_400(client):
    resp = client.post("/query", json={"query": "   "})
    assert resp.status_code == 400


def test_query_missing_field_returns_400(client):
    resp = client.post("/query", json={})
    assert resp.status_code == 400
