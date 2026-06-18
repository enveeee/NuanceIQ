import pytest
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

async def test_predict_returns_200(client: AsyncClient):
    response = await client.post("/predict", json={"text": "This movie was fantastic!"})
    assert response.status_code == 200

async def test_predict_response_structure(client: AsyncClient):
    response = await client.post("/predict", json={"text": "Great film."})
    data = response.json()
    assert "label" in data
    assert "confidence" in data
    assert "positive_score" in data
    assert "negative_score" in data
    assert "cached" in data

async def test_predict_label_is_valid(client: AsyncClient):
    response = await client.post("/predict", json={"text": "Absolutely loved it."})
    data = response.json()
    assert data["label"] in ("POSITIVE", "NEGATIVE")

async def test_predict_confidence_range(client: AsyncClient):
    response = await client.post("/predict", json={"text": "Not bad."})
    data = response.json()
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["positive_score"] <= 1.0
    assert 0.0 <= data["negative_score"] <= 1.0

async def test_predict_scores_sum_to_one(client: AsyncClient):
    response = await client.post("/predict", json={"text": "Decent enough."})
    data = response.json()
    total = round(data["positive_score"] + data["negative_score"], 3)
    assert abs(total - 1.0) < 0.01

async def test_predict_empty_text_returns_422(client: AsyncClient):
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422

async def test_predict_whitespace_only_returns_422(client: AsyncClient):
    response = await client.post("/predict", json={"text": "     "})
    assert response.status_code == 422

async def test_predict_missing_text_field_returns_422(client: AsyncClient):
    response = await client.post("/predict", json={})
    assert response.status_code == 422

async def test_predict_very_long_text(client: AsyncClient):
    long_text = "This film was great. " * 100
    response = await client.post("/predict", json={"text": long_text})
    assert response.status_code == 200

async def test_predict_special_characters(client: AsyncClient):
    response = await client.post(
        "/predict",
        json={"text": "Wów! Ämäzing… 🎬 #cinema @director"}
    )
    assert response.status_code == 200