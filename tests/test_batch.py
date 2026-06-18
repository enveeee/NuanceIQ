import pytest
import io
import csv
from httpx import AsyncClient

pytestmark = pytest.mark.asyncio

def make_csv(rows: list[str]) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text"])
    for row in rows:
        writer.writerow([row])
    return output.getvalue().encode("utf-8")

async def test_batch_valid_csv(client: AsyncClient):
    csv_data = make_csv(["Great movie!", "Terrible film.", "Not bad."])
    response = await client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 3
    assert len(data["results"]) == 3

async def test_batch_response_structure(client: AsyncClient):
    csv_data = make_csv(["Amazing!"])
    response = await client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    result = response.json()["results"][0]
    assert "label" in result
    assert "confidence" in result
    assert "positive_score" in result
    assert "negative_score" in result

async def test_batch_rejects_non_csv(client: AsyncClient):
    response = await client.post(
        "/predict/batch",
        files={"file": ("test.txt", b"some text", "text/plain")}
    )
    assert response.status_code == 400

async def test_batch_rejects_csv_without_text_column(client: AsyncClient):
    csv_data = b"review,score\nGreat,5\nBad,1"
    response = await client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 400

async def test_batch_empty_csv(client: AsyncClient):
    csv_data = make_csv([])
    response = await client.post(
        "/predict/batch",
        files={"file": ("test.csv", csv_data, "text/csv")}
    )
    assert response.status_code == 200
    assert response.json()["total"] == 0