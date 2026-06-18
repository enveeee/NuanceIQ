import pytest
from pydantic import ValidationError
from api.schemas.sentiment import PredictRequest, PredictResponse, SentimentLabel

def test_predict_request_valid():
    req = PredictRequest(text="Hello world")
    assert req.text == "Hello world"

def test_predict_request_strips_whitespace():
    req = PredictRequest(text="  Hello  ")
    assert req.text == "Hello"

def test_predict_request_empty_raises():
    with pytest.raises(ValidationError):
        PredictRequest(text="")

def test_predict_request_whitespace_raises():
    with pytest.raises(ValidationError):
        PredictRequest(text="   ")

def test_predict_response_valid():
    resp = PredictResponse(
        text="Great!",
        label=SentimentLabel.POSITIVE,
        confidence=0.95,
        positive_score=0.95,
        negative_score=0.05,
        cached=False
    )
    assert resp.label == SentimentLabel.POSITIVE

def test_sentiment_label_enum():
    assert SentimentLabel.POSITIVE == "POSITIVE"
    assert SentimentLabel.NEGATIVE == "NEGATIVE"