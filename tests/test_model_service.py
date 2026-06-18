import pytest
from unittest.mock import patch, MagicMock
from api.services.model_service import ModelService

def test_model_service_not_loaded_initially():
    svc = ModelService()
    assert svc.is_loaded is False

def test_predict_raises_when_not_loaded():
    svc = ModelService()
    with pytest.raises(RuntimeError, match="not loaded"):
        svc.predict("Hello")

def test_predict_returns_expected_keys():
    from api.services.model_service import model_service
    result = model_service.predict("This is a test.")
    assert "label" in result
    assert "confidence" in result
    assert "positive_score" in result
    assert "negative_score" in result
    assert "processing_time_ms" in result

def test_predict_label_values():
    from api.services.model_service import model_service
    result = model_service.predict("I loved this.")
    assert result["label"] in ("POSITIVE", "NEGATIVE")