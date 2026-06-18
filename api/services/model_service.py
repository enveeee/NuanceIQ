import time
import torch
import numpy as np
from pathlib import Path
from loguru import logger
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)
from api.config import get_settings

settings = get_settings()

class ModelService:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._loaded = False

    def load(self):
        model_path = Path(settings.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                f"Run: python model/train.py"
            )
        logger.info(f"Loading model from {model_path} on {self.device}")
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_path))
        self.model = DistilBertForSequenceClassification.from_pretrained(
            str(model_path)
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> dict:
        if not self._loaded:
            raise RuntimeError("Model is not loaded")

        start = time.perf_counter()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=settings.model_max_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
        label_idx = int(np.argmax(probs))
        label = "POSITIVE" if label_idx == 1 else "NEGATIVE"
        confidence = float(probs[label_idx])

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "positive_score": round(float(probs[1]), 4),
            "negative_score": round(float(probs[0]), 4),
            "processing_time_ms": round(elapsed_ms, 2)
        }

model_service = ModelService()