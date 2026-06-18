"""
NuanceIQ — Standalone Evaluation Script
Run after training to regenerate evaluation artifacts from saved model.
Run: python model/evaluate.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc, classification_report
)
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer, TrainingArguments
)
import torch

ARTIFACTS_DIR = Path("model/artifacts/distilbert-imdb")
MAX_LENGTH = 512
BATCH_SIZE = 16

print("Loading saved model and tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(str(ARTIFACTS_DIR))
model = DistilBertForSequenceClassification.from_pretrained(str(ARTIFACTS_DIR))

print("Loading IMDB test set...")
dataset = load_dataset("imdb")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

encoded = dataset["test"].map(tokenize, batched=True, batch_size=256)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

args = TrainingArguments(
    output_dir="model/artifacts/eval_tmp",
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none"
)
trainer = Trainer(model=model, args=args)

print("Running predictions...")
output = trainer.predict(encoded)
logits = output.predictions
labels = output.label_ids
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
preds = np.argmax(logits, axis=-1)

acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))

metrics = {
    "accuracy": round(acc, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4)
}
with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

cm = confusion_matrix(labels, preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("NuanceIQ — Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "confusion_matrix.png"), dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(labels, probs[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#7B2FBE", lw=2,
         label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("NuanceIQ — ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(str(ARTIFACTS_DIR / "roc_curve.png"), dpi=150)
plt.close()

print("Evaluation complete. Artifacts saved.")