"""
NuanceIQ — DistilBERT Fine-Tuning Script
Trains on IMDB dataset, saves model + tokenizer + evaluation artifacts.
Run: python model/train.py

NOTE: Currently set to 2000 train / 500 test samples for fast local training.
For full 92%+ accuracy, use Google Colab with full dataset (comment out the .select lines).
"""

import os
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
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
import torch

# ── Config ────────────────────────────────────────────────────────────────────
ARTIFACTS_DIR = Path("model/artifacts/distilbert-imdb")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 256          # Reduced from 512 — faster on CPU, minimal accuracy loss
BATCH_SIZE = 8            # Reduced from 16 — safer on low RAM
EPOCHS = 3
LEARNING_RATE = 2e-5
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load Dataset ──────────────────────────────────────────────────────────────
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

# ── Subset for local CPU training (comment out for full training on Colab) ────
dataset["train"] = dataset["train"].select(range(2000))
dataset["test"] = dataset["test"].select(range(500))
print(f"Using {len(dataset['train'])} train / {len(dataset['test'])} test samples")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

print("Tokenizing dataset...")
encoded = dataset.map(tokenize, batched=True, batch_size=256)
encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = encoded["train"]
test_dataset = encoded["test"]

# ── Model ─────────────────────────────────────────────────────────────────────
print("Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary"
    )
    acc = accuracy_score(labels, predictions)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

# ── Training Args ─────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=str(ARTIFACTS_DIR / "checkpoints"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    warmup_steps=100,         # Reduced from 500 — suits smaller dataset
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=str(ARTIFACTS_DIR / "logs"),
    logging_steps=50,         # More frequent logs on small dataset
    seed=SEED,
    report_to="none"
)

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print("Starting training... (estimated 10–15 min on CPU)")
trainer.train()

# ── Save Model + Tokenizer ────────────────────────────────────────────────────
print("Saving model and tokenizer...")
trainer.save_model(str(ARTIFACTS_DIR))
tokenizer.save_pretrained(str(ARTIFACTS_DIR))
print(f"Model saved to {ARTIFACTS_DIR}")

# ── Full Evaluation ───────────────────────────────────────────────────────────
print("Running full evaluation...")
predictions_output = trainer.predict(test_dataset)
logits = predictions_output.predictions
labels = predictions_output.label_ids
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
preds = np.argmax(logits, axis=-1)

acc = accuracy_score(labels, preds)
precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
report = classification_report(labels, preds, target_names=["Negative", "Positive"])

print(f"\nAccuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"\nClassification Report:\n{report}")

# ── Save Metrics JSON ─────────────────────────────────────────────────────────
metrics = {
    "accuracy": round(acc, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1": round(f1, 4),
    "model": MODEL_NAME,
    "dataset": "imdb",
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH
}
with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Saved metrics.json")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
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
print("Saved confusion_matrix.png")

# ── ROC Curve ─────────────────────────────────────────────────────────────────
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
print("Saved roc_curve.png")

print("\nTraining complete. All artifacts saved to model/artifacts/distilbert-imdb/")