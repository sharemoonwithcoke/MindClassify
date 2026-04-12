"""
transformer_trainer.py
-----------------------
Step 2 — Main Model (BERT fine-tuning)
Step 3 — Advanced Comparison (MentalBERT fine-tuning)

Fine-tunes BERT-base-uncased or mental/mental-bert-base-uncased for 7-class
mental health classification.

Key design decisions:
  - Input truncated to 512 tokens (full BERT limit)
  - AdamW optimiser, lr=2e-5, 3 epochs
  - Weighted cross-entropy loss to address class imbalance
  - Early stopping on Macro F1 (not accuracy) as the primary metric
  - Macro F1 reported as primary; Suicidal Recall highlighted separately

Usage:
  # Step 2 — BERT
  python transformer_trainer.py --model_name bert-base-uncased \\
      --data_path data/mental_health.csv

  # Step 3 — MentalBERT
  python transformer_trainer.py --model_name mental/mental-bert-base-uncased \\
      --data_path data/mental_health.csv

  # Quick test (<20 min on CPU)
  python transformer_trainer.py --fast --data_path data/mental_health.csv
"""

import argparse
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    precision_recall_fscore_support,
)
from tqdm import tqdm

from data_preprocessing import (
    load_dataset, split_dataset, LABEL_NAMES, ID2LABEL, LABEL2ID, MAX_LENGTH,
)

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_LABELS   = len(LABEL_NAMES)
SUICIDAL_IDX = LABEL_NAMES.index("Suicidal")


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _version_info() -> dict:
    try:
        import importlib.metadata as importlib_metadata
        return {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": importlib_metadata.version("transformers"),
            "scikit-learn": importlib_metadata.version("scikit-learn"),
            "numpy": np.__version__,
        }
    except Exception:
        return {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": "unknown",
            "scikit-learn": "unknown",
            "numpy": np.__version__,
        }


class MentalHealthDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def train_epoch(model, loader, optimizer, scheduler, device, loss_fn):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for batch in tqdm(loader, desc="  train", leave=False):
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss   = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1


@torch.no_grad()
def eval_epoch(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for batch in tqdm(loader, desc="  eval ", leave=False):
        labels = batch.pop("labels").to(device)
        batch  = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch).logits
        loss   = loss_fn(logits, labels)
        total_loss += loss.item()
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(np.argmax(probs, axis=-1))
        all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader), acc, f1, all_preds, all_labels, all_probs


def fine_tune(
    model_name: str, data_path: str,
    epochs: int = 3, batch_size: int = 16, lr: float = 2e-5,
    max_length: int = None, warmup_ratio: float = 0.1,
    patience: int = 3, use_weighted_loss: bool = True,
    seed: int = 42, max_train_samples: int = None, max_val_samples: int = None,
):
    if max_length is None:
        max_length = MAX_LENGTH

    _set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Epochs : {epochs}  |  batch_size : {batch_size}  "
          f"|  max_length : {max_length}  |  lr : {lr}")

    # ── Data (light cleaning for transformer) ─────────────────────────────────
    df = load_dataset(data_path, cleaning="transformer")
    train_df, val_df, test_df = split_dataset(df)

    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(max_train_samples, random_state=seed).reset_index(drop=True)
        print(f"  [fast mode] Training on {len(train_df)} samples")
    if max_val_samples and max_val_samples < len(val_df):
        val_df = val_df.sample(max_val_samples, random_state=seed).reset_index(drop=True)
        print(f"  [fast mode] Validating on {len(val_df)} samples")

    # ── Weighted loss ─────────────────────────────────────────────────────────
    train_labels = train_df["label_id"].tolist()
    if use_weighted_loss:
        raw_weights = compute_class_weight("balanced", classes=np.arange(NUM_LABELS), y=train_labels)
        weight_tensor = torch.FloatTensor(raw_weights).to(device)
        print("Class weights (balanced):")
        for name, w in zip(LABEL_NAMES, raw_weights):
            print(f"  {name:<25} {w:.4f}")
    else:
        weight_tensor = None
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    # ── Tokenize ──────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def encode(texts):
        return tokenizer(texts, truncation=True, padding="max_length",
                         max_length=max_length, return_tensors=None)

    train_enc = encode(train_df["text"].tolist())
    val_enc   = encode(val_df["text"].tolist())
    test_enc  = encode(test_df["text"].tolist())

    train_ds = MentalHealthDataset(train_enc, train_labels)
    val_ds   = MentalHealthDataset(val_enc,   val_df["label_id"].tolist())
    test_ds  = MentalHealthDataset(test_enc,  test_df["label_id"].tolist())

    _pin         = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=_pin)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size*2, num_workers=0, pin_memory=_pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size*2, num_workers=0, pin_memory=_pin)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)
    model.to(device)

    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = len(train_loader) * epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_f1      = 0.0
    patience_counter = 0
    history          = []
    safe_name        = model_name.replace("/", "_")
    ckpt_path        = os.path.join(SAVE_DIR, f"{safe_name}_best")

    print(f"\nTraining up to {epochs} epoch(s) "
          f"(early stopping on Macro F1, patience={patience})...")

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn)
        val_loss, val_acc, val_f1, _, _, _ = eval_epoch(
            model, val_loader, device, loss_fn)

        print(f"  train  loss={train_loss:.4f}  acc={train_acc:.4f}  macro_f1={train_f1:.4f}")
        print(f"  val    loss={val_loss:.4f}  acc={val_acc:.4f}  macro_f1={val_f1:.4f}  ← primary")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        if val_f1 > best_val_f1:
            best_val_f1      = val_f1
            patience_counter = 0
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            training_config = {
                "max_length": max_length,
                "model_name": model_name,
                "seed": seed,
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr,
                "warmup_ratio": warmup_ratio,
                "patience": patience,
                "use_weighted_loss": use_weighted_loss,
                "max_train_samples": max_train_samples,
                "max_val_samples": max_val_samples,
                "versions": _version_info(),
            }
            with open(os.path.join(ckpt_path, "training_config.json"), "w") as f:
                json.dump(training_config, f, indent=2)
            print(f"  ** Saved best (val Macro F1={val_f1:.4f}) → {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping after epoch {epoch}.")
                break

    # ── Test ──────────────────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint from {ckpt_path} ...")
    best_model = AutoModelForSequenceClassification.from_pretrained(ckpt_path).to(device)
    _, test_acc, test_f1, test_preds, test_labels, _ = eval_epoch(
        best_model, test_loader, device, loss_fn)

    print(f"\nTest Macro F1 : {test_f1:.4f}  ← primary metric")
    print(f"Test Accuracy : {test_acc:.4f}")

    _, recall_per, _, support_per = precision_recall_fscore_support(
        test_labels, test_preds, labels=list(range(NUM_LABELS)), zero_division=0)
    suicidal_recall  = recall_per[SUICIDAL_IDX]
    suicidal_support = int(support_per[SUICIDAL_IDX])
    suicidal_fn      = suicidal_support - int(round(suicidal_recall * suicidal_support))
    print(f"\nSuicidal Recall : {suicidal_recall:.4f}  ← key safety metric")
    print(f"  (missed {suicidal_fn} of {suicidal_support} suicidal posts)")

    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds,
                                target_names=LABEL_NAMES, zero_division=0))

    hist_path = os.path.join(SAVE_DIR, f"{safe_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved → {hist_path}")

    run_metadata = {
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_length": max_length,
        "patience": patience,
        "use_weighted_loss": use_weighted_loss,
        "max_train_samples": max_train_samples,
        "max_val_samples": max_val_samples,
        "test_metrics": {
            "accuracy": test_acc,
            "f1_macro": test_f1,
            "suicidal_recall": suicidal_recall,
        },
        "versions": _version_info(),
    }
    meta_path = os.path.join(SAVE_DIR, f"{safe_name}_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)
    print(f"Metadata saved → {meta_path}")

    test_results = {
        "test_df": test_df, "test_preds": test_preds, "test_labels": test_labels,
        "test_acc": test_acc, "test_f1": test_f1, "suicidal_recall": suicidal_recall,
    }
    return best_model, tokenizer, history, test_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune BERT / MentalBERT")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--data_path", default="data/mental_health.csv")
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_length", type=int,   default=None)
    parser.add_argument("--patience",   type=int,   default=3)
    parser.add_argument("--no_weighted_loss", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast", action="store_true",
                        help="Quick-test: 2 epochs, seq_len=64, batch=32, 2k train/500 val")
    args = parser.parse_args()

    if args.fast:
        if args.epochs     == 3:   args.epochs     = 2
        if args.max_length is None: args.max_length = 64
        if args.batch_size == 16:  args.batch_size = 32
        if args.patience   == 3:   args.patience   = 1
        if args.max_train_samples is None: args.max_train_samples = 2_000
        if args.max_val_samples   is None: args.max_val_samples   = 500
        print("=" * 60)
        print(f"FAST MODE — epochs={args.epochs}  max_length={args.max_length}  "
              f"batch={args.batch_size}  patience={args.patience}")
        print("=" * 60)

    best_model, tokenizer, history, test_results = fine_tune(
        model_name=args.model_name, data_path=args.data_path,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        max_length=args.max_length, patience=args.patience,
        seed=args.seed,
        use_weighted_loss=not args.no_weighted_loss,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )

    from analysis import (
        plot_confusion_matrix, plot_per_class_f1,
        plot_training_history, qualitative_suicidal_analysis,
    )
    safe = args.model_name.replace("/", "_")
    model_label = "MentalBERT" if "mental" in args.model_name.lower() else "BERT-base-uncased"

    plot_confusion_matrix(test_results["test_labels"], test_results["test_preds"], model_label)
    plot_per_class_f1(test_results["test_labels"], test_results["test_preds"], model_label)
    plot_training_history(os.path.join("saved_models", f"{safe}_history.json"))
    qualitative_suicidal_analysis(
        test_results["test_df"], test_results["test_labels"], test_results["test_preds"])
