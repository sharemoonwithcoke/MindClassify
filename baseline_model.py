"""
baseline_model.py
-----------------
Step 1 — Baseline Model (Traditional Machine Learning)

TF-IDF + Logistic Regression.

Objective: establish a performance floor for the 7-class mental health
classification task and verify feasibility without requiring a GPU.
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import cross_val_score

from data_preprocessing import load_dataset, split_dataset, LABEL_NAMES

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

SUICIDAL_IDX = LABEL_NAMES.index("Suicidal")


def build_pipeline(fast: bool = False) -> Pipeline:
    """TF-IDF (unigram + bigram) + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=20_000 if fast else 50_000,
            sublinear_tf=True,
            min_df=2,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",   # compensates for label imbalance
            n_jobs=-1,
            random_state=42,
        )),
    ])


def evaluate(pipeline, X_test, y_test) -> dict:
    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1_macro = f1_score(y_test, preds, average="macro", zero_division=0)
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_test, preds,
        labels=list(range(len(LABEL_NAMES))),
        zero_division=0,
    )
    report = classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "report": report,
        "preds": list(preds),
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1_per,
        "support": support,
    }


def train_baseline(
    data_path: str,
    cv_folds: int = 5,
    max_train_samples: int = None,
    fast: bool = False,
):
    df = load_dataset(data_path)
    train, val, test = split_dataset(df)

    # Combine train + val for final model; test is held out
    train_full = pd.concat([train, val], ignore_index=True)

    if max_train_samples and max_train_samples < len(train_full):
        train_full = (
            train_full.sample(max_train_samples, random_state=42)
            .reset_index(drop=True)
        )
        print(f"  [fast mode] Training on {len(train_full)} samples")

    X_train = train_full["text"].tolist()
    y_train = train_full["label_id"].tolist()
    X_test  = test["text"].tolist()
    y_test  = test["label_id"].tolist()

    pipeline = build_pipeline(fast=fast)

    print(f"\n{'='*60}")
    print("Step 1 — Baseline: TF-IDF + Logistic Regression")
    print(f"{'='*60}")

    # Cross-validation (primary metric: Macro F1)
    cv_scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv_folds, scoring="f1_macro", n_jobs=-1,
    )
    print(f"CV Macro F1 ({cv_folds}-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Final fit on full train set, evaluate on held-out test
    pipeline.fit(X_train, y_train)
    metrics = evaluate(pipeline, X_test, y_test)
    metrics["cv_mean"] = float(cv_scores.mean())
    metrics["cv_std"]  = float(cv_scores.std())

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\nTest Macro F1  : {metrics['f1_macro']:.4f}  ← primary metric")
    print(f"Test Accuracy  : {metrics['accuracy']:.4f}")

    suicidal_recall = metrics["per_class_recall"][SUICIDAL_IDX]
    suicidal_support = int(metrics["support"][SUICIDAL_IDX])
    suicidal_fn = suicidal_support - int(round(suicidal_recall * suicidal_support))
    print(f"\nSuicidal Recall: {suicidal_recall:.4f}  ← key safety metric")
    print(f"  (missed {suicidal_fn} of {suicidal_support} suicidal posts)")

    print(f"\nPer-class results:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*65}")
    for i, name in enumerate(LABEL_NAMES):
        marker = "  ◄" if i == SUICIDAL_IDX else ""
        print(
            f"  {name:<25}"
            f" {metrics['per_class_precision'][i]:>10.4f}"
            f" {metrics['per_class_recall'][i]:>10.4f}"
            f" {metrics['per_class_f1'][i]:>10.4f}"
            f" {int(metrics['support'][i]):>10}"
            f"{marker}"
        )

    # Save
    save_path = os.path.join(SAVE_DIR, "baseline_lr.pkl")
    joblib.dump(pipeline, save_path)
    print(f"\nSaved → {save_path}")

    return pipeline, metrics, test


def load_baseline(path: str = None):
    path = path or os.path.join(SAVE_DIR, "baseline_lr.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved baseline at {path}. Run training first.")
    return joblib.load(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 1 baseline: TF-IDF + Logistic Regression"
    )
    parser.add_argument("--data_path", default="data/mental_health.csv")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Cap training set size for quick tests")
    parser.add_argument("--fast", action="store_true",
                        help="Quick-test mode: 3-fold CV, 20 000 TF-IDF features, "
                             "3 000 training samples. Target <5 min on CPU.")
    args = parser.parse_args()

    if args.fast:
        if args.cv_folds == 5:          args.cv_folds = 3
        if args.max_train_samples is None: args.max_train_samples = 3_000
        print("=" * 60)
        print("FAST MODE — reduced settings for testing")
        print(f"  cv_folds={args.cv_folds}, "
              f"max_train_samples={args.max_train_samples}, "
              "max_features=20 000")
        print("=" * 60)

    pipeline, metrics, test_df = train_baseline(
        args.data_path, args.cv_folds,
        max_train_samples=args.max_train_samples,
        fast=args.fast,
    )

    from analysis import plot_confusion_matrix, plot_per_class_f1, qualitative_suicidal_analysis
    y_true = test_df["label_id"].tolist()
    y_pred = metrics["preds"]
    plot_confusion_matrix(y_true, y_pred, "TF-IDF + Logistic Regression")
    plot_per_class_f1(y_true, y_pred, "TF-IDF + Logistic Regression")
    qualitative_suicidal_analysis(test_df, y_true, y_pred)
