"""
analysis.py
-----------
Evaluation plan for MindClassify.

Primary metric   : Macro F1 (class-imbalance aware)
Secondary metrics: Accuracy, per-class Precision / Recall / F1
Safety focus     : Suicidal Recall (false negatives have the highest real-world cost)

Generates:
  - 7×7 confusion matrices (normalised + raw) per model
  - Per-class F1 / Recall bar charts with Suicidal highlighted
  - 3-model comparison chart (Baseline vs BERT vs MentalBERT)
  - Training history plots (loss + macro F1 curves)
  - Qualitative analysis: 15-20 misclassified suicidal posts saved to text file
"""

import os
import json
import glob
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, precision_recall_fscore_support,
)

from data_preprocessing import LABEL_NAMES, ID2LABEL

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
SUMMARY_CSV = os.path.join(PLOTS_DIR, "model_metrics_summary.csv")
SUMMARY_JSON = os.path.join(PLOTS_DIR, "model_metrics_summary.json")

SUICIDAL_IDX = LABEL_NAMES.index("Suicidal")

_CLASS_COLORS = [
    "#4CAF50", "#2196F3", "#F44336", "#FF9800",
    "#9C27B0", "#00BCD4", "#795548",
]


def print_metrics_report(y_true, y_pred, model_name: str) -> dict:
    acc      = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision, recall, f1_per, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABEL_NAMES))), zero_division=0)
    suicidal_recall  = recall[SUICIDAL_IDX]
    suicidal_support = int(support[SUICIDAL_IDX])
    suicidal_fn      = suicidal_support - int(round(suicidal_recall * suicidal_support))

    print(f"\n{'='*65}")
    print(f"EVALUATION — {model_name}")
    print(f"{'='*65}")
    print(f"  Macro F1   : {f1_macro:.4f}  ← primary metric")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"\n  Suicidal Recall : {suicidal_recall:.4f}  ← key safety metric")
    print(f"  (missed {suicidal_fn} of {suicidal_support} suicidal posts)")
    print(f"\n  Per-class breakdown:")
    print(f"  {'Class':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*67}")
    for i, name in enumerate(LABEL_NAMES):
        marker = "  ◄ safety" if i == SUICIDAL_IDX else ""
        print(f"  {name:<25} {precision[i]:>10.4f} {recall[i]:>10.4f}"
              f" {f1_per[i]:>10.4f} {int(support[i]):>10}{marker}")
    print()
    return {
        "accuracy": acc, "f1_macro": f1_macro, "suicidal_recall": suicidal_recall,
        "per_class_precision": precision, "per_class_recall": recall,
        "per_class_f1": f1_per, "support": support,
    }


def plot_confusion_matrix(y_true, y_pred, model_name: str, normalise: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred)
    if normalise:
        cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, title = ".2f", f"Confusion Matrix (row-normalised) — {model_name}"
    else:
        cm_plot, fmt, title = cm, "d", f"Confusion Matrix — {model_name}"

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                linewidths=0.4, ax=ax)
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.add_patch(plt.Rectangle(
        (0, SUICIDAL_IDX), len(LABEL_NAMES), 1,
        fill=False, edgecolor="#F44336", lw=2.5, clip_on=False))
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    safe = model_name.lower().replace(" ", "_").replace("/", "_")
    norm_tag = "norm" if normalise else "raw"
    path = os.path.join(PLOTS_DIR, f"cm_{safe}_{norm_tag}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_per_class_f1(y_true, y_pred, model_name: str) -> None:
    _, _, f1_per, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(LABEL_NAMES))), zero_division=0)
    colors = ["#F44336" if i == SUICIDAL_IDX else _CLASS_COLORS[i]
              for i in range(len(LABEL_NAMES))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(LABEL_NAMES, f1_per, color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title(f"Per-class F1 — {model_name}")
    for bar, val in zip(bars, f1_per):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9)
    red_patch = mpatches.Patch(color="#F44336", label="Suicidal (safety focus)")
    ax.legend(handles=[red_patch], fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    safe = model_name.lower().replace(" ", "_").replace("/", "_")
    path = os.path.join(PLOTS_DIR, f"f1_{safe}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_suicidal_recall_comparison(results: dict) -> None:
    names   = list(results.keys())
    recalls = [results[n].get("suicidal_recall", 0.0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = ["#F44336" if r < 0.7 else "#4CAF50" for r in recalls]
    bars    = ax.bar(names, recalls, color=colors)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.7, color="grey", linestyle="--", linewidth=1, label="0.70 target")
    ax.set_ylabel("Recall")
    ax.set_title("Suicidal Recall Comparison (key safety metric)")
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, recalls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "suicidal_recall_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_model_comparison(results: dict) -> None:
    names = list(results.keys())
    accs  = [results[n]["accuracy"] for n in names]
    f1s   = [results[n]["f1_macro"] for n in names]
    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_f1  = ax.bar(x - width/2, f1s,  width, label="Macro F1 (primary)", color="#DD8452")
    bars_acc = ax.bar(x + width/2, accs, width, label="Accuracy",            color="#4C72B0")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — MindClassify\n(Baseline → BERT → MentalBERT)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar in list(bars_f1) + list(bars_acc):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                f"{h:.3f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def print_comparison_table(results: dict) -> None:
    print("\n" + "=" * 72)
    print("FINAL MODEL COMPARISON  (sorted by Macro F1)")
    print("=" * 72)
    print(f"{'Model':<32} {'Macro F1':>10} {'Accuracy':>10} {'Suicidal Rec':>14}")
    print("-" * 72)
    for name, m in sorted(results.items(), key=lambda x: -x[1].get("f1_macro", 0)):
        sr = m.get("suicidal_recall", float("nan"))
        sr_str = f"{sr:.4f}" if not np.isnan(sr) else "  N/A"
        print(f"{name:<32} {m['f1_macro']:>10.4f} {m['accuracy']:>10.4f} {sr_str:>14}")


def export_comparison_summary(results: dict, csv_path: str = SUMMARY_CSV,
                              json_path: str = SUMMARY_JSON) -> pd.DataFrame:
    rows = []
    for name, metrics in sorted(results.items(), key=lambda x: -x[1].get("f1_macro", 0)):
        rows.append({
            "model": name,
            "accuracy": metrics.get("accuracy"),
            "f1_macro": metrics.get("f1_macro"),
            "suicidal_recall": metrics.get("suicidal_recall"),
        })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(csv_path, index=False)
    summary_df.to_json(json_path, orient="records", indent=2)
    print(f"Saved summary CSV → {csv_path}")
    print(f"Saved summary JSON → {json_path}")
    return summary_df


def plot_training_history(history_path: str) -> None:
    with open(history_path) as f:
        history = json.load(f)
    epochs     = [h["epoch"]      for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss   = [h["val_loss"]   for h in history]
    train_f1   = [h.get("train_f1", h.get("train_acc", 0)) for h in history]
    val_f1     = [h["val_f1"]     for h in history]
    model_name = os.path.basename(history_path).replace("_history.json", "")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, train_loss, "o-",  label="Train Loss", color="#4C72B0")
    axes[0].plot(epochs, val_loss,   "s--", label="Val Loss",   color="#DD8452")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"Training Loss — {model_name}"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_f1, "o-",  label="Train Macro F1", color="#4C72B0")
    axes[1].plot(epochs, val_f1,   "s--", label="Val Macro F1",   color="#DD8452")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1")
    axes[1].set_title(f"Macro F1 — {model_name}"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"history_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def qualitative_suicidal_analysis(test_df, y_true, y_pred,
                                   n_samples: int = 20, save_path: str = None):
    y_true = list(y_true)
    y_pred = list(y_pred)
    df = test_df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    fn_mask = (df["y_true"] == SUICIDAL_IDX) & (df["y_pred"] != SUICIDAL_IDX)
    false_negatives = df[fn_mask].copy()
    false_negatives["pred_label"] = [ID2LABEL[p] for p in false_negatives["y_pred"]]

    fp_mask = (df["y_pred"] == SUICIDAL_IDX) & (df["y_true"] != SUICIDAL_IDX)
    false_positives = df[fp_mask].copy()
    false_positives["true_label"] = [ID2LABEL[t] for t in false_positives["y_true"]]

    total_suicidal  = int((df["y_true"] == SUICIDAL_IDX).sum())
    fn_count = len(false_negatives)
    fp_count = len(false_positives)
    suicidal_recall = (total_suicidal - fn_count) / max(1, total_suicidal)

    sample_fn = false_negatives.head(n_samples)
    sample_fp = false_positives.head(n_samples // 2)

    lines = []
    lines.append("=" * 70)
    lines.append("QUALITATIVE ANALYSIS — SUICIDAL CLASS MISCLASSIFICATIONS")
    lines.append("=" * 70)
    lines.append(f"Total Suicidal posts in test set : {total_suicidal}")
    lines.append(f"False Negatives (missed)         : {fn_count}  "
                 f"({100*fn_count/max(1,total_suicidal):.1f}%)")
    lines.append(f"False Positives (over-flagged)   : {fp_count}")
    lines.append(f"Suicidal Recall                  : {suicidal_recall:.4f}")
    lines.append("")
    lines.append("Focus: linguistic patterns the model misses —")
    lines.append("  indirect expressions, understatement, metaphor, coded language.")

    lines.append(f"\n{'─'*70}")
    lines.append(f"FALSE NEGATIVES — missed suicidal intent ({len(sample_fn)} samples)")
    lines.append(f"{'─'*70}")
    for i, row in enumerate(sample_fn.itertuples(), 1):
        raw_text = str(getattr(row, "raw_text_original", getattr(row, "text", "")))
        lines.append(f"\n[{i:02d}] Misclassified as: {row.pred_label}")
        lines.append("     Text:")
        for chunk in textwrap.wrap(raw_text, width=70):
            lines.append(f"       {chunk}")

    if len(sample_fp) > 0:
        lines.append(f"\n{'─'*70}")
        lines.append(f"FALSE POSITIVES — incorrectly flagged as suicidal ({len(sample_fp)} samples)")
        lines.append(f"{'─'*70}")
        for i, row in enumerate(sample_fp.itertuples(), 1):
            raw_text = str(getattr(row, "raw_text_original", getattr(row, "text", "")))
            lines.append(f"\n[{i:02d}] True label: {row.true_label}")
            lines.append("     Text:")
            for chunk in textwrap.wrap(raw_text, width=70):
                lines.append(f"       {chunk}")

    lines.append(f"\n{'='*70}")
    lines.append("END OF QUALITATIVE ANALYSIS")
    lines.append(f"{'='*70}")

    output = "\n".join(lines)
    print(output)

    if save_path is None:
        save_path = os.path.join(PLOTS_DIR, "suicidal_misclassifications.txt")
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output)
    print(f"\nSaved qualitative analysis → {save_path}")
    return false_negatives, false_positives


def run_full_analysis(baseline_results: dict, transformer_results: dict) -> None:
    all_results = {**baseline_results, **transformer_results}

    for name, res in all_results.items():
        if "preds" not in res or "true" not in res:
            continue
        print_metrics_report(res["true"], res["preds"], name)
        plot_confusion_matrix(res["true"], res["preds"], name, normalise=True)
        plot_confusion_matrix(res["true"], res["preds"], name, normalise=False)
        plot_per_class_f1(res["true"], res["preds"], name)
        if "test_df" in res:
            qualitative_suicidal_analysis(
                res["test_df"], res["true"], res["preds"],
                save_path=os.path.join(
                    PLOTS_DIR,
                    f"suicidal_misclassifications_{name.lower().replace(' ', '_')}.txt"))

    for hist_file in glob.glob("saved_models/*_history.json"):
        plot_training_history(hist_file)

    summary = {
        n: {"accuracy": r["accuracy"], "f1_macro": r["f1_macro"],
            "suicidal_recall": r.get("suicidal_recall", float("nan"))}
        for n, r in all_results.items()
    }
    plot_model_comparison(summary)
    plot_suicidal_recall_comparison(summary)
    print_comparison_table(summary)
    export_comparison_summary(summary)


if __name__ == "__main__":
    for hist_file in glob.glob("saved_models/*_history.json"):
        plot_training_history(hist_file)
    print("Analysis complete. Check the 'plots/' directory.")
