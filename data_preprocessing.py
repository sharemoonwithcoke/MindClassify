"""
data_preprocessing.py
---------------------
Handles data loading, cleaning, and splitting for MindClassify.
Expected CSV columns: 'statement' (text) and 'status' (label string).
"""

import re
import string
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Download required NLTK resources (once)
for resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

# ── Label mapping ─────────────────────────────────────────────────────────────
LABEL_NAMES = [
    "Normal",
    "Depression",
    "Suicidal",
    "Anxiety",
    "Stress",
    "Bipolar",
    "Personality Disorder",
]

LABEL2ID = {label: idx for idx, label in enumerate(LABEL_NAMES)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}

# Aliases in raw data → canonical label
LABEL_ALIASES = {
    "normal": "Normal",
    "depression": "Depression",
    "suicidal": "Suicidal",
    "anxiety": "Anxiety",
    "stress": "Stress",
    "bipolar": "Bipolar",
    "bipolar disorder": "Bipolar",
    "personality disorder": "Personality Disorder",
    "personality": "Personality Disorder",
}


# ── Cleaning ──────────────────────────────────────────────────────────────────

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Keep negations — they carry sentiment
_NEGATIONS = {"no", "not", "nor", "never", "none", "nobody", "nothing", "neither"}
_EFFECTIVE_STOPWORDS = _stop_words - _NEGATIONS


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Lowercase → remove URLs/mentions/hashtags → strip punctuation/digits →
    remove stopwords (keep negations) → optionally lemmatize.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                     # mentions
    text = re.sub(r"#(\w+)", r"\1", text)                 # hashtags → word
    text = re.sub(r"\d+", " ", text)                      # digits
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    tokens = [t for t in tokens if t not in _EFFECTIVE_STOPWORDS]
    if lemmatize:
        tokens = [_lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


# ── Loading ───────────────────────────────────────────────────────────────────

def load_dataset(path: str, text_col: str = "statement", label_col: str = "status") -> pd.DataFrame:
    """Load CSV and return a cleaned DataFrame with columns ['text', 'label', 'label_id']."""
    df = pd.read_csv(path)

    # Flexible column names
    col_map = {}
    for col in df.columns:
        if col.lower() in ("statement", "text", "post", "content"):
            col_map["text"] = col
        if col.lower() in ("status", "label", "class", "category"):
            col_map["label"] = col

    if "text" not in col_map:
        col_map["text"] = text_col
    if "label" not in col_map:
        col_map["label"] = label_col

    df = df.rename(columns={col_map["text"]: "raw_text", col_map["label"]: "raw_label"})
    df = df.dropna(subset=["raw_text", "raw_label"])

    # Normalise labels
    df["label"] = df["raw_label"].str.strip().map(
        lambda x: LABEL_ALIASES.get(x.lower(), x)
    )
    df = df[df["label"].isin(LABEL_NAMES)].reset_index(drop=True)

    df["text"] = df["raw_text"].apply(clean_text)
    df["raw_text_original"] = df["raw_text"]
    df["label_id"] = df["label"].map(LABEL2ID)

    print(f"Loaded {len(df)} samples")
    print("Class distribution:")
    for label, count in sorted(Counter(df["label"]).items(), key=lambda x: -x[1]):
        print(f"  {label:<25} {count:>5}  ({100*count/len(df):.1f}%)")

    return df[["text", "raw_text_original", "label", "label_id"]]


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    """Stratified train / val / test split."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_id"], random_state=seed
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label_id"], random_state=seed
    )
    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


# ── HuggingFace Dataset helper ────────────────────────────────────────────────

def to_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128):
    """Convert a DataFrame split to a HuggingFace Dataset with tokenized inputs."""
    from datasets import Dataset

    hf = Dataset.from_pandas(df[["text", "label_id"]].rename(columns={"label_id": "labels"}))

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    hf = hf.map(tokenize, batched=True)
    hf.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return hf


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/mental_health.csv"
    df = load_dataset(path)
    train, val, test = split_dataset(df)
    print("\nSample:")
    print(train.head(3).to_string())
