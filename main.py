"""
main.py
-------
Flask REST API for MindClassify inference.

Endpoints:
  GET  /health              — health check
  POST /predict             — classify a single text
  POST /batch_predict       — classify a list of texts

Loads the best available model:
  1. Transformer checkpoint (saved_models/<model_name>_best/)
  2. Baseline sklearn pipeline  (saved_models/best_baseline.pkl)
  3. Demo mode — returns mock scores when no model is found

Environment variables:
  MODEL_TYPE   : "transformer" | "baseline" | "auto" (default: auto)
  MODEL_PATH   : override checkpoint/pkl path
  PORT         : server port (default: 5001)
"""

import os
import json
import logging
from importlib import metadata as importlib_metadata
from typing import List, Dict, Any

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

from data_preprocessing import clean_text, LABEL_NAMES, ID2LABEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Global model state ────────────────────────────────────────────────────────
_model = None
_tokenizer = None
_model_type = None   # "transformer" | "baseline" | "demo"
_device = None
_model_source = None
_runtime_mode = None

APP_VERSION = "1.0.0"


def _load_transformer(path: str):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    global _model, _tokenizer, _device, _model_source, _runtime_mode

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading transformer from {path} on {_device}")
    _tokenizer = AutoTokenizer.from_pretrained(path)
    _model = AutoModelForSequenceClassification.from_pretrained(path)
    _model.eval()
    _model.to(_device)
    _model_source = path
    _runtime_mode = "trained"
    logger.info("Transformer loaded.")


def _load_baseline(path: str):
    import joblib
    global _model, _model_source, _runtime_mode
    logger.info(f"Loading baseline pipeline from {path}")
    _model = joblib.load(path)
    _model_source = path
    _runtime_mode = "trained"
    logger.info("Baseline loaded.")


def _safe_package_version(package_name: str) -> str:
    try:
        return importlib_metadata.version(package_name)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def _get_system_info() -> Dict[str, Any]:
    return {
        "app_version": APP_VERSION,
        "runtime_mode": _runtime_mode or "unknown",
        "model_type": _model_type,
        "model_source": _model_source or "demo-heuristics",
        "python": os.sys.version.split()[0],
        "package_versions": {
            "flask": _safe_package_version("flask"),
            "gradio": _safe_package_version("gradio"),
            "transformers": _safe_package_version("transformers"),
            "torch": _safe_package_version("torch"),
            "scikit-learn": _safe_package_version("scikit-learn"),
        },
        "supported_labels": LABEL_NAMES,
    }


def _init_model():
    global _model_type, _runtime_mode, _model_source

    model_type_env = os.getenv("MODEL_TYPE", "auto").lower()
    model_path_env = os.getenv("MODEL_PATH", "")

    # ── Explicit path provided ────────────────────────────────────────────────
    if model_path_env:
        if model_path_env.endswith(".pkl"):
            _load_baseline(model_path_env)
            _model_type = "baseline"
        else:
            _load_transformer(model_path_env)
            _model_type = "transformer"
        return

    # ── Auto-detect ───────────────────────────────────────────────────────────
    save_dir = "saved_models"
    if model_type_env in ("transformer", "auto"):
        candidates = [
            os.path.join(save_dir, d)
            for d in os.listdir(save_dir)
            if d.endswith("_best") and os.path.isdir(os.path.join(save_dir, d))
        ] if os.path.isdir(save_dir) else []

        if candidates:
            # pick the most recently modified
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            _load_transformer(candidates[0])
            _model_type = "transformer"
            return

    if model_type_env in ("baseline", "auto"):
        pkl = os.path.join(save_dir, "best_baseline.pkl")
        if os.path.exists(pkl):
            _load_baseline(pkl)
            _model_type = "baseline"
            return

    logger.warning("No trained model found — running in DEMO mode with mock predictions.")
    _model_type = "demo"
    _runtime_mode = "demo"
    _model_source = "keyword-heuristics"


def _predict_transformer(texts: List[str]) -> List[Dict[str, Any]]:
    import torch
    cleaned = [clean_text(t) for t in texts]
    enc = _tokenizer(
        cleaned,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(_device) for k, v in enc.items()}
    with torch.no_grad():
        logits = _model(**enc).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    results = []
    for prob_row in probs:
        pred_id = int(np.argmax(prob_row))
        results.append({
            "label": ID2LABEL[pred_id],
            "label_id": pred_id,
            "confidence": float(prob_row[pred_id]),
            "probabilities": {LABEL_NAMES[i]: float(prob_row[i]) for i in range(len(LABEL_NAMES))},
        })
    return results


def _predict_baseline(texts: List[str]) -> List[Dict[str, Any]]:
    cleaned = [clean_text(t) for t in texts]
    preds = _model.predict(cleaned)

    # Try to get probabilities if the final estimator supports it
    try:
        probs_matrix = _model.predict_proba(cleaned)
    except AttributeError:
        # LinearSVC doesn't have predict_proba; use decision function as proxy
        try:
            df = _model.decision_function(cleaned)
            # Softmax approximation
            exp_df = np.exp(df - df.max(axis=1, keepdims=True))
            probs_matrix = exp_df / exp_df.sum(axis=1, keepdims=True)
        except Exception:
            probs_matrix = None

    results = []
    for i, pred_id in enumerate(preds):
        if probs_matrix is not None:
            row = probs_matrix[i]
            probs = {LABEL_NAMES[j]: float(row[j]) for j in range(len(LABEL_NAMES))}
            confidence = float(row[pred_id])
        else:
            probs = {label: (1.0 if j == pred_id else 0.0) for j, label in enumerate(LABEL_NAMES)}
            confidence = 1.0

        results.append({
            "label": ID2LABEL[pred_id],
            "label_id": int(pred_id),
            "confidence": confidence,
            "probabilities": probs,
        })
    return results


def _predict_demo(texts: List[str]) -> List[Dict[str, Any]]:
    """Mock predictor for demo mode — uses keyword heuristics."""
    keyword_map = {
        "depress": "Depression", "sad": "Depression", "hopeless": "Depression",
        "suicid": "Suicidal", "kill myself": "Suicidal", "end my life": "Suicidal",
        "anxious": "Anxiety", "anxiety": "Anxiety", "panic": "Anxiety", "worry": "Anxiety",
        "stress": "Stress", "overwhelm": "Stress", "pressure": "Stress",
        "bipolar": "Bipolar", "manic": "Bipolar", "mania": "Bipolar",
        "personality": "Personality Disorder", "borderline": "Personality Disorder",
    }
    results = []
    rng = np.random.default_rng(seed=42)

    for text in texts:
        lower = text.lower()
        matched = "Normal"
        for kw, label in keyword_map.items():
            if kw in lower:
                matched = label
                break

        # Build plausible probability distribution
        base = rng.dirichlet(np.ones(len(LABEL_NAMES)) * 0.5)
        pred_id = LABEL_NAMES.index(matched)
        base[pred_id] = max(base[pred_id], 0.55)
        base /= base.sum()

        results.append({
            "label": matched,
            "label_id": pred_id,
            "confidence": float(base[pred_id]),
            "probabilities": {LABEL_NAMES[i]: float(base[i]) for i in range(len(LABEL_NAMES))},
            "demo_mode": True,
        })
    return results


def predict(texts: List[str]) -> List[Dict[str, Any]]:
    if _model_type == "transformer":
        return _predict_transformer(texts)
    if _model_type == "baseline":
        return _predict_baseline(texts)
    return _predict_demo(texts)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_type": _model_type,
        "runtime_mode": _runtime_mode,
        "model_source": _model_source,
        "labels": LABEL_NAMES,
    })


@app.route("/system_info", methods=["GET"])
def system_info():
    return jsonify(_get_system_info())


@app.route("/predict", methods=["POST"])
def predict_single():
    data = request.get_json(force=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400

    results = predict([text])
    return jsonify(results[0])


@app.route("/batch_predict", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field (list)"}), 400

    texts = data["texts"]
    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "'texts' must be a non-empty list"}), 400
    if len(texts) > 100:
        return jsonify({"error": "Maximum 100 texts per batch"}), 400

    results = predict([str(t) for t in texts])
    return jsonify({"results": results, "count": len(results)})


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _init_model()
    port = int(os.getenv("PORT", 5001))
    logger.info(f"Starting MindClassify Flask API on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
