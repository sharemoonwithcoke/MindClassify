"""
demo.py
-------
Gradio interactive demo for MindClassify.

Launches a web UI at http://localhost:7860 where users can type text
and see the predicted mental health category with a confidence breakdown.

Run:
  python demo.py
  python demo.py --model_type transformer --model_path saved_models/bert-base-uncased_best
  python demo.py --share   # generate public Gradio link
"""

import argparse
import os
import json
import numpy as np

CLASS_COLORS = {
    "Normal": "#4CAF50", "Depression": "#2196F3", "Suicidal": "#F44336",
    "Anxiety": "#FF9800", "Stress": "#9C27B0", "Bipolar": "#00BCD4",
    "Personality Disorder": "#795548",
}

LABEL_NAMES = [
    "Normal", "Depression", "Suicidal", "Anxiety",
    "Stress", "Bipolar", "Personality Disorder",
]

SAFE_RESOURCES = {
    "Suicidal": "If you or someone you know is in crisis, please contact the **988 Suicide & Crisis Lifeline** (call or text 988) or visit https://988lifeline.org",
    "Depression": "For support with depression, visit the **NIMH** resource page: https://www.nimh.nih.gov/health/topics/depression",
    "Anxiety": "For anxiety support, visit **ADAA**: https://adaa.org",
    "Stress": "For stress management resources, visit **APA Help Center**: https://www.apa.org/topics/stress",
    "Bipolar": "For bipolar disorder information, visit **DBSA**: https://www.dbsalliance.org",
    "Personality Disorder": "For personality disorder resources, visit **NAMI**: https://www.nami.org/About-Mental-Illness/Mental-Health-Conditions/Borderline-Personality-Disorder",
}

EXAMPLES = [
    ["I've been feeling really down lately and can't find motivation to do anything."],
    ["Today was a great day! I went for a walk and caught up with old friends."],
    ["I can't stop worrying about everything and my heart races all the time."],
    ["I haven't slept in days, my thoughts are racing and I feel invincible."],
    ["The pressure of work and family is just too much to handle anymore."],
    ["I don't see any reason to keep going. Everything feels completely hopeless."],
    ["My emotions swing so wildly — one minute I'm euphoric, the next I'm crashing."],
]

UI_CSS = """
:root {
    --bg: linear-gradient(135deg, #f7f8fc 0%, #eef3ff 52%, #fdf6f0 100%);
    --panel: rgba(255, 255, 255, 0.92);
    --border: rgba(31, 41, 55, 0.10);
    --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
    --primary: #1d4ed8;
    --danger: #dc2626;
    --text: #0f172a;
    --muted: #475569;
}

.gradio-container {
    background: var(--bg) !important;
}

.hero {
    background: linear-gradient(135deg, rgba(29, 78, 216, 0.96), rgba(37, 99, 235, 0.82));
    color: white;
    padding: 28px 30px;
    border-radius: 24px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(255, 255, 255, 0.16);
}

.hero h1 {
    margin: 0;
    font-size: 2.1rem;
    line-height: 1.1;
}

.hero p {
    margin: 10px 0 0 0;
    font-size: 1rem;
    opacity: 0.95;
    color: #ffffff;
}

.hero * {
    color: #ffffff !important;
}

.section-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: var(--shadow);
    padding: 18px;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 14px;
}

.metric {
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(148,163,184,0.22);
    border-radius: 16px;
    padding: 14px 16px;
}

.metric-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
}

.metric-value {
    font-size: 1.2rem;
    font-weight: 700;
    color: var(--text);
}

.badge {
    display: inline-block;
    padding: 0.45rem 0.75rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.88rem;
    margin-bottom: 10px;
}

.badge-high { background: rgba(220, 38, 38, 0.14); color: #991b1b; }
.badge-medium { background: rgba(245, 158, 11, 0.16); color: #92400e; }
.badge-low { background: rgba(34, 197, 94, 0.16); color: #166534; }

.result-title { font-size: 1.1rem; font-weight: 700; margin: 0 0 8px 0; }
.result-subtitle { color: var(--muted); margin-bottom: 10px; }
.small-note { color: var(--muted); font-size: 0.92rem; }
"""


def _read_training_config(path: str) -> dict:
    cfg_path = os.path.join(path, "training_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            return json.load(f)
    return {}


def _load_model(model_type: str, model_path: str | None):
    from data_preprocessing import MAX_LENGTH

    if model_type == "transformer" and model_path:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval().to(device)
        cfg = _read_training_config(model_path)
        max_len = cfg.get("max_length", MAX_LENGTH)
        return ("transformer", model, tokenizer, device, max_len)

    if model_type == "baseline" and model_path:
        import joblib
        return ("baseline", joblib.load(model_path), None, None, None)

    # Auto-detect
    save_dir = "saved_models"
    if os.path.isdir(save_dir):
        candidates = [
            os.path.join(save_dir, d)
            for d in os.listdir(save_dir)
            if d.endswith("_best") and os.path.isdir(os.path.join(save_dir, d))
        ]
        if candidates:
            candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return _load_model("transformer", candidates[0])

        pkl = os.path.join(save_dir, "baseline_lr.pkl")
        if os.path.exists(pkl):
            return _load_model("baseline", pkl)

    # Fallback to demo mode with keyword heuristics
    print("Using demo mode with keyword-based heuristics...")
    return ("demo", None, None, None, None)


def run_prediction(text: str, state: tuple) -> tuple:
    from data_preprocessing import clean_text_baseline, clean_text_transformer, ID2LABEL

    kind = state[0]
    model = state[1]
    tokenizer = state[2]
    device = state[3]
    max_len = state[4]

    if kind == "transformer":
        import torch
        cleaned = clean_text_transformer(text)
        enc = tokenizer(cleaned, truncation=True, padding=True,
                        max_length=max_len, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    elif kind == "baseline":
        cleaned = clean_text_baseline(text)
        pred_id = int(model.predict([cleaned])[0])
        try:
            probs = model.predict_proba([cleaned])[0]
        except Exception:
            try:
                df = model.decision_function([cleaned])[0]
                exp = np.exp(df - df.max())
                probs = exp / exp.sum()
            except Exception:
                probs = np.zeros(len(LABEL_NAMES))
                probs[pred_id] = 1.0
        confidence = float(probs[pred_id])
        prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    else:  # demo
        keyword_map = {
            "depress": "Depression", "sad": "Depression", "hopeless": "Depression",
            "suicid": "Suicidal", "kill myself": "Suicidal", "end my life": "Suicidal",
            "anxious": "Anxiety", "anxiety": "Anxiety", "panic": "Anxiety",
            "stress": "Stress", "overwhelm": "Stress",
            "bipolar": "Bipolar", "manic": "Bipolar",
            "personality": "Personality Disorder", "borderline": "Personality Disorder",
        }
        matched = "Normal"
        for kw, label in keyword_map.items():
            if kw in text.lower():
                matched = label
                break
        pred_id = LABEL_NAMES.index(matched)
        rng = np.random.default_rng(abs(hash(text)) % (2**31))
        base = rng.dirichlet(np.ones(len(LABEL_NAMES)) * 0.5)
        base[pred_id] = max(base[pred_id], 0.55)
        base /= base.sum()
        probs = base
        confidence = float(probs[pred_id])
        prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    return LABEL_NAMES[pred_id], confidence, prob_dict


def _prediction_risk(label: str, confidence: float, prob_dict: dict) -> tuple[str, str]:
    suicidal_score = float(prob_dict.get("Suicidal", 0.0))
    if label == "Suicidal" or suicidal_score >= 0.45:
        return "HIGH RISK", "high"
    if label in {"Depression", "Bipolar"} or confidence >= 0.55:
        return "MEDIUM RISK", "medium"
    return "LOW RISK", "low"


def _top_k_markdown(prob_dict: dict, k: int = 3) -> str:
    top_items = sorted(prob_dict.items(), key=lambda x: -x[1])[:k]
    lines = [f"- **{label}**: {score:.1%}" for label, score in top_items]
    return "\n".join(lines)


def _input_summary(text: str) -> str:
    stripped = text.strip()
    words = len(stripped.split()) if stripped else 0
    chars = len(stripped)
    return (
        f"<div class='metric-grid'>"
        f"<div class='metric'><div class='metric-label'>Characters</div><div class='metric-value'>{chars}</div></div>"
        f"<div class='metric'><div class='metric-label'>Words</div><div class='metric-value'>{words}</div></div>"
        f"<div class='metric'><div class='metric-label'>Mode</div><div class='metric-value'>{'Demo'}</div></div>"
        f"</div>"
    )


def build_interface(model_state: tuple):
    import gradio as gr
    import pandas as pd

    def on_submit(text):
        if not text or not text.strip():
            empty_df = pd.DataFrame([], columns=["Category", "Probability"])
            return (
                "Please enter some text.",
                "",
                "",
                _input_summary(""),
                empty_df,
            )

        label, confidence, prob_dict = run_prediction(text, model_state)
        color = CLASS_COLORS.get(label, "#607D8B")
        resource = SAFE_RESOURCES.get(label, "")
        risk_text, risk_class = _prediction_risk(label, confidence, prob_dict)
        md = f"""<div class='result-title'>Prediction</div>
<div class='result-subtitle'>Model output for the submitted text</div>
<div class='badge badge-{risk_class}'>{risk_text}</div>
<div style='font-size:1.35rem; font-weight:800; color:{color}; margin:4px 0 8px 0'>{label}</div>
<div style='font-size:1rem; margin-bottom:10px'><strong>Confidence:</strong> {confidence:.1%}</div>
{f"<div class='small-note'><strong>Resource:</strong> {resource}</div>" if resource else ""}
<div class='small-note'>This is a research demo, not a clinical diagnosis tool.</div>"""

        sorted_probs = sorted(prob_dict.items(), key=lambda x: -x[1])
        df = pd.DataFrame(sorted_probs, columns=["Category", "Probability"])
        top3_md = _top_k_markdown(prob_dict, k=3)
        return md, top3_md, _input_summary(text), df

    mode_tag = f"Mode: **{model_state[0].upper()}**"

    with gr.Blocks(title="MindClassify — Mental Health NLP") as demo:
        gr.HTML(f"""
        <div class='hero'>
                    <h1 style='color:#ffffff; margin:0;'>MindClassify</h1>
                    <p style='color:#ffffff; margin:10px 0 0 0;'>Automatic mental health text classification for presentation and demo use.</p>
                    <p style='color:#ffffff; margin-top:10px; opacity:0.95;'>{mode_tag} | 7 labels: Normal · Depression · Suicidal · Anxiety · Stress · Bipolar · Personality Disorder</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML("""
                <div class='section-card'>
                  <div class='result-title'>Input Text</div>
                  <div class='small-note'>Paste a post, diary entry, or short message. The model will return a label, confidence, and top probabilities.</div>
                </div>
                """)
                text_input = gr.Textbox(
                    label="Text to classify",
                    placeholder="How are you feeling today? Describe your thoughts...",
                    lines=6,
                )
                with gr.Row():
                    submit_btn = gr.Button("Classify", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                input_summary = gr.HTML(_input_summary(""))

            with gr.Column(scale=2):
                gr.HTML("""
                <div class='section-card'>
                  <div class='result-title'>Prediction Output</div>
                  <div class='small-note'>The right panel shows the main label, risk level, top-3 probabilities, and a full probability chart.</div>
                </div>
                """)
                result_md = gr.HTML()
                top3_md = gr.Markdown()
                prob_bar = gr.BarPlot(
                    x="Category", y="Probability",
                    title="Class Probabilities",
                    height=320,
                    visible=True,
                )

        gr.Markdown("### Example Inputs")
        gr.Examples(examples=EXAMPLES, inputs=text_input)

        gr.Markdown("""
---
**Disclaimer:** MindClassify is a research prototype for academic purposes only. It is **not** a clinical diagnostic tool.
If you or someone you know is in crisis, please contact local emergency services or a suicide prevention hotline.
""")

        submit_btn.click(fn=on_submit, inputs=text_input, outputs=[result_md, top3_md, input_summary, prob_bar])
        text_input.submit(fn=on_submit, inputs=text_input, outputs=[result_md, top3_md, input_summary, prob_bar])
        clear_btn.click(
            fn=lambda: ("", "", _input_summary(""), pd.DataFrame([], columns=["Category", "Probability"])),
            inputs=[],
            outputs=[result_md, top3_md, input_summary, prob_bar],
        )

    return demo


def main():
    import gradio as gr

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="auto",
                        choices=["auto", "transformer", "baseline", "demo"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", dest="server_name", default="127.0.0.1")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model_state = _load_model(args.model_type, args.model_path)
    print(f"Model type: {model_state[0]}")

    demo = build_interface(model_state)
    demo.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        css=UI_CSS,
    )


if __name__ == "__main__":
    main()
