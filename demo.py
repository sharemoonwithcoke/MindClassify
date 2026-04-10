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
import sys

import numpy as np

# ── Colour palette for classes ────────────────────────────────────────────────
CLASS_COLORS = {
    "Normal": "#4CAF50",
    "Depression": "#2196F3",
    "Suicidal": "#F44336",
    "Anxiety": "#FF9800",
    "Stress": "#9C27B0",
    "Bipolar": "#00BCD4",
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


# ── Prediction helpers ────────────────────────────────────────────────────────

def _load_model(model_type: str, model_path: str | None):
    if model_type == "transformer" and model_path:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval().to(device)
        return ("transformer", model, tokenizer, device)

    if model_type == "baseline" and model_path:
        import joblib
        return ("baseline", joblib.load(model_path), None, None)

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

        pkl = os.path.join(save_dir, "baseline.pkl")
        if os.path.exists(pkl):
            return _load_model("baseline", pkl)

    return ("demo", None, None, None)


def run_prediction(text: str, state: tuple) -> tuple:
    from data_preprocessing import clean_text, ID2LABEL

    kind, model, tokenizer, device = state

    if kind == "transformer":
        import torch
        cleaned = clean_text(text)
        enc = tokenizer(cleaned, truncation=True, padding=True,
                        max_length=128, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        confidence = float(probs[pred_id])
        prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(len(LABEL_NAMES))}

    elif kind == "baseline":
        from data_preprocessing import clean_text
        cleaned = clean_text(text)
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


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_interface(model_state: tuple):
    import gradio as gr

    def classify(text: str):
        if not text or not text.strip():
            return "Please enter some text.", None, None

        label, confidence, prob_dict = run_prediction(text, model_state)

        # Result markdown
        color = CLASS_COLORS.get(label, "#607D8B")
        resource = SAFE_RESOURCES.get(label, "")
        md = f"""
### Prediction: <span style='color:{color}; font-weight:bold'>{label}</span>
**Confidence:** {confidence:.1%}

{f'> {resource}' if resource else ''}
{'> ⚠️ *This is a demo system — not a medical diagnosis tool.*' if model_state[0] == 'demo' else '> *Note: Not a substitute for professional mental health advice.*'}
"""
        # Bar chart data for Gradio
        sorted_probs = sorted(prob_dict.items(), key=lambda x: -x[1])
        bar_labels = [item[0] for item in sorted_probs]
        bar_values = [item[1] for item in sorted_probs]

        return md, (bar_labels, bar_values), prob_dict

    mode_tag = f"Mode: **{model_state[0].upper()}**"

    with gr.Blocks(title="MindClassify — Mental Health NLP", theme=gr.themes.Soft()) as demo:
        gr.Markdown(f"""
# MindClassify
### Automatic Mental Health Classification via NLP
{mode_tag} | 7 categories: Normal · Depression · Suicidal · Anxiety · Stress · Bipolar · Personality Disorder
---
""")

        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(
                    label="Enter text (social media post, diary entry, etc.)",
                    placeholder="How are you feeling today? Describe your thoughts...",
                    lines=5,
                )
                submit_btn = gr.Button("Classify", variant="primary")

            with gr.Column(scale=2):
                result_md = gr.Markdown(label="Result")
                prob_bar = gr.BarPlot(
                    x="Category",
                    y="Probability",
                    title="Class Probabilities",
                    height=280,
                    visible=True,
                )

        gr.Markdown("### Examples")
        gr.Examples(
            examples=EXAMPLES,
            inputs=text_input,
        )

        gr.Markdown("""
---
**Disclaimer:** MindClassify is a research prototype for academic purposes only.
It is **not** a clinical diagnostic tool. If you or someone you know needs help,
please contact a qualified mental health professional.
""")

        def on_submit(text):
            if not text or not text.strip():
                return "Please enter some text.", gr.update(visible=False)
            label, confidence, prob_dict = run_prediction(text, model_state)
            color = CLASS_COLORS.get(label, "#607D8B")
            resource = SAFE_RESOURCES.get(label, "")
            md = f"""### Prediction: <span style='color:{color}; font-weight:bold'>{label}</span>
**Confidence:** {confidence:.1%}

{f"> {resource}" if resource else ""}
> *Not a substitute for professional mental health advice.*"""

            import pandas as pd
            sorted_probs = sorted(prob_dict.items(), key=lambda x: -x[1])
            df = pd.DataFrame(sorted_probs, columns=["Category", "Probability"])
            return md, df

        submit_btn.click(fn=on_submit, inputs=text_input, outputs=[result_md, prob_bar])
        text_input.submit(fn=on_submit, inputs=text_input, outputs=[result_md, prob_bar])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="auto", choices=["auto", "transformer", "baseline", "demo"])
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--server-name", dest="server_name", default="127.0.0.1",
                        help="Host to bind (use 0.0.0.0 inside Docker)")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio link")
    args = parser.parse_args()

    print("Loading model...")
    model_state = _load_model(args.model_type, args.model_path)
    print(f"Model type: {model_state[0]}")

    demo = build_interface(model_state)
    demo.launch(server_name=args.server_name, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
