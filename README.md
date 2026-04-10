# MindClassify — Model Service

Python NLP pipeline for automatic mental health text classification.

## Setup

```bash
pip install -r requirements.txt
```

## Dataset

Place your dataset CSV at `data/mental_health.csv`.
Expected columns: `statement` (text), `status` (label).

Supported labels: Normal, Depression, Suicidal, Anxiety, Stress, Bipolar, Personality Disorder

## Workflow

### 1. Preprocess & inspect

```bash
python data_preprocessing.py data/mental_health.csv
```

### 2. Train baselines (TF-IDF + ML)

```bash
python baseline_model.py --data_path data/mental_health.csv --cv_folds 5
```

Saves best baseline to `saved_models/best_baseline.pkl`.

### 3. Fine-tune transformer

```bash
# BERT
python transformer_trainer.py \
  --model_name bert-base-uncased \
  --data_path data/mental_health.csv \
  --epochs 5 --batch_size 16 --lr 2e-5

# MentalBERT
python transformer_trainer.py \
  --model_name mental/mental-bert-base-uncased \
  --data_path data/mental_health.csv \
  --epochs 5 --batch_size 16 --lr 2e-5
```

Saves best checkpoint to `saved_models/<model_name>_best/`.

### 4. Analyse results

```bash
python analysis.py
```

Generates plots in `plots/`.

### 5. Start Flask API

```bash
python main.py
# Listens on http://localhost:5001
```

### 6. Start Gradio demo

```bash
python demo.py
# Opens http://localhost:7860

python demo.py --share   # Generate public link
```

## API Reference

### POST /predict

```json
{"text": "I've been feeling really anxious lately..."}
```

Response:
```json
{
  "label": "Anxiety",
  "label_id": 3,
  "confidence": 0.847,
  "probabilities": {
    "Normal": 0.03,
    "Depression": 0.08,
    "Suicidal": 0.01,
    "Anxiety": 0.847,
    "Stress": 0.02,
    "Bipolar": 0.01,
    "Personality Disorder": 0.003
  }
}
```

### POST /batch_predict

```json
{"texts": ["text 1", "text 2", "text 3"]}
```

### GET /health

```json
{"status": "ok", "model_type": "transformer", "labels": [...]}
```
