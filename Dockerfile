# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System dependencies (curl for healthcheck, build tools for some wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first to keep the layer small and reproducible,
# then install the remaining requirements (pip skips torch since it's already met)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download NLTK corpora so the container is self-contained
RUN python -c "\
import nltk; \
nltk.download('stopwords', quiet=True); \
nltk.download('wordnet', quiet=True); \
nltk.download('omw-1.4', quiet=True)"

COPY . .

# Ensure runtime directories exist (also created at runtime via volume mounts)
RUN mkdir -p data saved_models plots

ENV PYTHONUNBUFFERED=1 \
    PORT=5001 \
    MODEL_TYPE=auto

EXPOSE 5001

CMD ["python", "main.py"]
