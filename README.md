<p align="center">
  <img src="https://img.shields.io/badge/model-T5--small-orange?style=flat-square" alt="model">
  <img src="https://img.shields.io/badge/framework-FastAPI-009688?style=flat-square" alt="framework">
  <img src="https://img.shields.io/badge/dataset-SAMSum-blue?style=flat-square" alt="dataset">
  <img src="https://img.shields.io/badge/task-Abstractive%20Summarization-purple?style=flat-square" alt="task">
  <img src="https://img.shields.io/badge/hardware-CPU%20%7C%20CUDA%20%7C%20MPS-lightgrey?style=flat-square" alt="hardware">
</p>

<h1 align="center">Dialogue Summarizer</h1>
<p align="center"><i>Abstractive dialogue summarization using a fine-tuned T5-small transformer, served with a FastAPI backend and a minimal web UI.</i></p>

---

## Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Training — Step by Step](#training--step-by-step)
- [Running the App](#running-the-app)
- [API Reference](#api-reference)
- [Training Results](#training-results)
- [Tech Stack](#tech-stack)

---

## Overview

**Dialogue Summarizer** fine-tunes a `T5-small` model on the [SAMSum dataset](https://huggingface.co/datasets/samsum) — a corpus of ~16,000 messenger-style conversations with human-written summaries. The trained model is served through a **FastAPI** backend with a clean browser-based interface where users can paste any conversation and receive an abstractive summary instantly.

> ⚠️ Model weights are **not** included in this repo (file size ~230MB). You can reproduce them fully using the provided notebook — see [Training — Step by Step](#training--step-by-step).

---

## Demo

![App Screenshot](https://raw.githubusercontent.com/Soumadipta-Konar/dialogue-summarizer/main/demo.png)

> **Sample Input:**
> ```
> Reporter: AI continues to expand rapidly across industries...
> Expert: AI systems are becoming more capable due to deep learning...
> ```
> **Generated Summary:**
> ```
> ai technology continues to expand rapidly across industries, from healthcare
> to finance and education. ai adoption has significantly increased over the past few years.
> ```

---

## Project Structure

```
dialogue-summarizer/
├── templates/
│   └── index.html              # Frontend UI (HTML + CSS + Vanilla JS)
├── app.py                      # FastAPI backend — loads model, serves UI & /summarize/ endpoint
├── Text_Summarizer.ipynb       # End-to-end training notebook
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## How It Works

```
User Input (dialogue)
        │
        ▼
  clean_data()          ← strips whitespace, HTML tags, normalises case
        │
        ▼
  T5Tokenizer           ← tokenizes to max_length=512, returns PyTorch tensors
        │
        ▼
  T5ForConditionalGeneration.generate()
    - num_beams=4       ← beam search for better quality
    - max_length=150    ← summary length cap
    - early_stopping    ← stop when all beams hit EOS
        │
        ▼
  tokenizer.decode()    ← skips special tokens, returns readable summary
        │
        ▼
  JSON Response → { "summary": "..." }
```

---

## Training — Step by Step

This section walks you through reproducing the model from scratch using `Text_Summarizer.ipynb`.

### Prerequisites

```bash
# 1. Download the SAMSum dataset (CSV format) and place files in the project root:
#    samsum-train.csv
#    samsum-validation.csv
#    samsum-test.csv
# Available at: https://huggingface.co/datasets/samsum

# 2. Install dependencies
pip install -r requirements.txt
```

### Notebook Walkthrough

Open `Text_Summarizer.ipynb` and run cells in order:

#### Step 1 — Load Data
```python
import pandas as pd
from transformers import T5Tokenizer, Trainer, TrainingArguments, T5ForConditionalGeneration

train_data = pd.read_csv("samsum-train.csv")      # 14,732 rows
val_data   = pd.read_csv("samsum-validation.csv") # 818 rows
```

#### Step 2 — Sample (optional, speeds up training)
```python
# Training uses a 4000-sample subset for faster iteration
train_data = train_data.sample(n=4000, random_state=42).reset_index(drop=True)
val_data   = val_data.sample(n=500,  random_state=42).reset_index(drop=True)
```

#### Step 3 — Preprocess
```python
# Strips \r\n, extra whitespace, HTML tags, and lowercases
train_data["dialogue"] = train_data["dialogue"].apply(clean_data)
train_data["summary"]  = train_data["summary"].apply(clean_data)
val_data["dialogue"]   = val_data["dialogue"].apply(clean_data)
val_data["summary"]    = val_data["summary"].apply(clean_data)
```

#### Step 4 — Tokenize
```python
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def tokenize(data):
    inputs  = tokenizer(data["dialogue"], padding="max_length", max_length=512, truncation=True)
    targets = tokenizer(data["summary"],  padding="max_length", max_length=150, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

train_dataset = train_data.apply(tokenize, axis=1).tolist()
val_dataset   = val_data.apply(tokenize,   axis=1).tolist()
```

#### Step 5 — Load Model & Train
```python
model = T5ForConditionalGeneration.from_pretrained("t5-small")
model.to(device)  # auto-detects CUDA / MPS / CPU

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=6,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    warmup_steps=500
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset
)

trainer.train()
```

#### Step 6 — Save the Model
```python
# Saves model weights + tokenizer config to ./saved_summary_model/
model.save_pretrained("./saved_summary_model")
tokenizer.save_pretrained("./saved_summary_model")
```

#### Step 7 — Test the Model (quick sanity check)
```python
model     = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

summary = summarize_dialogue("Reporter: AI is expanding rapidly...")
print("Summary:", summary)
# → "ai technology continues to expand rapidly across industries..."
```

> ⏱️ **Training time:** ~13 minutes on a CUDA GPU (RTX class), significantly longer on CPU.

---

## Running the App

### 1. Clone the repo

```bash
git clone https://github.com/Soumadipta-Konar/dialogue-summarizer.git
cd dialogue-summarizer
```

### 2. Set up environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Train the model (if not already done)

Follow the [Training — Step by Step](#training--step-by-step) guide above. Ensure `./saved_summary_model/` exists before starting the server.

### 4. Start the server

```bash
uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## API Reference

### `POST /summarize/`

Accepts a JSON body and returns an abstractive summary.

**Request:**
```json
{
  "dialogue": "Alice: Hey, are we still on for Saturday?\nBob: Yes! I'll bring the food."
}
```

**Response:**
```json
{
  "summary": "alice and bob confirmed their plans for saturday. bob will bring the food."
}
```

---

## Training Results

| Epoch | Training Loss | Validation Loss |
|-------|--------------|----------------|
| 1 | 3.598 | 0.381 |
| 2 | 0.397 | 0.359 |
| 3 | 0.374 | 0.355 |
| 4 | 0.362 | 0.350 |
| 5 | 0.355 | 0.350 |
| **6** | **0.352** | **0.349** |

> Trained on 4,000 samples · 6 epochs · batch size 8 · ~13 min on GPU

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Model | T5-small (HuggingFace Transformers) |
| Training | PyTorch + HuggingFace Trainer API |
| Dataset | SAMSum Corpus (~16k dialogues) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, Vanilla JS |
| Hardware | CUDA GPU / Apple MPS / CPU |

---

## License

MIT License — feel free to use, modify, and distribute.
