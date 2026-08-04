# 🗒️ Dialogue Summarizer — T5

A dialogue summarization web app powered by a fine-tuned **T5-small** transformer model, served with **FastAPI**. Paste any conversation and get a concise, abstractive summary instantly.

---

---

## 🧠 Model

- **Architecture**: T5-small (Text-to-Text Transfer Transformer)
- **Task**: Abstractive Dialogue Summarization
- **Dataset**: [SAMSum Corpus](https://huggingface.co/datasets/samsum) — ~16k messenger-style conversations with human-written summaries
- **Framework**: HuggingFace Transformers + PyTorch

> The model weights are not included in this repo. Train your own using the provided notebook.

---

## 📁 Project Structure

```
dialogue-summarizer/
├── templates/
│   └── index.html              # Frontend UI
├── app.py                      # FastAPI backend
├── Text_Summarizer.ipynb       # Training notebook (T5 fine-tuning)
├── requirements.txt            # Python dependencies
└── .gitignore
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/Soumadipta-Konar/dialogue-summarizer.git
cd dialogue-summarizer
```

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
Open and run `Text_Summarizer.ipynb` end-to-end. This will:
- Load and preprocess the SAMSum dataset
- Fine-tune T5-small
- Save the model to `./saved_summary_model/`

### 5. Run the app
```bash
uvicorn app:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Model | T5-small (HuggingFace Transformers) |
| Backend | FastAPI + Uvicorn |
| Frontend | HTML, CSS, Vanilla JS |
| Training | PyTorch |

---

## 📜 License

MIT License — feel free to use and modify.
