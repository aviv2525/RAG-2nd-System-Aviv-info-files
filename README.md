# Aviv RAG — Personal Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about Aviv in Hebrew or English.  
Built with HuggingFace embeddings, FAISS vector search, Gemini 2.5 Flash, and a Flask web interface.

---

## How It Works

```
Question → HuggingFace Embeddings → FAISS Search → Top Chunks → Gemini 2.5 Flash → Answer
```

1. Documents (`.txt` / `.pdf`) are split into sentences and embedded via HuggingFace
2. FAISS finds the most relevant chunks for the question
3. Gemini generates an answer based on the retrieved context
4. The answer is returned in the same language as the question (Hebrew or English)

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` via HuggingFace API |
| Vector Search | FAISS (local, cosine similarity) |
| LLM | Gemini 2.5 Flash (Google AI) |
| Web Framework | Flask |
| PDF Parsing | pypdf |

---

## Project Structure

```
RAG_Project/
├── Rag.py               # Core RAG logic (embeddings, FAISS, Gemini)
├── app.py               # Flask web server
├── templates/
│   └── index.html       # Web UI
├── static/
│   └── style.css        # Styling
├── data/                # Place your .txt / .pdf documents here
├── .env                 # API keys (not committed)
├── .env.example         # Key template
└── requirements.txt
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

### 2. Create a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # macOS / Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```
GEMINI_API_KEY=your_gemini_key_here
HF_TOKEN=your_huggingface_token_here
```

- **Gemini API key** → [aistudio.google.com](https://aistudio.google.com)
- **HuggingFace token** → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 5. Add your documents

Place `.txt` or `.pdf` files inside the `data/` folder.

### 6. Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

---

## Features

- **Bilingual** — ask in Hebrew or English, get an answer in the same language
- **File upload** — attach `.txt` or `.pdf` files directly from the web UI; they are embedded and added to the index instantly
- **Source viewer** — toggle "Show sources" to see which document chunks were retrieved
- **Session history** — previous Q&A pairs are displayed below the current answer
- **Drag & drop** — drag a file onto the upload area

---

## CLI Mode

You can also run the RAG system without the web interface:

```bash
python Rag.py
```

Type your question and press Enter. Type `exit` to quit.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API key |
| `HF_TOKEN` | HuggingFace access token (for embeddings) |
