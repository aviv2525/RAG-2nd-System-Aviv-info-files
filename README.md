# Aviv RAG — Personal Q&A System

A Retrieval-Augmented Generation (RAG) system that answers questions about Aviv in Hebrew or English.
Built with HuggingFace embeddings, FAISS vector search, Gemini 2.5 Flash, Flask, and Docker.

---

# How It Works

```text
Question
   ↓
HuggingFace Embedding
   ↓
FAISS Similarity Search
   ↓
Top Relevant Chunks
   ↓
Gemini 2.5 Flash
   ↓
Answer
```

1. Documents (`.txt` / `.pdf`) are split into sentence chunks
2. HuggingFace generates embeddings
3. FAISS retrieves the most relevant chunks
4. Gemini generates an answer using the retrieved context
5. Answers are returned in Hebrew or English depending on the question

---

# Features

* Persistent FAISS vector database
* Dockerized setup
* Flask web interface
* Upload `.txt` and `.pdf` files
* Hebrew + English support
* Source chunk viewer
* Session history
* Drag & drop uploads
* Cached embeddings (no rebuild on every startup)

---

# Tech Stack

| Component        | Tool                                                          |
| ---------------- | ------------------------------------------------------------- |
| Embeddings       | HuggingFace Inference API                                     |
| Embedding Model  | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Vector Search    | FAISS                                                         |
| LLM              | Gemini 2.5 Flash                                              |
| Backend          | Flask                                                         |
| PDF Parsing      | pypdf                                                         |
| Containerization | Docker + Docker Compose                                       |

---

# Project Structure

```text
RAG_Project/
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   ├── rag.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
│
├── data/
│   └── *.txt / *.pdf
│
├── vector_store/
│   ├── faiss.index
│   ├── embeddings.npy
│   └── chunks.npy
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── .gitignore
└── .dockerignore
```

---

# Persistent Vector Store

The system stores:

* FAISS index
* embeddings
* document chunks

inside:

```text
vector_store/
```

This prevents rebuilding embeddings every startup.

Startup behavior:

```text
If vector_store exists:
    Load existing FAISS index

Else:
    Build embeddings and save them
```

---

# Setup (Local)

## 1. Clone the repository

```bash
git clone https://github.com/aviv2525/RAG-2nd-System-Aviv-info-files
cd https://github.com/aviv2525/RAG-2nd-System-Aviv-info-files
```

---

## 2. Create virtual environment

```bash
python -m venv .venv
```

Activate:

### Windows

```bash
.venv\Scripts\activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure environment variables

Create `.env`

```env
GEMINI_API_KEY=your_gemini_key
HF_TOKEN=your_huggingface_token
```

---

## 5. Add documents

Place `.txt` or `.pdf` files inside:

```text
data/
```

---

## 6. Run locally

```bash
python -m app.app
```

Open:

```text
http://localhost:5000
```

---

# Docker Setup

## Build

```bash
docker compose build
```

---

## Run

```bash
docker compose up
```

---

## Stop

```bash
CTRL + C
```

or:

```bash
docker compose down
```

---

# Docker Volumes

Docker persists:

```text
./data
./vector_store
```

This keeps:

* uploaded files
* embeddings
* FAISS index

even after container restarts.

---

# CLI Mode

Run without Flask UI:

```bash
python -m app.rag
```

Type questions directly in terminal.

Type:

```text
exit
```

to quit.

---

# Environment Variables

| Variable       | Description           |
| -------------- | --------------------- |
| GEMINI_API_KEY | Google Gemini API key |
| HF_TOKEN       | HuggingFace API token |

---

# Notes

* The first startup may take longer because embeddings are created
* Later startups load the cached FAISS index instantly
* Uploaded files are added dynamically to the vector database
* The system currently uses cloud embeddings via HuggingFace API

---

# Future Improvements

* Local embeddings with `sentence-transformers`
* FastAPI backend
* Streaming responses
* Authentication
* Vector DB migration (Qdrant / Chroma)
* Production deployment
