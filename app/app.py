import os
import uuid
import faiss
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from nltk.tokenize import sent_tokenize
from pypdf import PdfReader

from app.rag import (
    save_vector_store,
    load_vector_store,
    setup_nltk,
    load_documents,
    embed_texts_with_huggingface,
    create_faiss_index,
    retrieve,
    ask_gemini,
    DATA_FOLDER,
    TOP_K,
)

app = Flask(__name__)

chunks = None
index = None
ready = False
startup_error = None


def initialize_rag():
    global chunks, index, ready, startup_error

    try:
        setup_nltk()

        print("Checking for existing vector store...")

        index, embeddings, chunks = load_vector_store()

        if index is None:
            print("No vector store found. Building new one...")

            chunks = load_documents(DATA_FOLDER)

            embeddings = embed_texts_with_huggingface(chunks)

            index = create_faiss_index(embeddings)

            save_vector_store(index, embeddings, chunks)

        else:
            print("Using cached vector store.")

        ready = True

    except Exception as e:
        startup_error = str(e)
        print("RAG startup failed:", e)


@app.route("/")
def home():
    return render_template("index.html", ready=ready, error=startup_error)


@app.route("/ask", methods=["POST"])
def ask():
    if not ready:
        return jsonify({"error": "RAG system is still loading, please wait."}), 503

    data = request.get_json()
    question = (data or {}).get("question", "").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400

    top_chunks = retrieve(query=question, index=index, chunks=chunks, k=TOP_K)
    context = "\n".join(top_chunks)
    answer = ask_gemini(context, question)

    return jsonify({"answer": answer, "chunks": top_chunks})


@app.route("/upload", methods=["POST"])
def upload():
    global chunks, index

    if not ready:
        return jsonify({"error": "RAG system is still loading, please wait."}), 503

    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    fname = file.filename.lower()
    if not (fname.endswith(".txt") or fname.endswith(".pdf")):
        return jsonify({"error": "Only .txt and .pdf files are supported."}), 400

    ext = ".pdf" if fname.endswith(".pdf") else ".txt"
    safe_name = secure_filename(file.filename)
    filename = safe_name if safe_name and safe_name != ext else f"{uuid.uuid4().hex}{ext}"

    os.makedirs(DATA_FOLDER, exist_ok=True)
    filepath = os.path.join(DATA_FOLDER, filename)
    file.save(filepath)

    try:
        if ext == ".pdf":
            reader = PdfReader(filepath)
            text = "\n".join(page.extract_text()
                             or "" for page in reader.pages)
        else:
            text = open(filepath, encoding="utf-8").read()
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {e}"}), 500

    new_chunks = [s.strip() for s in sent_tokenize(text) if s.strip()]

    if not new_chunks:
        return jsonify({"error": "File is empty or has no readable sentences."}), 400

    new_embeddings = embed_texts_with_huggingface(new_chunks)
    faiss.normalize_L2(new_embeddings)
    index.add(new_embeddings)
    chunks.extend(new_chunks)
    save_vector_store(index, new_embeddings, chunks)

    return jsonify({"message": f"Added {len(new_chunks)} chunks from '{filename}'.", "count": len(new_chunks)})


if __name__ == "__main__":
    initialize_rag()
    app.run(debug=True, port=5000)
