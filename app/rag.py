import os
import time
import faiss
import numpy as np
import nltk
from dotenv import load_dotenv
from pathlib import Path

from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize
from numpy.char import index

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FOLDER = "data"

# Hugging Face cloud embedding model
HF_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# You can also try:
# HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini cloud LLM
GEMINI_MODEL = "gemini-2.5-flash"

TOP_K = 5

# Start with 1 to avoid connection problems.
# Later you can try 4 or 8.
BATCH_SIZE = 1


# ==========================================================
# CLIENTS
# ==========================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


# ==========================================================
# NLTK SETUP
# ==========================================================

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def load_documents(folder=DATA_FOLDER):
    """
    Load .txt files from the data folder and split them into text chunks.
    """

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    chunks = []

    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            sentences = sent_tokenize(text)

            for sentence in sentences:
                sentence = sentence.strip()

                if sentence:
                    chunks.append(sentence)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure the '{folder}' folder contains .txt files."
        )

    print(f"Loaded {len(chunks)} text chunks.")
    return chunks


# ==========================================================
# HUGGING FACE CLOUD EMBEDDINGS
# ==========================================================

def normalize_embedding_output(raw_output, expected_count):
    """
    Converts Hugging Face embedding output into a clean 2D numpy array.

    Final shape:
        [number_of_texts, embedding_dimension]
    """

    arr = np.array(raw_output, dtype="float32")

    # Case 1:
    # Single embedding:
    # [embedding_dimension]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Case 2:
    # Batch embeddings:
    # [batch_size, embedding_dimension]
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass

        # Token embeddings for one input:
        # [tokens, embedding_dimension]
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)

        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )

    # Case 3:
    # Token embeddings for batch:
    # [batch_size, tokens, embedding_dimension]
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)

    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


def hf_feature_extraction_with_retries(inputs, expected_count, max_retries=5):
    """
    Calls Hugging Face cloud embedding model with retries.

    inputs can be:
    - string
    - list of strings
    """

    for attempt in range(1, max_retries + 1):
        try:
            result = hf_client.feature_extraction(
                inputs,
                model=HF_EMBEDDING_MODEL
            )

            embeddings = normalize_embedding_output(
                raw_output=result,
                expected_count=expected_count
            )

            return embeddings

        except Exception as e:
            print(
                f"Hugging Face embedding failed. Attempt {attempt}/{max_retries}")
            print("Error:", e)

            if attempt == max_retries:
                raise

            wait_seconds = attempt * 3
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)


def embed_texts_with_huggingface(texts, batch_size=BATCH_SIZE):
    """
    Creates document embeddings using Hugging Face cloud inference.
    """

    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        current_batch = start // batch_size + 1
        print(f"Embedding batch {current_batch}/{total_batches}...")

        embeddings = hf_feature_extraction_with_retries(
            inputs=batch,
            expected_count=len(batch)
        )

        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings).astype("float32")

    print(f"Created document embeddings. Shape: {final_embeddings.shape}")

    return final_embeddings


def embed_query_with_huggingface(query):
    """
    Creates one query embedding using Hugging Face cloud inference.
    """

    embedding = hf_feature_extraction_with_retries(
        inputs=query,
        expected_count=1
    )

    return embedding.astype("float32")


# ==========================================================
# FAISS VECTOR SEARCH
# ==========================================================
# ==========================================================
# FAISS VECTOR SEARCH
# ==========================================================


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FOLDER = BASE_DIR / "data"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.npy"
EMBEDDINGS_PATH = VECTOR_STORE_DIR / "embeddings.npy"


def save_vector_store(index, embeddings, chunks):
    """
    Save FAISS index + embeddings + chunks to disk.
    """

    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    np.save(EMBEDDINGS_PATH, embeddings)

    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))

    print("Vector store saved to disk.")


def load_vector_store():
    print("BASE_DIR:", BASE_DIR)
    print("VECTOR_STORE_DIR:", VECTOR_STORE_DIR)
    print("FAISS_INDEX_PATH:", FAISS_INDEX_PATH)
    print("FAISS EXISTS:", FAISS_INDEX_PATH.exists())
    """
    Load FAISS index + embeddings + chunks from disk.
    """

    if not FAISS_INDEX_PATH.exists():
        return None, None, None

    index = faiss.read_index(str(FAISS_INDEX_PATH))

    embeddings = np.load(EMBEDDINGS_PATH)

    chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()

    print("Vector store loaded from disk.")

    return index, embeddings, chunks


def create_faiss_index(embeddings):
    """
    Creates FAISS index.
    """

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    return index


def retrieve(query, index, chunks, k=TOP_K):
    """
    Embeds the query and retrieves the top-k most relevant chunks from the FAISS index.
    """

    query_embedding = embed_query_with_huggingface(query)
    faiss.normalize_L2(query_embedding)

    distances, indices = index.search(query_embedding, k)

    return [chunks[i] for i in indices[0] if i < len(chunks)]

# ==========================================================
# GEMINI LLM
# ==========================================================


def ask_gemini(context, question):
    """
    Gemini is the LLM.
    Hugging Face is only used for embeddings.
    """

    prompt = f"""
You are a helpful RAG assistant that answers questions about Aviv.

The user may ask in Hebrew or English.
Answer in the same language as the user's question.
If the question is in Hebrew, answer in clear and natural Hebrew.

Use the provided context to answer the user's question.

Important background rule:
If the context says or implies that Aviv has a Bachelor's degree in Computer Science,
you may reasonably assume that he has academic familiarity with common CS topics, such as:
- programming
- object-oriented programming
- data structures
- algorithms
- operating systems
- databases
- computer networks
- software engineering basics

Rules:
1. First answer using only the provided context.
2. You may add reasonable assumptions only when they are based on Aviv's CS degree/background.
3. Clearly separate facts from assumptions.
4. Do not invent specific experience, companies, grades, projects, or technologies unless they appear in the context.
5. If the context does not contain enough information, say that clearly.
6. Keep the answer simple, clear, and professional.
7. If answering in Hebrew, use this phrase when needed:
   "לא מצאתי מספיק מידע במסמכים כדי לענות בוודאות, אבל על סמך הרקע של אביב במדעי המחשב אפשר להניח ש..."

Context:
{context}

Question:
{question}

Answer:
"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=700,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    return response.text.strip()


# ==========================================================
# MAIN APP
# ==========================================================

def main():
    setup_nltk()

    print("Checking for existing vector store...")

    index, embeddings, chunks = load_vector_store()

    if index is None:
        print("No existing vector store found.")

        print("Loading documents...")
        chunks = load_documents(DATA_FOLDER)

        print("\nCreating Hugging Face cloud embeddings...")
        embeddings = embed_texts_with_huggingface(chunks)

        print("\nCreating FAISS index...")
        index = create_faiss_index(embeddings)

        save_vector_store(index, embeddings, chunks)

    else:
        print("Using cached vector store.")

    print("\nRAG system is ready.")
    print("Embeddings: Hugging Face cloud")
    print("Vector search: FAISS local")
    print("LLM: Gemini cloud")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk something: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a real question.")
            continue

        top_chunks = retrieve(
            query=question,
            index=index,
            chunks=chunks,
            k=TOP_K
        )

        context = "\n".join(top_chunks)

        print("\nRetrieved Context:")
        print(context)

        answer = ask_gemini(context, question)

        print("\nGemini Answer:")
        print(answer)


if __name__ == "__main__":
    main()
