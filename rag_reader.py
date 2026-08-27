import os
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load the same embedding model used during creation
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Load RAG database
# --------------------------------------------------
def load_rag_database(db_path):

    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database file not found:\n{db_path}"
        )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check whether required tables exist
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )

    tables = [row[0] for row in cursor.fetchall()]

    if "chunks" not in tables:
        conn.close()
        raise ValueError(
            f"'chunks' table not found.\n"
            f"Available tables: {tables}"
        )

    if "embeddings" not in tables:
        conn.close()
        raise ValueError(
            f"'embeddings' table not found.\n"
            f"Available tables: {tables}"
        )

    # Load chunks and their corresponding embeddings
    cursor.execute("""
        SELECT
            chunks.id,
            chunks.chunk_text,
            embeddings.vector
        FROM chunks
        JOIN embeddings
            ON chunks.id = embeddings.chunk_id
        ORDER BY chunks.id
    """)

    data = cursor.fetchall()

    conn.close()

    if not data:
        raise ValueError(
            "The database contains no chunks with embeddings."
        )

    chunk_ids = []
    texts = []
    vectors = []

    for chunk_id, text, vector_blob in data:

        chunk_ids.append(chunk_id)
        texts.append(text)

        vector = np.frombuffer(
            vector_blob,
            dtype=np.float32
        )

        vectors.append(vector)

    vectors = np.array(vectors)

    # Make sure vectors have a valid shape
    if vectors.ndim != 2:
        raise ValueError(
            f"Invalid embedding shape: {vectors.shape}"
        )

    if len(texts) != len(vectors):
        raise ValueError(
            "Number of text chunks and embeddings do not match."
        )

    print(f"\nLoaded {len(texts)} chunks from database.")

    return texts, vectors


# --------------------------------------------------
# Query the RAG database
# --------------------------------------------------
def query_rag(db_path, question, top_k=3):

    texts, vectors = load_rag_database(db_path)

    # Create embedding for user's question
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True
    )

    # Calculate similarity
    similarities = cosine_similarity(
        question_embedding,
        vectors
    )[0]

    # Don't request more results than available
    top_k = min(top_k, len(texts))

    # Get highest scoring chunks
    top_indices = similarities.argsort()[-top_k:][::-1]

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    for rank, index in enumerate(top_indices, start=1):

        print(f"\nResult {rank}")
        print("-" * 70)

        print(
            f"Similarity Score: "
            f"{similarities[index]:.4f}"
        )

        print(
            f"\n{texts[index][:1000]}"
        )

    # --------------------------------------------------
    # Display combined context
    # --------------------------------------------------
    retrieved_context = "\n\n".join(
        texts[index]
        for index in top_indices
    )

    print("\n" + "=" * 70)
    print("COMBINED RETRIEVED CONTEXT")
    print("=" * 70)

    print(retrieved_context[:3000])

    print("\n" + "=" * 70)
    print("STATUS")
    print("=" * 70)

    print(
        f"Successfully retrieved "
        f"{top_k} relevant chunks."
    )


# --------------------------------------------------
# Main program
# --------------------------------------------------
if __name__ == "__main__":

    # IMPORTANT:
    # This is the database that contains your actual
    # documents, chunks and embeddings.
    db_path = r"rag_database\knowledge_base (3).db"

    print("=" * 70)
    print("RagStack - RAG Database Reader")
    print("=" * 70)

    print(f"\nDatabase:")
    print(db_path)

    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("\nPlease enter a question.")
    else:
        try:
            query_rag(
                db_path,
                question,
                top_k=3
            )

        except Exception as e:

            print("\n" + "=" * 70)
            print("ERROR")
            print("=" * 70)

            print(str(e))