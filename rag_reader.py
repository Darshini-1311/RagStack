import sqlite3
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Load the same embedding model used during creation
# --------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")


# --------------------------------------------------
# Automatically find the latest valid RAG database
# --------------------------------------------------
def find_latest_database():

    db_folder = "rag_database"

    if not os.path.exists(db_folder):
        raise FileNotFoundError(
            "rag_database folder does not exist."
        )

    db_files = []

    for file in os.listdir(db_folder):

        if file.lower().endswith(".db"):

            full_path = os.path.join(db_folder, file)

            # Ignore empty/incomplete databases
            if os.path.getsize(full_path) > 0:
                db_files.append(full_path)

    if not db_files:
        raise FileNotFoundError(
            "No valid .db file found inside rag_database."
        )

    # Select the most recently modified database
    latest_db = max(
        db_files,
        key=os.path.getmtime
    )

    return latest_db


# --------------------------------------------------
# Load RAG database
# --------------------------------------------------
def load_rag_database(db_path):

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT chunks.id,
               chunks.chunk_text,
               embeddings.vector
        FROM chunks
        JOIN embeddings
        ON chunks.id = embeddings.chunk_id
    """)

    data = cursor.fetchall()

    conn.close()

    if not data:
        raise ValueError(
            "The selected database does not contain any chunks."
        )

    texts = []
    vectors = []

    for row in data:

        texts.append(row[1])

        vector = np.frombuffer(
            row[2],
            dtype=np.float32
        )

        vectors.append(vector)

    return texts, np.array(vectors)


# --------------------------------------------------
# Query RAG database
# --------------------------------------------------
def query_rag(db_path, question, top_k=3):

    texts, vectors = load_rag_database(db_path)

    # Convert question into embedding
    question_embedding = model.encode([question])

    # Calculate similarity
    similarities = cosine_similarity(
        question_embedding,
        vectors
    )[0]

    # Get top matching chunks
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_context = "\n\n".join(
        [texts[i] for i in top_indices]
    )

    print("\n" + "=" * 70)
    print("RETRIEVED CONTEXT")
    print("=" * 70)

    print(retrieved_context[:3000])

    print("\n" + "=" * 70)
    print("SUGGESTED ANSWER")
    print("=" * 70)

    print(retrieved_context[:1000])


# --------------------------------------------------
# Main
# --------------------------------------------------
if __name__ == "__main__":

    try:

        # Automatically select latest database
        db_path = find_latest_database()

        print("\n" + "=" * 70)
        print("RAG DATABASE")
        print("=" * 70)

        print(db_path)

        print("=" * 70)

        question = input("\nEnter your question: ")

        query_rag(
            db_path,
            question
        )

    except Exception as e:

        print("\nERROR:")
        print(e)