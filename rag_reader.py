import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding model (same one used during creation)
model = SentenceTransformer("all-MiniLM-L6-v2")


def load_rag_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT chunks.id, chunks.chunk_text, embeddings.vector
        FROM chunks
        JOIN embeddings ON chunks.id = embeddings.chunk_id
    """)

    data = cursor.fetchall()
    conn.close()

    chunk_ids = []
    texts = []
    vectors = []

    for row in data:
        chunk_ids.append(row[0])
        texts.append(row[1])
        vectors.append(np.frombuffer(row[2], dtype=np.float32))

    return texts, np.array(vectors)


def query_rag(db_path, question, top_k=3):
    texts, vectors = load_rag_database(db_path)

    question_embedding = model.encode([question])
    similarities = cosine_similarity(question_embedding, vectors)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    retrieved_context = "\n\n".join([texts[i] for i in top_indices])

    print("\nRetrieved Context:\n")
    print(retrieved_context[:1000])

    # Simple answer generation (without external API)
    print("\n\nSuggested Answer:\n")
    print(retrieved_context[:500])



if __name__ == "__main__":
    db_path = "rag_database/knowledge_base.db"

    question = input("Enter your question: ")

    query_rag(db_path, question)
