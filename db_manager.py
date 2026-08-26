# This module manages SQLite database for storing:
# - document metadata
# - text chunks
# - embeddings (stored as BLOB)

import sqlite3
import numpy as np
import os


def create_database(db_path):
    """
    Create SQLite database with required tables.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create documents table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT,
        created_at TEXT,
        embedding_model TEXT
    )
    """)

    # Create chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        document_id INTEGER,
        page_number INTEGER,
        chunk_text TEXT,
        FOREIGN KEY(document_id) REFERENCES documents(id)
    )
    """)

    # Create embeddings table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        chunk_id INTEGER,
        vector BLOB,
        FOREIGN KEY(chunk_id) REFERENCES chunks(id)
    )
    """)

    conn.commit()
    conn.close()


def insert_document_data(db_path, file_name, chunks, embeddings):
    """
    Insert document metadata, chunks and embeddings into database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Insert document metadata
    cursor.execute("""
    INSERT INTO documents (file_name, created_at, embedding_model)
    VALUES (?, datetime('now'), ?)
    """, (file_name, "all-MiniLM-L6-v2"))

    document_id = cursor.lastrowid

    # Insert chunks and embeddings
    for chunk, embedding in zip(chunks, embeddings):

        cursor.execute("""
        INSERT INTO chunks (document_id, page_number, chunk_text)
        VALUES (?, ?, ?)
        """, (document_id, chunk["page"], chunk["text"]))

        chunk_id = cursor.lastrowid

        # Convert numpy array to bytes before storing
        vector_bytes = embedding.astype(np.float32).tobytes()

        cursor.execute("""
        INSERT INTO embeddings (chunk_id, vector)
        VALUES (?, ?)
        """, (chunk_id, vector_bytes))

    conn.commit()
    conn.close()


def validate_database(db_path):
    """
    Validate database contents.
    Returns count of documents, chunks, and embeddings.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM embeddings")
    embed_count = cursor.fetchone()[0]

    conn.close()

    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "embeddings": embed_count
    }


# -----------------------------
# TEST BLOCK (PDF → .db BUILD)
# -----------------------------
if __name__ == "__main__":

    from pdf_processor import extract_text_from_pdf
    from chunker import chunk_text
    from embedder import generate_embeddings

    db_folder = "rag_database"
    db_path = os.path.join(db_folder, "sample.db")

    # Ensure database folder exists
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    # Create database schema
    create_database(db_path)

    # Extract PDF
    pages = extract_text_from_pdf("sample.pdf")

    # Chunk text
    chunks = chunk_text(pages)

    # Generate embeddings
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)

    # Insert into database
    insert_document_data(db_path, "sample.pdf", chunks, embeddings)

    # Validate
    stats = validate_database(db_path)

    print("\nDatabase successfully created!")
    print("Validation Results:", stats)
