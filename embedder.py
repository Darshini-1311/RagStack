# This module loads a sentence-transformer model locally.
# It generates embeddings for text chunks.
# Return numpy array of embeddings.

from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model once
model = SentenceTransformer("all-MiniLM-L6-v2")

def generate_embeddings(text_list):
    """
    Convert list of text chunks into embedding vectors.
    """
    embeddings = model.encode(text_list)
    return np.array(embeddings)
if __name__ == "__main__":
    from pdf_processor import extract_text_from_pdf
    from chunker import chunk_text

    pages = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(pages)

    texts = [chunk["text"] for chunk in chunks]
    vectors = generate_embeddings(texts)

    print("Embedding shape:", vectors.shape)
