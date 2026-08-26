# This module splits extracted PDF text into overlapping chunks.
# Each chunk keeps the original page number.
# Chunk size: 500 words
# Overlap: 50 words
# Return format:
# [{"page": page_number, "text": chunk_text}, ...]

def chunk_text(pages, chunk_size=500, overlap=50):
    """
    pages: list of dictionaries from pdf_processor
    returns list of chunk dictionaries
    """
    chunks = []

    for page in pages:
        page_num = page["page"]
        text = page["text"]

        # Skip empty pages
        if not text:
            continue

        words = text.split()

        step = chunk_size - overlap
        if step <= 0:
            step = chunk_size

        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunks.append({
                "page": page_num,
                "text": chunk_text
            })

    return chunks


if __name__ == "__main__":
    from pdf_processor import extract_text_from_pdf

    pages = extract_text_from_pdf("sample.pdf")
    chunks = chunk_text(pages)

    print("First 2 chunks:")
    print(chunks[:2])
    print("Total chunks:", len(chunks))

