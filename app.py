import streamlit as st
import os
import traceback

from file_processor import extract_text_from_file
from chunker import chunk_text
from embedder import generate_embeddings
from db_manager import create_database, insert_document_data, validate_database


# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Universal Multi-File RAG DB Converter")

st.title("📂 Multi-File to RAG Database Converter")
st.write("Upload multiple supported files to create a single portable RAG .db file.")


supported_types = [
    "pdf", "txt", "docx", "csv", "json",
    "html", "htm", "png", "jpg", "jpeg"
]

uploaded_files = st.file_uploader(
    "Upload Files",
    type=supported_types,
    accept_multiple_files=True
)


if uploaded_files:

    st.write("### 📄 Uploaded Files")
    for file in uploaded_files:
        st.write(f"- {file.name} ({round(file.size / 1024, 2)} KB)")

    db_name = st.text_input("Enter database name", value="knowledge_base")

    if st.button("Convert All to RAG Database"):

        try:
            db_folder = "rag_database"
            if not os.path.exists(db_folder):
                os.makedirs(db_folder)

            db_path = os.path.join(db_folder, f"{db_name}.db")

            if os.path.exists(db_path):
                os.remove(db_path)

            create_database(db_path)

            all_chunks = []

            # -----------------------------
            # Process Each File
            # -----------------------------
            for uploaded_file in uploaded_files:

                file_path = os.path.join(db_folder, uploaded_file.name)

                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())

                st.info(f"🔍 Extracting text from {uploaded_file.name}...")
                pages = extract_text_from_file(file_path)

                if not pages:
                    continue

                chunks = chunk_text(pages)

                # Add source metadata
                for chunk in chunks:
                    chunk["source"] = uploaded_file.name

                all_chunks.extend(chunks)

            if not all_chunks:
                st.error("No text could be extracted from uploaded files.")
                st.stop()

            st.info("🧠 Generating embeddings for all files...")

            texts = [chunk["text"] for chunk in all_chunks]
            embeddings = generate_embeddings(texts)

            # Insert all data
            insert_document_data(db_path, "Multiple Files", all_chunks, embeddings)

            stats = validate_database(db_path)

            st.success("✅ Multi-file Database created successfully!")
            st.write("### 📊 Database Stats")
            st.json(stats)

            with open(db_path, "rb") as f:
                st.download_button(
                    label="⬇ Download RAG Database (.db)",
                    data=f,
                    file_name=f"{db_name}.db",
                    mime="application/octet-stream"
                )

        except Exception:
            st.error("❌ Something went wrong.")
            st.code(traceback.format_exc())

