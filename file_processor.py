import os
import fitz
import docx
import pandas as pd
import json
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract

# ⚠ Make sure this path is correct on your system
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# -----------------------------
# Universal File Extractor
# -----------------------------
def extract_text_from_file(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    elif extension == ".txt":
        return extract_txt(file_path)

    elif extension == ".docx":
        return extract_docx(file_path)

    elif extension == ".csv":
        return extract_csv(file_path)

    elif extension == ".json":
        return extract_json(file_path)

    elif extension in [".html", ".htm"]:
        return extract_html(file_path)

    elif extension in [".png", ".jpg", ".jpeg"]:
        return extract_image(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")


# -----------------------------
# Extractors
# -----------------------------

def extract_pdf(file_path):
    doc = fitz.open(file_path)
    pages = []

    for i, page in enumerate(doc):
        text = page.get_text()
        pages.append({"page": i + 1, "text": text})

    return pages


def extract_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    return [{"page": 1, "text": text}]


def extract_docx(file_path):
    document = docx.Document(file_path)
    text = "\n".join([para.text for para in document.paragraphs])

    return [{"page": 1, "text": text}]


def extract_csv(file_path):
    df = pd.read_csv(file_path)
    text = df.to_string()

    return [{"page": 1, "text": text}]


def extract_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    text = json.dumps(data, indent=2)

    return [{"page": 1, "text": text}]


def extract_html(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "lxml")

    text = soup.get_text()
    return [{"page": 1, "text": text}]


def extract_image(file_path):
    image = Image.open(file_path)
    text = pytesseract.image_to_string(image)

    return [{"page": 1, "text": text}]
