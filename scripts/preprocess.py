import os
import re

DATA_DIR = "data"
CLEANED_PATH = "data/cleaned_docs.txt"

def clean_text(text):
    # Удаление лишних пробелов, спецсимволов, HTML-тегов
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def load_and_clean_documents():
    documents = []
    for filename in os.listdir(DATA_DIR):
        filepath = os.path.join(DATA_DIR, filename)
        if filename.endswith(".txt"):
            with open(filepath, "r", encoding="utf-8") as f:
                raw = f.read()
                cleaned = clean_text(raw)
                documents.append(cleaned)
    return documents

def save_cleaned(documents):
    with open(CLEANED_PATH, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(doc + "\n")

if __name__ == "__main__":
    docs = load_and_clean_documents()
    save_cleaned(docs)
    print(f"✅ Обработано и сохранено {len(docs)} документов.")
