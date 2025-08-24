import os

DATA_DIR = "data"
OUTPUT_FILE = "data/cleaned_docs.txt"

def collect_documents(data_dir):
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.startswith("doc") and fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    docs.append(content)
    return docs

def save_cleaned_docs(docs, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc + "\n")

if __name__ == "__main__":
    docs = collect_documents(DATA_DIR)
    save_cleaned_docs(docs, OUTPUT_FILE)
    print(f"✅ Загружено {len(docs)} документов в {OUTPUT_FILE}")
