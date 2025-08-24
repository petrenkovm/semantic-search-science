from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INPUT_PATH = "data/cleaned_docs.txt"
OUTPUT_PATH = "embeddings/doc_embeddings.npy"

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def generate_embeddings(docs, model_name):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(docs, show_progress_bar=True)
    return embeddings

def save_embeddings(embeddings, path):
    np.save(path, embeddings)

if __name__ == "__main__":
    docs = load_documents(INPUT_PATH)
    print(f"📄 Загружено {len(docs)} документов.")
    embeddings = generate_embeddings(docs, MODEL_NAME)
    save_embeddings(embeddings, OUTPUT_PATH)
    print(f"✅ Эмбеддинги сохранены в {OUTPUT_PATH}")
