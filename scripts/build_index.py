import os
import pickle
import argparse
from sentence_transformers import SentenceTransformer
import faiss
from scripts.utils import load_documents

def build_faiss_index(model_name: str, top_k: int = 5):
    print(f"[INFO] Загружаем модель: {model_name}")
    model = SentenceTransformer(model_name)

    print("[INFO] Загружаем документы...")
    documents = load_documents()
    texts = [doc["text"] for doc in documents]
    metadata = [doc["metadata"] for doc in documents]

    print("[INFO] Генерируем эмбеддинги...")
    embeddings = model.encode(texts, show_progress_bar=True)

    dim = embeddings.shape[1]
    print(f"[INFO] Размерность эмбеддингов: {dim}")

    print("[INFO] Создаём FAISS-индекс...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    print("[INFO] Сохраняем индекс и метаданные...")
    faiss.write_index(index, f"data/faiss_index_{dim}.index")
    with open(f"data/metadata_{dim}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("[SUCCESS] Индекс успешно создан и сохранён.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index with selected model")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Название модели эмбеддинга")
    parser.add_argument("--top_k", type=int, default=5, help="Количество ближайших соседей")
    args = parser.parse_args()

    build_faiss_index(model_name=args.model, top_k=args.top_k)
