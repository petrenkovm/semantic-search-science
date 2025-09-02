import os
import pickle
import argparse
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from scripts.utils import load_documents

def build_faiss_index(model_name: str, data_dir: str = "data", index_type: str = "ip", top_k: int = 5):
    print(f"[INFO] Загружаем модель: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[INFO] Загружаем документы из: {data_dir}")
    documents = load_documents(data_dir=data_dir)
    documents = [doc for doc in documents if doc["text"].strip()]
    if not documents:
        print("[ERROR] Нет валидных документов для индексации.")
        return

    texts = [doc["text"] for doc in documents]
    metadata = [doc["metadata"] for doc in documents]

    print("[INFO] Генерируем эмбеддинги...")
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    print(f"[INFO] Размерность эмбеддингов: {dim}")

    print("[INFO] Нормализуем эмбеддинги для косинусного сходства...")
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    print(f"[INFO] Создаём FAISS-индекс ({index_type})...")
    if index_type == "ip":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, 32)
    elif index_type == "ivf":
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, 100)
        index.train(embeddings)
    else:
        raise ValueError(f"[ERROR] Неизвестный тип индекса: {index_type}")

    index.add(embeddings)

    print("[INFO] Сохраняем эмбеддинги...")
    np.save(f"{data_dir}/embeddings_{dim}.npy", embeddings)

    print("[INFO] Сохраняем индекс и метаданные...")
    faiss.write_index(index, f"{data_dir}/faiss_index_{dim}.index")
    with open(f"{data_dir}/metadata_{dim}.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("[SUCCESS] Индекс успешно создан и сохранён.")
    print(f"[INFO] Загружено документов: {len(documents)}")
    print(f"[INFO] Всего чанков: {len(texts)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index with cosine similarity")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-mpnet-base-v2",
                        help="Название модели эмбеддинга")
    parser.add_argument("--data_dir", type=str, default="data", help="Папка с документами")
    parser.add_argument("--top_k", type=int, default=5, help="Количество ближайших соседей")
    parser.add_argument("--index_type", type=str, default="ip", choices=["ip", "hnsw", "ivf"],
                        help="Тип FAISS-индекса (Inner Product)")
    args = parser.parse_args()

    build_faiss_index(
        model_name=args.model,
        data_dir=args.data_dir,
        index_type=args.index_type,
        top_k=args.top_k
    )
