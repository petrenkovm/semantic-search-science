import os
import pickle
from sentence_transformers import SentenceTransformer
import faiss

class SemanticSearcher:
    def __init__(self, model_name: str, top_k: int = 5):
        self.model_name = model_name
        self.top_k = top_k

        print(f"[INFO] Загружаем модель: {model_name}")
        self.model = SentenceTransformer(model_name)

        # Определяем размерность эмбеддингов
        dummy_embedding = self.model.encode(["тестовое предложение"])
        self.dim = dummy_embedding.shape[1]
        print(f"[INFO] Размерность модели: {self.dim}")

        # Загружаем соответствующий индекс
        index_path = f"data/faiss_index_{self.dim}.index"
        metadata_path = f"data/metadata_{self.dim}.pkl"

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(f"[ERROR] Индекс или метаданные не найдены для размерности {self.dim}. "
                                    f"Сначала запусти build_index.py с этой моделью.")

        print(f"[INFO] Загружаем FAISS-индекс: {index_path}")
        self.index = faiss.read_index(index_path)

        print(f"[INFO] Загружаем метаданные: {metadata_path}")
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        self.texts = []
        for meta in self.metadata:
            try:
                with open(meta["source"], "r", encoding="utf-8") as f:
                    self.texts.append(f.read())
            except Exception as e:
                print(f"[WARN] Не удалось прочитать {meta['source']}: {e}")
                self.texts.append("")


    def search(self, query: str):
        print(f"[INFO] Поиск запроса: {query}")
        query_embedding = self.model.encode([query])
        if query_embedding.shape[1] != self.dim:
            raise ValueError(f"[ERROR] Размерность запроса ({query_embedding.shape[1]}) не совпадает с индексом ({self.dim})")

        distances, indices = self.index.search(query_embedding, self.top_k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            results.append({
                "text": self.texts[idx],
                "score": float(score),
                "metadata": self.metadata[idx]
            })
            results
            print(f"[DEBUG] idx={idx}, score={score}")
            print(f"[DEBUG] text={self.texts[idx][:100]}")
            
        return results