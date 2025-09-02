# scripts/utils.py
from pathlib import Path

def load_documents(data_dir: str = "data"):
    """
    Читает все .txt и .md из папки root и возвращает список словарей:
    {
      "text": содержимое файла,
      "metadata": {"source": путь_к_файлу, "name": имя_файла}
    }
    """
    root_path = Path(data_dir)
    docs = []
    for path in root_path.rglob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8")
                docs.append({
                    "text": text,
                    "metadata": {
                        "source": str(path),
                        "name": path.stem
                    }
                })
            except Exception as e:
                print(f"[WARN] Не удалось прочитать {path}: {e}")
    return docs
