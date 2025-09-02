# Semantic Search Science 🔍

**Semantic Search Science** — это open-source инструмент для семантического поиска по научным текстам. Он использует эмбеддинги Sentence Transformers и FAISS для быстрого и точного поиска по смыслу, а не по ключевым словам.

## 🚀 Возможности

- Поддержка `.txt` и `.pdf` документов
- Автоматическая конвертация PDF → TXT
- Построение FAISS-индекса с косинусным сходством
- Выбор модели эмбеддингов (MiniLM, MPNet, Multilingual)
- Streamlit-интерфейс с визуализацией результатов
- Порог отображения и сортировка по релевантности

## 🧰 Стек технологий

- Python 3.12+
- Sentence Transformers
- FAISS
- Streamlit
- PyMuPDF
- Altair

## 📂 Структура проекта
```plaintext
semantic-search-science/
├── app/
│   └── interface.py
├── scripts/
│   ├── build_index.py
│   ├── search.py
│   └── utils.py
├── data/
│   └── *.txt / *.pdf
├── requirements.txt
├── Makefile
├── README.md
└── assets/
    └── logo.png


## ⚙️ Установка
```bash
git clone https://github.com/petrenkovm/semantic-search-science.git
cd semantic-search-science
python -m venv .venv
source .venv/bin/activate    # или .venv\Scripts\activate в Windows
pip install -r requirements.txt

## ⚙️ Построение индекса
make build-index MODEL=sentence-transformers/all-mpnet-base-v2

Можно заменить модель на любую совместимую из Hugging Face.

📄 Обработка PDF и проверка качества
make process-pdf         # Конвертация всех PDF → TXT
make check-quality       # Проверка длины, символов, структуры
make reindex             # Переиндексация после изменений


🧪 Запуск интерфейса
make build-index MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
streamlit run app/interface.py

📸 Интерфейс

🧪 Пример запроса
Запрос: "Цель проекта" Результаты:

Документ с описанием цели проекта — score: 0.917

Технический фрагмент — score: 0.936

Генеративные модели — score: 0.717

🧠 Автор
Разработано Виталием — архитектором ML/AI-процессов, практиком MLOps и Python-разработчиком.

---

## 🎥 Демо-описание

> Streamlit-интерфейс позволяет загружать документы, переиндексировать их и выполнять семантический поиск. Результаты отображаются с подсветкой ключевых слов, прогресс-барами и графиком сходства. Поддерживается мультиязычный поиск.

---