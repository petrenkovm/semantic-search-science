# Semantic Search Science 🔍

**Semantic Search Science** — это воспроизводимый пайплайн для построения семантического поиска на основе эмбеддингов и FAISS-индекса.

## 🚀 Возможности
- 📄 Загрузка и обработка текстовых данных (`.txt`, `.md`)
- 🔎 Генерация эмбеддингов с помощью `sentence-transformers`
- ⚡ Быстрое построение FAISS-индекса
- 🛠 Простая конфигурация модели через параметры CLI
- 📦 Воспроизводимость сборки через `Makefile`

## 📂 Структура проекта
semantic-search-science/
├── data/                   # Исходные документы
├── scripts/
│   ├── utils.py             # Загрузка и подготовка данных
│   ├── build_index.py       # Создание FAISS-индекса
├── index.faiss              # (игнорируется в гите)
├── metadata.json            # (игнорируется в гите)
├── requirements.txt         # Зависимости
├── Makefile                 # Автоматизация команд
└── README.md


## ⚙️ Установка
```bash
git clone https://github.com/petrenkovm/semantic-search-science.git
cd semantic-search-science
python -m venv .venv
source .venv/bin/activate    # или .venv\Scripts\activate в Windows
pip install -r requirements.txt

## ⚙️ Построение индекса
make build-index MODEL=sentence-transformers/all-mpnet-base-v2

