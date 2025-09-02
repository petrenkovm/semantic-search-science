# 🚀 Швидкий старт

Semantic Search Science — це система семантичного пошуку по наукових текстах з використанням ембеддингів та FAISS.

## 🔧 Встановлення

```bash
git clone https://github.com/petrenkovm/semantic-search-science.git
cd semantic-search-science
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

⚙️ Побудова індексу

make build-index MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

▶️ Запуск інтерфейсу

streamlit run app/interface.py

Відкриється браузер за адресою http://localhost:8501


---

### 📄 `docs/architecture.md`

```markdown
# 🏗️ Архітектура проєкту

## Загальна схема

Документи (.txt/.pdf) → Обробка → Ембеддинги → FAISS → Пошук → Інтерфейс


## Компоненти

- `build_index.py`: генерація ембеддингів та створення індексу
- `search.py`: пошук по запиту
- `interface.py`: Streamlit-інтерфейс
- `utils.py`: обробка документів
- `data/`: збереження текстів та індексу

