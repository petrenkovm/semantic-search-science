import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import re
import pandas as pd
import altair as alt
from scripts.search import SemanticSearcher

# 📦 Константы
AVAILABLE_MODELS = {
    "MiniLM (быстро)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (точно)": "sentence-transformers/all-mpnet-base-v2",
    "Multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
INDEX_PATH = "embeddings/faiss_index.index"
DOCS_PATH = "data/cleaned_docs.txt"

# ✨ Подсветка ключевых слов
def highlight(text: str, query: str):
    keywords = re.findall(r'\w+', query.lower())
    for kw in keywords:
        text = re.sub(f"(?i)({kw})", r"**\1**", text)
    return text

# 🎯 Основной интерфейс
def main():
    st.set_page_config(page_title="Semantic Search", layout="wide")
    st.title("🔍 Семантический поиск по научным текстам")

    # История запросов
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.subheader("🕘 История запросов")
        for q in reversed(st.session_state.history[-10:]):
            if st.button(q):
                st.session_state.query = q

    # Выбор модели
    model_label = st.selectbox("Выберите модель эмбеддингов:", list(AVAILABLE_MODELS.keys()))
    model_name = AVAILABLE_MODELS[model_label]

    # Ввод запроса
    query = st.text_input("Введите поисковый запрос:", value=st.session_state.get("query", ""))
    top_k = st.slider("Сколько результатов показать:", min_value=1, max_value=10, value=3)

    if query:
        st.session_state.history.append(query)

        with st.spinner("Ищем..."):
            searcher = SemanticSearcher(model_name=model_name, top_k=top_k)
            results = searcher.search(query)

        st.markdown(f"**Используемая модель:** `{model_name}`")
        st.subheader("📄 Результаты:")

        if results:
            max_score = max(score for _, score in results)

            for i, (doc, score) in enumerate(results, 1):
                st.markdown(f"**#{i}** — score: `{score:.4f}`")
                st.write(highlight(doc[:500], query) + ("..." if len(doc) > 500 else ""))
                st.progress(round(float(score / max_score), 4))
                st.divider()

            # 📊 Визуализация score
            st.subheader("📊 Score-график:")
            df = pd.DataFrame(results, columns=["doc", "score"])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("score:Q", title="Score"),
                y=alt.Y("doc:N", sort="-x", title="Документ"),
                tooltip=["score"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.warning("Ничего не найдено по вашему запросу.")

# 🚀 Запуск
if __name__ == "__main__":
    main()
