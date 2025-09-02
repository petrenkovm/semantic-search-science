import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import re
import pandas as pd
import altair as alt
import fitz  # PyMuPDF

from scripts.search import SemanticSearcher
from scripts.build_index import build_faiss_index

# 📦 Константы
AVAILABLE_MODELS = {
    "MiniLM (быстро)": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet (точно)": "sentence-transformers/all-mpnet-base-v2",
    "Multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}
DATA_DIR = "data"

# ✨ Подсветка ключевых слов
def highlight(text: str, query: str):
    keywords = re.findall(r'\w+', query.lower())
    for kw in keywords:
        text = re.sub(f"(?i)({kw})", r"**\1**", text)
    return text

# 📄 Извлечение текста из PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# 📊 Проверка качества текста
def check_text_quality(text: str) -> dict:
    words = re.findall(r'\w+', text)
    return {
        "length": len(text),
        "lines": text.count("\n"),
        "unique_words": len(set(words)),
        "avg_line_length": round(len(text) / (text.count("\n") + 1), 2),
        "non_ascii": any(ord(c) > 127 for c in text),
        "symbols": any(c in "!@#$%^&*()[]{}<>?" for c in text)
    }

# 🎯 Основной интерфейс
def main():
    st.set_page_config(page_title="Semantic Search", layout="wide")
    st.title("🔍 Семантический поиск по научным текстам")

    # 📌 Выбор модели
    model_label = st.selectbox("Выберите модель эмбеддингов:", list(AVAILABLE_MODELS.keys()))
    model_name = AVAILABLE_MODELS[model_label]

    # 📤 Загрузка новых документов
    st.subheader("📤 Загрузка документов (.txt или .pdf)")
    uploaded_file = st.file_uploader("Выберите файл", type=["txt", "pdf"])
    if uploaded_file:
        file_path = os.path.join(DATA_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"✅ Файл сохранён: {uploaded_file.name}")

        if uploaded_file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
            txt_path = file_path.replace(".pdf", ".txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            st.info(f"📄 PDF конвертирован в: {os.path.basename(txt_path)}")
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

        quality = check_text_quality(text)
        st.subheader("📊 Качество текста")
        st.json(quality)

        if st.button("🔄 Переиндексировать"):
            with st.spinner("Обновляем индекс..."):
                build_faiss_index(model_name=model_name, data_dir="data", index_type="ip")
            st.success("✅ Индекс обновлён!")

    # 🕘 История запросов
    if "history" not in st.session_state:
        st.session_state.history = []

    with st.sidebar:
        st.subheader("🕘 История запросов")
        for q in reversed(st.session_state.history[-10:]):
            if st.button(q):
                st.session_state.query = q

    # 🔎 Ввод запроса
    query = st.text_input("Введите поисковый запрос:", value=st.session_state.get("query", ""))
    top_k = st.slider("Сколько результатов показать:", min_value=1, max_value=10, value=3)
    score_threshold = st.slider("Минимальный порог score:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    
    if query:
        st.session_state.history.append(query)

        with st.spinner("Ищем..."):
            searcher = SemanticSearcher(model_name=model_name, top_k=top_k)
            results = searcher.search(query)
            similarity_results = [
                                    (result["text"], result["score"])
                                    for result in results
                                    if result["score"] >= score_threshold
                                ]
            similarity_results.sort(key=lambda x: x[1], reverse=True)

        st.markdown(f"**Используемая модель:** `{model_name}`")
        st.subheader("📄 Результаты:")

        if similarity_results:
            max_score = max(score for _, score in similarity_results)

            for i, (doc, score) in enumerate(similarity_results, 1):
                st.markdown(f"**#{i}** — score: `{score:.4f}`")
                st.write(highlight(doc[:500], query) + ("..." if len(doc) > 500 else ""))
                st.progress(round(float(score / max_score), 4))
                st.divider()

            # 📊 Визуализация score
            st.subheader("📊 Score-график:")
            df = pd.DataFrame(similarity_results, columns=["doc", "score"])
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X("score:Q", title="Score"),
                y=alt.Y("doc:N", sort="-x", title="Документ"),
                tooltip=["score"]
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Ничего не найдено по вашему запросу с заданным порогом.")

# 🚀 Запуск
if __name__ == "__main__":
    main()
