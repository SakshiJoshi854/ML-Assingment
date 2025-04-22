# 📄 Save this as app.py and run with: streamlit run app.py

import streamlit as st
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load models
embedder = SentenceTransformer('all-MiniLM-L6-v2')
qa = pipeline("question-answering")

# Streamlit UI
st.set_page_config(page_title="Web Q&A Tool")
st.title("🧠 Web Content Q&A Tool")

# Get URLs from user
urls = st.text_area("Enter webpage URLs (comma-separated):")

if st.button("Ingest Content"):
    st.session_state.docs = []
    for url in urls.split(","):
        try:
            res = requests.get(url.strip())
            soup = BeautifulSoup(res.text, "html.parser")
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            st.session_state.docs.append(text)
        except Exception as e:
            st.warning(f"Error reading {url.strip()}: {e}")

    if st.session_state.docs:
        st.session_state.embeddings = embedder.encode(st.session_state.docs)
        index = faiss.IndexFlatL2(st.session_state.embeddings.shape[1])
        index.add(np.array(st.session_state.embeddings))
        st.session_state.index = index
        st.success("✅ Content ingested and indexed!")

# Ask questions
question = st.text_input("Ask a question:")

if st.button("Get Answer") and "docs" in st.session_state:
    q_vec = embedder.encode([question])
    D, I = st.session_state.index.search(np.array(q_vec), k=3)
    context = " ".join([st.session_state.docs[i] for i in I[0]])
    result = qa({'context': context, 'question': question})
    st.markdown(f"**Answer:** {result['answer']}")
