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
            # Fetch the content from the URL
            res = requests.get(url.strip())
            res.raise_for_status()  # Ensure we handle any HTTP errors
            soup = BeautifulSoup(res.text, "html.parser")
            
            # Extract all the text from paragraph tags
            text = " ".join([p.get_text() for p in soup.find_all("p")])
            st.session_state.docs.append(text)
        except requests.exceptions.RequestException as e:
            st.warning(f"Error fetching {url.strip()}: {e}")
        except Exception as e:
            st.warning(f"Error processing {url.strip()}: {e}")

    if st.session_state.docs:
        # Embed the content into vectors
        st.session_state.embeddings = embedder.encode(st.session_state.docs)
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(st.session_state.embeddings.shape[1])
        index.add(np.array(st.session_state.embeddings))
        st.session_state.index = index
        st.success("✅ Content ingested and indexed!")

# Ask questions
question = st.text_input("Ask a question:")

if st.button("Get Answer") and "docs" in st.session_state:
    q_vec = embedder.encode([question])
    
    # Search the FAISS index for the most relevant document
    D, I = st.session_state.index.search(np.array(q_vec), k=3)
    
    # Combine the content of the top documents to form context for the question
    context = " ".join([st.session_state.docs[i] for i in I[0]])
    
    # Use the QA model to find an answer based on the context
    result = qa({'context': context, 'question': question})
    
    # Display the result
    st.markdown(f"**Answer:** {result['answer']}")
