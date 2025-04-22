# ML-Assingment

This is an online service that enables users to feed in URLs, ingest webpage material, and pose questions purely based on this material. It employs embeddings and vector search to identify appropriate information and compute correct answers utilizing a language model.

Features
1. Enter one or more URLs of webpages

2. Scrape and cache content of every URL

3. Convert text to vector embeddings for searching

4. Pose questions on the basis of ingested content alone

5. Clean, easy-to-use Streamlit UI

Tech Stack
Streamlit – Web UI

BeautifulSoup – HTML Parsing

Sentence Transformers – Text embeddings

FAISS – Vector similarity search

Transformers (Hugging Face) – Q&A model
