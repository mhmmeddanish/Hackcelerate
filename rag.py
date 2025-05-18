import streamlit as st
import os
import numpy as np
import re
from collections import Counter
import io

try:
    import PyPDF2
except ImportError:
    import subprocess
    subprocess.check_call(["pip", "install", "PyPDF2"])
    import PyPDF2

# ----------------------------------
# Simple RAG System Components
# ----------------------------------

class SimpleVectorizer:
    def __init__(self):
        self.vocabulary = set()
        self.vocab_to_idx = {}
        self.idf = None

    def _tokenize(self, text):
        return re.findall(r'\w+', text.lower())

    def fit(self, text_documents):
        all_tokens = []
        for doc in text_documents:
            tokens = self._tokenize(doc)
            all_tokens.extend(tokens)
            self.vocabulary.update(tokens)

        self.vocabulary = sorted(self.vocabulary)
        self.vocab_to_idx = {word: i for i, word in enumerate(self.vocabulary)}

        doc_freq = Counter()
        for doc in text_documents:
            tokens = set(self._tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        N = len(text_documents)
        self.idf = np.zeros(len(self.vocabulary))
        for word, idx in self.vocab_to_idx.items():
            df = doc_freq.get(word, 0) + 1
            self.idf[idx] = np.log(N / df)

    def transform(self, documents):
        doc_vectors = []
        for doc in documents:
            tokens = self._tokenize(doc)
            term_freq = Counter(tokens)
            vector = np.zeros(len(self.vocabulary))
            for token, count in term_freq.items():
                if token in self.vocab_to_idx:
                    idx = self.vocab_to_idx[token]
                    vector[idx] = count * self.idf[idx]
            doc_vectors.append(vector)
        return np.array(doc_vectors)

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0

# ----------------------------------
# State Management
# ----------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []
    st.session_state.document_sources = []
    st.session_state.vectorizer = None
    st.session_state.doc_vectors = None

# ----------------------------------
# PDF Processing
# ----------------------------------

def process_pdf_buffer(file_buffer, filename="uploaded_document.pdf"):
    try:
        pdf_reader = PyPDF2.PdfReader(file_buffer)
        page_texts = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
        chunks = []
        for text in page_texts:
            for para in text.split('\n\n'):
                if len(para.strip()) > 50:
                    chunks.append(para.strip())
        for chunk in chunks:
            st.session_state.documents.append(chunk)
            st.session_state.document_sources.append(filename)
        return f"✅ Processed {filename} with {len(chunks)} chunks."
    except Exception as e:
        return f"❌ Error: {e}"

def update_vectors():
    if not st.session_state.documents:
        return False
    st.session_state.vectorizer = SimpleVectorizer()
    st.session_state.doc_vectors = st.session_state.vectorizer.fit_transform(st.session_state.documents)
    return True

def query_question(user_question, top_k=3):
    if not st.session_state.documents:
        return "No documents uploaded yet."
    if st.session_state.vectorizer is None or st.session_state.doc_vectors is None:
        if not update_vectors():
            return "Error updating vectors."

    query_vector = st.session_state.vectorizer.transform([user_question])[0]
    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in st.session_state.doc_vectors]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    top_docs = [(st.session_state.documents[i], similarities[i], st.session_state.document_sources[i]) for i in top_indices]

    result = "### 🔍 Top Matching Chunks:\n"
    for i, (doc, score, source) in enumerate(top_docs):
        if score < 0.01:
            continue
        snippet = doc[:300] + "..." if len(doc) > 300 else doc
        result += f"**[{i+1}]** *From {source}* (score: `{score:.2f}`)\n\n```\n{snippet}\n```\n\n"
    return result.strip() or "No relevant information found."

# ----------------------------------
# Streamlit UI
# ----------------------------------

st.set_page_config(page_title="PDF RAG Query", layout="centered")
st.title("📄 RAG PDF Query System")

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file:
        msg = process_pdf_buffer(uploaded_file, uploaded_file.name)
        update_vectors()
        st.success(msg)

# Show number of chunks
st.markdown(f"**Total Text Chunks Stored:** `{len(st.session_state.documents)}`")

# Question Box
question = st.text_input("💬 Ask a question based on uploaded PDFs")

if st.button("Get Answer") and question:
    with st.spinner("Thinking..."):
        answer = query_question(question)
        st.markdown(answer)
