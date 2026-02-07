from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text+= page.extract_text() + '\n'
    return text

def clean_text(text):
    cleaned_text = text.replace('\n', ' ')
    return cleaned_text

def chunk_text(text, chunk_size = 300, overlap = 50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def create_embeddings(chunks, embedding_model):
    embeddings = embedding_model.encode(chunks)
    return embeddings


def create_faiss_index(embeddings):
    embeddings = embeddings.astype('float32')
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)
    return index

def search_similar_chunks(query, chunks, embedding_model, index, k = 3):
    query_emb = embedding_model.encode([query])

    query_emb = query_emb.astype('float32')

    distances, indices = index.search(query_emb, k = k)

    retreived_chunks = [chunks[idx] for idx in indices[0]]

    return retreived_chunks, indices[0], distances[0]

def generate_prompt(query, context):
    prompt = f"""Answer the question using the information from the context below. You can summarize, synthesize, and infer from the context to provide a helpful answer. Only say you don't have enough information if the context is truly irrelevant to the question.

    Context: {context}

    Question: {query}

    Answer:"""
    
    return prompt