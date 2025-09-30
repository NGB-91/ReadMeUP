import os
import ast
import faiss
import numpy as np
import pandas as pd
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from app.utils.language import lang_to_iso, iso_to_display  

# Paths and model setup
DATA_PATH = "data/final/books_translated.csv"
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"
IDX_PATH = "data/final/faiss_index.idx"
EMB_PATH = "data/final/embeddings.npy"

@lru_cache(maxsize=1)
def load_dataset():
    """Load and preprocess the translated books dataset."""
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=['Unnamed: 0'], errors='ignore')

    # Parse genres if stored as stringified lists
    if isinstance(df['genres'].iloc[0], str) and df['genres'].iloc[0].startswith("["):
        df['genres'] = df['genres'].apply(ast.literal_eval)

    # Normalize author field
    df['author'] = df['author'].apply(
        lambda x: [a.strip() for a in x.split(',')] if isinstance(x, str)
        else (x if isinstance(x, list) else [])
    )

    # Clean and normalize text fields
    df['description'] = df['description'].fillna("").astype(str)
    df['language'] = df['language'].fillna("").astype(str)
    df['language_code'] = df['language'].apply(lang_to_iso)
    df['language'] = df['language_code'].apply(iso_to_display)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce').fillna(0.0)
    df['genres'] = df['genres'].apply(lambda g: g if isinstance(g, list) else [])

    # Create embedding input text
    df['text_for_embedding'] = df['description'].str.strip() + ". Genres: " + df['genres'].apply(lambda g: ", ".join(g))
    return df

@lru_cache(maxsize=1)
def load_embedder():
    """Load the sentence transformer model from local path."""
    return SentenceTransformer(MODEL_PATH)

@lru_cache(maxsize=1)
def load_faiss_artifacts():
    """Load FAISS index and embedding matrix if available."""
    if os.path.exists(IDX_PATH) and os.path.exists(EMB_PATH):
        return faiss.read_index(IDX_PATH), np.load(EMB_PATH)
    return None, None

def semantic_recommend(query, df, embedder, faiss_index, top_n=5, language=None, min_rating=0.0):
    """Return top semantic recommendations based on query and filters, with variation on each call."""
    if not query or not query.strip():
        return None, "Please enter a description or idea for the book you want."

    # Encode query and normalize vector
    q_vec = embedder.encode([query], convert_to_numpy=True).astype('float32')
    q_vec /= np.clip(np.linalg.norm(q_vec, axis=1, keepdims=True), 1e-12, None)

    # Search FAISS index (use a larger pool to allow variation)
    pool_n = max(top_n * 5, top_n + 10)
    sims, idxs = faiss_index.search(q_vec, pool_n)
    candidates = df.iloc[idxs[0]].copy()
    candidates['similarity'] = sims[0]

    # Apply filters
    if language:
        lang_str = str(language).strip()
        candidates = candidates[
            candidates['language_code'].str.lower().eq(lang_str.lower()) |
            candidates['language'].str.casefold().eq(lang_str.casefold())
        ]
    if min_rating and min_rating > 0:
        candidates = candidates[candidates['rating'] >= float(min_rating)]

    # Sample to introduce variation across calls while keeping relevance
    if len(candidates) >= top_n:
        results = candidates.sample(n=top_n, replace=False)
    else:
        results = candidates.head(top_n)

    if results.empty:
        return None, "No results found. Try broadening your query or relaxing filters."
    return results, f"Semantic recommendations for: **{query.strip()}**"
