from typing import List
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.emb_model = SentenceTransformer(model_name)
        self.index = None
        self.corpus = []

    def build_index(self, texts: List[str]):
        self.corpus = texts
        embeddings = self.emb_model.encode(texts, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        query_emb = self.emb_model.encode([query])
        faiss.normalize_L2(query_emb)
        scores, idx = self.index.search(query_emb, k)
        return [self.corpus[i] for i in idx[0]]
