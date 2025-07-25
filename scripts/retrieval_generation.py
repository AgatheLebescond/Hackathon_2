import os
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.docs = []

    def build_index(self, texts):
        self.docs = list(texts)
        embeddings = self.embedder.encode(self.docs, convert_to_numpy=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def query(self, text, top_k=3):
        embedding = self.embedder.encode([text], convert_to_numpy=True)
        scores, indices = self.index.search(embedding, top_k)
        return [self.docs[i] for i in indices[0]]


def generate_response(model, tokenizer, user_query, context_texts):
    context = "\n".join(context_texts)
    prompt = f"Context:\n{context}\n\nUser: {user_query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example usage
    docs = ["The movie was great!", "I did not like the food", "Amazing performance by the actor."]
    retriever = Retriever()
    retriever.build_index(docs)
    context = retriever.query("What did people think of the movie?", top_k=2)

    model_name = os.environ.get("GEN_MODEL", "google/flan-t5-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    response = generate_response(model, tokenizer, "What was the sentiment?", context)
    print(response)
