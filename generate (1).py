from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from retrieval import Retriever
from data_utils import load_imdb


class SentimentResponder:
    def __init__(self, model_path: str = "fine_tuned_model"):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.retriever = Retriever()

    def classify(self, text: str) -> str:
        prompt = f"Review: {text}\nSentiment:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=5)
        sentiment = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return sentiment.strip()

    def build_retrieval(self):
        dataset = load_imdb(train_size=8000, val_size=2000)[0]
        self.retriever.build_index([ex["text"] for ex in dataset])

    def respond(self, text: str, k: int = 3) -> str:
        sentiment = self.classify(text)
        context = self.retriever.retrieve(text, k)
        context_text = "\n".join(context)
        prompt = (
            f"Sentiment: {sentiment}\n"
            f"Context: {context_text}\n"
            f"User: {text}\nResponse:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=128)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip(), sentiment
