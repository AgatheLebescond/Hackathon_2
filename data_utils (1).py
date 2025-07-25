import random
from typing import Tuple
from datasets import load_dataset
from transformers import AutoTokenizer


def load_imdb(train_size: int = 8000, val_size: int = 2000) -> Tuple[dict, dict, AutoTokenizer]:
    """Load and tokenize IMDB dataset with a subsample."""
    dataset = load_dataset("imdb")
    train_dataset = dataset["train"].shuffle(seed=42).select(range(train_size))
    val_dataset = dataset["test"].shuffle(seed=42).select(range(val_size))

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    def preprocess(example):
        text = example["text"]
        sentiment = "positive" if example["label"] == 1 else "negative"
        prompt = f"Review: {text}\nSentiment:"  # for seq2seq classification
        return {
            "input_ids": tokenizer(prompt, truncation=True).input_ids,
            "labels": tokenizer(sentiment).input_ids,
        }

    train_dataset = train_dataset.map(preprocess)
    val_dataset = val_dataset.map(preprocess)

    return train_dataset, val_dataset, tokenizer
