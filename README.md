# Hackathon 2 - Sentiment Analysis with Contextual Responses

This project demonstrates a small-scale pipeline that fine-tunes a language model using LoRA for sentiment classification and augments responses with a retrieval component. A simple Streamlit UI allows you to test the system end-to-end.

## Setup

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train the LoRA sentiment model (optional if you have a checkpoint)

```bash
python scripts/train_lora_sentiment.py
```

This script loads the IMDB dataset, applies a LoRA adapter on top of a distilled BERT model and trains for one epoch.

3. Run the Streamlit app

```bash
streamlit run app.py
```

Upload a text file in the sidebar to build the retrieval index. Enter a sentence to classify its sentiment and generate a context-aware answer.
