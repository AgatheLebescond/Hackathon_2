# Fine-Tuned LLM for Sentiment Analysis and Contextual Responses

This project demonstrates how to fine‑tune a language model using LoRA for sentiment classification and then generate context‑aware responses using retrieval. A small Streamlit UI exposes the pipeline.

## Setup

1. Install the required packages (internet connection needed):
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (runs a small epoch by default):
   ```bash
   python -m src.train
   ```

3. Build the retrieval index and launch the UI:
   ```bash
   streamlit run src/app.py
   ```

Training downloads the IMDB dataset and subsamples 8k/2k examples. After training, a FAISS index is built from the training texts. The Streamlit app classifies user input, retrieves similar reviews, and generates a response with the fine‑tuned model. User feedback is logged to `feedback.log` for future improvements.
