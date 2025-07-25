import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from scripts.retrieval_generation import Retriever, generate_response
import torch

@st.cache_resource
def load_models():
    cls_model = AutoModelForSequenceClassification.from_pretrained("./lora_sentiment")
    cls_tokenizer = AutoTokenizer.from_pretrained("./lora_sentiment")

    gen_model_name = "google/flan-t5-base"
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name)
    gen_tokenizer = AutoTokenizer.from_pretrained(gen_model_name)

    retriever = Retriever()
    return cls_model, cls_tokenizer, gen_model, gen_tokenizer, retriever


cls_model, cls_tokenizer, gen_model, gen_tokenizer, retriever = load_models()

st.title("Sentiment Analysis with Contextual Responses")
text_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    inputs = cls_tokenizer(text_input, return_tensors="pt")
    with torch.no_grad():
        preds = cls_model(**inputs).logits
        label = preds.argmax(-1).item()
    sentiment = "positive" if label == 1 else "negative"
    st.write(f"Sentiment: {sentiment}")

    if retriever.index is None and st.session_state.get("docs"):
        retriever.build_index(st.session_state["docs"])

    if retriever.index:
        context = retriever.query(text_input, top_k=3)
        response = generate_response(gen_model, gen_tokenizer, text_input, context)
        st.write(response)
    else:
        st.write("No retrieval data loaded.")

st.sidebar.header("Corpus Management")
upload = st.sidebar.file_uploader("Upload corpus file", type="txt")
if upload:
    text = upload.read().decode("utf-8")
    docs = [line.strip() for line in text.splitlines() if line.strip()]
    st.session_state["docs"] = docs
    retriever.build_index(docs)
    st.sidebar.write(f"Loaded {len(docs)} documents")
