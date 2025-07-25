import streamlit as st
from generate import SentimentResponder


@st.cache_resource
def load_responder():
    responder = SentimentResponder()
    responder.build_retrieval()
    return responder


responder = load_responder()

st.title("Sentiment Analysis with Contextual Response")

user_input = st.text_area("Enter text", "")
if st.button("Analyze") and user_input:
    with st.spinner("Generating..."):
        reply, sentiment = responder.respond(user_input)
    st.markdown(f"**Sentiment:** {sentiment}")
    st.markdown(f"**Response:** {reply}")

feedback = st.radio("Was this response helpful?", ("Yes", "No"))
if st.button("Submit Feedback"):
    with open("feedback.log", "a") as f:
        f.write(f"{user_input}\t{sentiment}\t{feedback}\n")
    st.success("Thanks for your feedback!")
