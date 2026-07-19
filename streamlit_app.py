from __future__ import annotations

import streamlit as st

from src.chatbot import ask_question
from src.memory import ConversationMemory


st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("MLGPT: Machine Learning RAG Assistant")
st.caption("based on my professor's Machine Learning course handbook")


def initialize_session() -> None:
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory(max_turns=4)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def clear_conversation() -> None:
    st.session_state.memory.clear()
    st.session_state.chat_messages = []


initialize_session()

with st.sidebar:
    st.header("Controls")

    if st.button("Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()

    st.divider()
    st.subheader("System")
    st.write("Retrieval: Hybrid BM25 + Vector")
    st.write("Fusion: Reciprocal Rank Fusion")
    st.write("Reranking: CrossEncoder")


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    st.write(
                        f"• {source['source']} — "
                        f"page {source['page']} "
                        f"(score: {source['score']:.3f})"
                    )


question = st.chat_input("Ask a question about the document")

if question:
    st.session_state.chat_messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                result = ask_question(
                    question=question,
                    memory=st.session_state.memory,
                )
            except Exception as error:
                st.error("The assistant could not complete the request.")
                st.exception(error)
            else:
                st.markdown(result["answer"])

                if result["sources"]:
                    with st.expander("Sources"):
                        for source in result["sources"]:
                            st.write(
                                f"• {source['source']} — "
                                f"page {source['page']} "
                                f"(score: {source['score']:.3f})"
                            )

                st.session_state.chat_messages.append(
                    {
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"],
                    }
                )
    