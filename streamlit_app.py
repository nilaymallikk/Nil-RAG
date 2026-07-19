from __future__ import annotations

import os
from uuid import UUID, uuid4

import httpx
import streamlit as st


API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8001")
REQUEST_TIMEOUT_SECONDS = 120.0


st.set_page_config(
    page_title="MLGPT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_or_create_browser_id() -> str:
    """Persist an anonymous browser identifier in the page URL."""

    candidate = st.query_params.get("user_id")
    try:
        return str(UUID(candidate))
    except (TypeError, ValueError, AttributeError):
        user_id = str(uuid4())
        st.query_params["user_id"] = user_id
        return user_id


def request_api(method: str, path: str, **kwargs) -> dict | list:
    response = httpx.request(
        method=method,
        url=f"{API_URL}{path}",
        timeout=REQUEST_TIMEOUT_SECONDS,
        **kwargs,
    )
    response.raise_for_status()
    return response.json()


def initialize_session() -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = get_or_create_browser_id()
    if "active_conversation_id" not in st.session_state:
        st.session_state.active_conversation_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


def start_new_conversation() -> None:
    st.session_state.active_conversation_id = None
    st.session_state.chat_messages = []


def open_conversation(conversation_id: str) -> None:
    messages = request_api(
        "GET",
        f"/v1/conversations/{conversation_id}/messages",
        params={"user_id": st.session_state.user_id},
    )
    st.session_state.active_conversation_id = conversation_id
    st.session_state.chat_messages = messages


def get_conversations() -> list[dict]:
    return request_api(
        "GET",
        "/v1/conversations",
        params={"user_id": st.session_state.user_id},
    )


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return

    with st.expander("Sources"):
        for source in sources:
            st.write(
                f"• {source['source']} — page {source['page']} "
                f"(score: {source['score']:.3f})"
            )


initialize_session()

st.title("MLGPT: Machine Learning RAG Assistant")
st.caption("Answers grounded in the indexed Machine Learning course handbook.")

with st.sidebar:
    if st.button("＋ New chat", use_container_width=True, type="primary"):
        start_new_conversation()
        st.rerun()

    st.divider()
    st.subheader("Recent chats")

    try:
        conversations = get_conversations()
    except httpx.HTTPError:
        st.error("Cannot load chat history. Is FastAPI running?")
        conversations = []

    if not conversations:
        st.caption("No conversations yet.")

    for conversation in conversations:
        is_active = (
            conversation["conversation_id"]
            == st.session_state.active_conversation_id
        )
        if st.button(
            conversation["title"],
            key=f"conversation-{conversation['conversation_id']}",
            use_container_width=True,
            disabled=is_active,
        ):
            try:
                open_conversation(conversation["conversation_id"])
            except httpx.HTTPError:
                st.error("Could not open this conversation.")
            else:
                st.rerun()

    st.divider()
    st.caption("Anonymous browser history")


for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


question = st.chat_input("Ask a question about the document")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the document and generating an answer..."):
            try:
                result = request_api(
                    "POST",
                    "/v1/chat",
                    json={
                        "question": question,
                        "user_id": st.session_state.user_id,
                        "conversation_id": st.session_state.active_conversation_id,
                    },
                )
            except httpx.HTTPStatusError as error:
                st.error(f"The API returned {error.response.status_code}.")
            except httpx.HTTPError:
                st.error("Cannot reach the RAG API. Is FastAPI running?")
            else:
                st.session_state.active_conversation_id = result["conversation_id"]
                st.session_state.chat_messages.extend(
                    [
                        {"role": "user", "content": question},
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        },
                    ]
                )
                st.markdown(result["answer"])
                render_sources(result["sources"])
                st.rerun()
