from __future__ import annotations

import os
from pathlib import Path
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


def inject_styles() -> None:
    """Apply presentation-only styling without changing chat behavior."""

    st.markdown(
        """
        <style>
        :root {
            --accent: #10a37f;
            --surface: rgba(255, 255, 255, 0.045);
            --border: rgba(255, 255, 255, 0.09);
            --muted: #9ca3af;
        }

        .block-container {
            max-width: 980px;
            padding-top: 2rem;
            padding-bottom: 7rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] .stButton > button {
            border-radius: 10px;
            text-align: left;
            min-height: 42px;
        }

        [data-testid="stChatMessage"] {
            background: transparent;
            border: 0;
            padding: 1.05rem 0;
        }

        [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
            font-size: 1.02rem;
            line-height: 1.75;
        }

        [data-testid="stChatMessage"] h1,
        [data-testid="stChatMessage"] h2,
        [data-testid="stChatMessage"] h3 {
            margin-top: 1.35rem;
            margin-bottom: 0.55rem;
            letter-spacing: -0.02em;
        }

        [data-testid="stChatMessage"] pre {
            border: 1px solid var(--border);
            border-radius: 12px;
        }

        [data-testid="stChatInput"] {
            max-width: 980px;
            margin: 0 auto;
        }

        .app-kicker {
            color: var(--accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .app-title {
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 750;
            letter-spacing: -0.045em;
            line-height: 1.08;
            margin: 0;
        }

        .app-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0.65rem 0 2rem;
        }

        .welcome-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            margin: 2.5rem 0 1.25rem;
            padding: 1.5rem 1.65rem;
        }

        .welcome-card h3 {
            margin: 0 0 0.4rem;
        }

        .welcome-card p {
            color: var(--muted);
            margin: 0;
        }

        .source-title {
            font-weight: 650;
            margin-bottom: 0.15rem;
        }

        .source-meta {
            color: var(--muted);
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
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

    unique_sources = {
        (source["source"], source["page"]): source
        for source in sources
    }

    with st.expander(f"View sources · {len(unique_sources)}"):
        for source in unique_sources.values():
            filename = Path(source["source"]).name
            with st.container(border=True):
                st.markdown(
                    f"<div class='source-title'>📄 {filename}</div>"
                    f"<div class='source-meta'>Page {source['page']} · "
                    f"retrieval score {source['score']:.3f}</div>",
                    unsafe_allow_html=True,
                )


def render_message(message: dict) -> None:
    avatar = "🧑" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="welcome-card">
            <h3>What would you like to learn?</h3>
            <p>Ask for an explanation, comparison, derivation, or summary from
            the indexed Machine Learning handbook.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    suggestions = [
        "What is an artificial neural network?",
        "Explain backpropagation step by step.",
        "Compare supervised and unsupervised learning.",
    ]
    columns = st.columns(len(suggestions))
    for column, suggestion in zip(columns, suggestions, strict=True):
        with column:
            if st.button(suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()


inject_styles()
initialize_session()

st.markdown(
    """
    <div class="app-kicker">Grounded AI assistant</div>
    <h1 class="app-title">Machine Learning, explained clearly.</h1>
    <p class="app-subtitle">Ask questions and get structured answers based on a very good handbook of my professor</p>
    """,
    unsafe_allow_html=True,
)

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
    st.caption("Anonymous history")


if not st.session_state.chat_messages:
    render_empty_state()

for message in st.session_state.chat_messages:
    render_message(message)


typed_question = st.chat_input("Message MLGPT…")
question = typed_question or st.session_state.pop("pending_question", None)

if question:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🤖"):
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
