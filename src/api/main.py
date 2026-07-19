from __future__ import annotations
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis import Redis
from redis.exceptions import RedisError

from src.chatbot import ask_question

from src.config import REDIS_URL, CONVERSATION_TTL_SECONDS
from src.conversation_store import Conversation, RedisConversationStore
from src.redis_memory import RedisConversationMemory


redis_client = Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)
conversation_store = RedisConversationStore(
    redis_client=redis_client,
    ttl_seconds=CONVERSATION_TTL_SECONDS,
)

app = FastAPI(
    title="Production RAG AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

class SourceResponse(BaseModel):
    source: str
    page: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    conversation_id: str


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    user_id: UUID
    conversation_id: str | None = None


@app.get("/")
def root() -> dict[str, str]:
    """Provide a useful response for browser visits to the API base URL."""

    return {
        "service": "Production RAG AI",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    try:
        redis_client.ping()
    except RedisError as error:
        raise HTTPException(
            status_code=503,
            detail="Redis is unavailable.",
        ) from error

    return {
        "status": "healthy",
        "redis": "connected",
    }


@app.get("/v1/conversations", response_model=list[ConversationResponse])
def list_conversations(user_id: UUID) -> list[Conversation]:
    """List recent conversations owned by one anonymous browser identifier."""

    return conversation_store.list_for_user(str(user_id))


@app.get(
    "/v1/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def get_conversation_messages(
    conversation_id: str,
    user_id: UUID,
) -> list[dict[str, str]]:
    """Load the saved messages for a browser-owned conversation."""

    if not conversation_store.belongs_to_user(
        user_id=str(user_id),
        conversation_id=conversation_id,
    ):
        raise HTTPException(status_code=404, detail="Conversation not found.")

    memory = RedisConversationMemory(
        redis_client=redis_client,
        conversation_id=conversation_id,
        max_turns=4,
        ttl_seconds=CONVERSATION_TTL_SECONDS,
    )
    return memory.get_messages()


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())
    user_id = str(request.user_id)

    if request.conversation_id:
        if not conversation_store.belongs_to_user(
            user_id=user_id,
            conversation_id=conversation_id,
        ):
            raise HTTPException(status_code=404, detail="Conversation not found.")
    else:
        conversation_store.create(
            user_id=user_id,
            conversation_id=conversation_id,
            title=request.question,
        )

    memory = RedisConversationMemory(
        redis_client=redis_client,
        conversation_id=conversation_id,
        max_turns=4,
        ttl_seconds=CONVERSATION_TTL_SECONDS,
    )

    try:
        result = await run_in_threadpool(
            ask_question,
            request.question,
            memory,
        )
    except RedisError as error:
        raise HTTPException(
            status_code=503,
            detail="Conversation memory is temporarily unavailable.",
        ) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="The RAG pipeline could not complete the request.",
        ) from error

    conversation_store.touch(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conversation_id,
    )
