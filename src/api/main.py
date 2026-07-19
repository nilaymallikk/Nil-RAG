from __future__ import annotations
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from redis import Redis
from redis.exceptions import RedisError

from src.chatbot import ask_question
from src.memory import ConversationMemory

from src.config import REDIS_URL, CONVERSATION_TTL_SECONDS
from src.redis_memory import RedisConversationMemory


redis_client = Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)

app = FastAPI(
    title="Producton RAG AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501/",
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

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    conversation_id: str | None = None


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


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    conversation_id = request.conversation_id or str(uuid4())

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

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conversation_id,
    )

