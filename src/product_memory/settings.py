from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql://product_memory:product_memory_local_only@db:5432/product_memory"
    knowledge_dir: Path = Path("/knowledge")
    scan_interval_seconds: float = 30.0
    supported_extensions: str = (
        ".txt,.md,.markdown,.rst,.log,.vtt,.srt,.pdf,.docx,.pptx,.xlsx,.msg,.eml,"
        ".png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.gif,"
        ".mp4,.mov,.m4v,.webm,.mkv,.m4a,.mp3,.wav"
    )

    chunk_size: int = Field(default=1800, ge=300)
    chunk_overlap: int = Field(default=250, ge=0)

    enable_ocr: bool = True
    ocr_languages: str = "eng+pol"
    ocr_max_images_per_document: int = Field(default=100, ge=1, le=2000)
    ocr_min_image_pixels: int = Field(default=10_000, ge=0)
    ocr_min_characters: int = Field(default=12, ge=1)
    ocr_min_words: int = Field(default=6, ge=1)
    ocr_min_word_confidence: int = Field(default=60, ge=0, le=100)
    ocr_timeout_seconds: float = Field(default=30.0, gt=0)
    # Keep the picture, not just the words in it, so an answer can hand back the screenshot.
    store_images: bool = True
    max_stored_image_bytes: int = Field(default=5_000_000, ge=1000)
    # Where a caller can reach this server, so returned images carry a link that works.
    public_base_url: str = "http://localhost:2600"

    # Recordings usually carry speech that exists nowhere else, so they are transcribed rather
    # than skipped. Whisper is confident even when wrong, so a recording in a language this index
    # does not hold is refused rather than turned into plausible nonsense.
    enable_transcription: bool = True
    transcription_model: str = "small"
    transcription_languages: str = "en"
    transcription_threads: int = Field(default=8, ge=1)
    # Greedy decoding. Wider beams cost proportionally more for a corpus this size.
    transcription_beam_size: int = Field(default=1, ge=1, le=10)
    # Audio is decoded a window at a time so peak memory follows the window, not the meeting.
    transcription_window_seconds: int = Field(default=600, ge=60, le=3600)
    transcription_timeout_seconds: float = Field(default=3600.0, gt=0)
    # One scan should not be spent entirely on recordings. Whatever is left over is picked up by
    # the next pass, so ordinary documents keep being indexed in between.
    transcription_per_scan_limit: int = Field(default=1, ge=1, le=100)
    # A shared screen is often the only record of what a meeting was looking at, so the moments it
    # changed are kept as pictures beside the words that were spoken over them.
    enable_video_frames: bool = True
    frame_scene_threshold: float = Field(default=0.25, gt=0, le=1)
    frame_max_interval_seconds: int = Field(default=120, ge=10)
    frame_max_per_recording: int = Field(default=200, ge=1, le=5000)
    # A gallery of faces reads as about ten words of names; a shared screen reads as many more.
    frame_min_words: int = Field(default=15, ge=1)
    frame_width: int = Field(default=1280, ge=320)
    frame_timeout_seconds: float = Field(default=3600.0, gt=0)

    embedding_provider: Literal["local_hf", "openai"] = "local_hf"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_revision: str | None = None
    embedding_dimensions: int | None = Field(default=None, ge=1)
    embedding_batch_size: int = Field(default=16, ge=1, le=256)
    # torch.set_num_threads is process-wide, so this and reranker_threads cannot differ in
    # practice: whichever model loads last wins. Measured here, throughput stopped improving
    # around 8 threads regardless of how many cores were available.
    local_embedding_threads: int = Field(default=4, ge=1)
    hf_home: Path = Path("/models/huggingface")

    # Guards the only outbound traffic this service makes on its own. Off means a cold model
    # cache fails loudly instead of quietly pulling gigabytes over the network.
    allow_model_download: bool = True

    openai_api_key: str | None = None
    openai_base_url: str | None = None

    semantic_weight: float = Field(default=0.72, ge=0)
    lexical_weight: float = Field(default=0.13, ge=0)
    recency_weight: float = Field(default=0.15, ge=0)
    recency_half_life_days: float = Field(default=180, gt=0)

    # The three signals are read on incompatible scales: cosine distance fills the top of its
    # range while a text rank sits near the bottom of its own, so adding them lets whichever
    # scale happens to be wider decide the order. Fusing positions instead of values keeps each
    # signal's opinion and none of its units. A large k keeps the fusion gentle, so a single
    # signal that is merely confident cannot overrule the other two that disagree.
    rrf_k: float = Field(default=150, gt=0)

    # Gates on meaning alone. Blending recency in here used to hide old documents that still answer
    # the question, which is the wrong call for signed contracts.
    min_semantic_score: float = Field(default=0.60, ge=0, le=1)
    max_returned_documents: int = Field(default=25, ge=1, le=25)
    default_top_k_chunks: int = Field(default=10, ge=1, le=50)
    default_top_k_documents: int = Field(default=7, ge=1, le=25)
    default_context_chars: int = Field(default=24000, ge=2000, le=250000)
    max_full_document_chars: int = Field(default=120000, ge=1000, le=2_000_000)

    # How many chunks are considered before deciding what to return. Kept well above the number
    # returned because the two questions are different: retrieval only has to get the right chunk
    # into the room, reranking decides where it stands. Searching this deep without reranking
    # would only add noise, so the pool and the reranker belong together.
    candidate_pool_chunks: int = Field(default=40, ge=1, le=1000)

    # Guarantees the pool holds each signal's own favourites, not just the ones the fused order
    # liked. A passage nobody else votes for can still be the only one that names the answer.
    candidate_pool_per_signal: int = Field(default=25, ge=0, le=1000)

    # How many chunks are read in full to score them against the question. Comparing the wording
    # of a question against a whole chunk of prose costs more than the vector search, the text
    # search and the recency decay together, so it is spent only on chunks a cheaper signal
    # already ranked highly. Raise it if a fuzzy phrase buried mid-document is being missed.
    scoring_pool_chunks: int = Field(default=400, ge=1, le=20000)

    reranker_enabled: bool = True
    # Multilingual, and small enough to read a shortlist on a CPU. The larger v2-m3 was measured
    # against both a handwritten and a generated question set at this truncation: it matched the
    # hit rate of this model exactly on both, moved reciprocal rank by less than noise in opposite
    # directions, and cost three times the latency. Bigger is not better here.
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_revision: str | None = None
    # Deliberately far below the chunk size. A reranker averages relevance over what it is
    # given, so feeding it a whole chunk dilutes the few lines that answer the question. Cutting
    # it short scored better at every step down to this point, and costs less; below it,
    # truncation starts removing answers rather than padding.
    #
    # Raising it was retested against both a handwritten and a generated question set: 320 scored
    # better on generated probes and worse on written questions, and cost twice the latency. The
    # generated set rewards seeing further into a chunk for its own sake, so 192 stands.
    reranker_max_length: int = Field(default=192, ge=64, le=2048)
    # Shares one process-wide setting with local_embedding_threads; see the note there.
    reranker_threads: int = Field(default=8, ge=1)
    # Fused against the shortlist it was given, so retrieval keeps a say. Much smaller than
    # rrf_k because it fuses two orderings of a few hundred rows, not of the whole index.
    reranker_rrf_k: float = Field(default=20, gt=0)
    reranker_weight: float = Field(default=0.5, ge=0, le=1)
    reranker_batch_size: int = Field(default=16, ge=1, le=256)

    # Conversation. Off by default: an index that nobody chats with should not hold a connection
    # open to an inference server that may not be running.
    chat_enabled: bool = False
    chat_provider: Literal["ollama"] = "ollama"
    # Only ever a machine you control. create_chat_provider refuses anything that is not loopback,
    # the docker host gateway, or a private address, so "local only" is enforced rather than meant.
    ollama_base_url: str = "http://host.docker.internal:11434"
    chat_model: str = "qwen3:8b"
    # Rewrites a follow-up into a standalone question before retrieval, which is the difference
    # between "and what about the second one?" finding anything and finding nothing. Empty falls
    # back to stitching the recent turns together, which costs no second model in memory.
    chat_condense_model: str | None = None
    # Ollama defaults a context window to 4096 regardless of what the model can hold, which would
    # silently drop the end of the retrieved context. Always sent explicitly.
    chat_num_ctx: int = Field(default=16384, ge=2048, le=262144)
    chat_temperature: float = Field(default=0.2, ge=0, le=2)
    chat_max_tokens: int = Field(default=1024, ge=64, le=32768)
    # Qwen3 reasons before answering unless told not to. For an answer that is meant to stay inside
    # the retrieved sources, that is latency spent on nothing the reader sees.
    chat_thinking: bool = False
    chat_keep_alive: str = "10m"
    # How much of the conversation is carried into condensing. Whole turns, user and assistant.
    chat_history_turns: int = Field(default=6, ge=0, le=50)
    # Deliberately below default_context_chars: an agent with a large window can be handed more
    # than a local model can read without losing the middle of it.
    chat_context_chars: int = Field(default=12000, ge=1000, le=200000)
    # Answer only from what was retrieved. Off lets the model fall back on its own training, which
    # is exactly the failure this index exists to prevent.
    chat_require_evidence: bool = True
    # Progress for someone watching a chat window. The MCP tool and the eval harness never see it.
    chat_show_progress: bool = True
    # How many pictures an answer may carry. Enough to show what was asked for, few enough that a
    # slide-heavy deck does not bury the words.
    chat_max_images: int = Field(default=4, ge=0, le=20)
    chat_timeout_seconds: float = Field(default=180.0, gt=0)
    # Shared secret for /v1. Empty leaves the endpoint unauthenticated, which is only defensible
    # while the port stays bound to loopback.
    chat_api_key: str | None = None

    mcp_allowed_hosts: str = "localhost,localhost:*,127.0.0.1,127.0.0.1:*,[::1],[::1]:*"
    log_level: str = "INFO"
    # json emits one object per line for Loki; rich stays readable in a terminal.
    log_format: Literal["rich", "json"] = "rich"
    # Empty means evaluations are not recorded anywhere.
    mlflow_tracking_uri: str = ""

    @field_validator(
        "embedding_revision",
        "embedding_dimensions",
        "openai_api_key",
        "openai_base_url",
        "reranker_revision",
        "chat_condense_model",
        "chat_api_key",
        mode="before",
    )
    @classmethod
    def empty_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_smaller_than_chunk(cls, value: int, info):  # type: ignore[no-untyped-def]
        chunk_size = info.data.get("chunk_size", 1800)
        if value >= chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        return value

    @property
    def extensions(self) -> set[str]:
        return {item.strip().lower() for item in self.supported_extensions.split(",") if item.strip()}

    @property
    def ocr_language_list(self) -> list[str]:
        return [item.strip() for item in self.ocr_languages.replace(",", "+").split("+") if item.strip()]

    @property
    def transcription_language_list(self) -> list[str]:
        return [
            item.strip().lower()
            for item in self.transcription_languages.replace(",", "+").split("+")
            if item.strip()
        ]

    @property
    def allowed_hosts(self) -> list[str]:
        return [item.strip() for item in self.mcp_allowed_hosts.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
