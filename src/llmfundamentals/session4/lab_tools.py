"""Small helpers shared by the labs of session 4. You do not need to read or change this file."""
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Which provider the labs use: "openai" or "anthropic". Your code does not change when you switch.
PROVIDER = "openai"

DATA = Path(__file__).resolve().parent / "data"

# makes `from common.llmwrapper import ...` work when you run a lab file
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()  # reads OPENROUTER_API_KEY from the .env file

from common.llmwrapper import LLMWrapper
from common.rag import (  # noqa: F401  (the labs import these from here)
    ask_rag,
    bm25_search,
    build_prompt,
    dense_search,
    hybrid_search,
    load_corpus,
)

# Show only the message (no time, no level) and hide the debug lines, so every screen looks the same.
logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def get_llm(use_cache=True):
    """The model wrapper. With the cache on, asking the same question twice does not call the model again."""
    return LLMWrapper(PROVIDER, use_cache=use_cache)


def get_corpus():
    """The chunks of the AI Act (as amended in July 2026) and of the GDPR, with their search indexes."""
    return load_corpus(DATA)


def rank_of(chunk_id, hits):
    """Position (1 = best) of a chunk id in a list of (chunk id, score), or "not in the top N"."""
    for position, (hit_id, _) in enumerate(hits, start=1):
        if hit_id == chunk_id:
            return position
    return f"not in the top {len(hits)}" if hits else "no result"
