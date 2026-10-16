"""
Cache the answers in a SQLite file, so that asking the same question twice costs nothing.
"""

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 1000
PROMPT = "Explain in one paragraph what a token is."
CACHE_DB = (
    Path(__file__).resolve().parents[3] / ".cache" / "session2_cache.sqlite"
)  # git-ignored
CACHE_TTL_SECONDS = 30 * 24 * 3600  # entries live one month

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)

CACHE_DB.parent.mkdir(exist_ok=True)
db = sqlite3.connect(CACHE_DB)
db.execute(
    "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, answer TEXT, created_at REAL)"
)


def cache_key(model, max_tokens, prompt):
    """SHA-256 of everything that changes the answer: the same inputs always give the same key."""
    return hashlib.sha256(json.dumps([model, max_tokens, prompt]).encode()).hexdigest()


def ask(prompt):
    key = cache_key(MODEL, MAX_TOKENS, prompt)
    row = db.execute(
        "SELECT answer FROM cache WHERE key = ? AND created_at >= ?",
        (key, time.time() - CACHE_TTL_SECONDS),
    ).fetchone()
    if row:
        logger.info("cache hit")
        return row[0]

    logger.info("cache miss, calling the API")
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    with db:  # commits the insert
        db.execute(
            "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)", (key, answer, time.time())
        )
    return answer


def main():
    for _ in range(2):
        start = time.perf_counter()
        ask(PROMPT)
        logger.info("answered in {:.3f} s", time.perf_counter() - start)


if __name__ == "__main__":
    main()
