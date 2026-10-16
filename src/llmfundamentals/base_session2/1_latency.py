"""
Measure the latency of a request: the time between sending it and receiving the full answer.
"""

import os
import time

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 1000
PROMPTS = [
    "Say hello in French.",
    "Write a cover letter for a data analyst internship at a bank.",
]

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def main():
    for prompt in PROMPTS:
        start = time.perf_counter()
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.perf_counter() - start
        logger.info(
            "{!r}: {:.3f} s for {} output tokens",
            prompt,
            latency,
            response.usage.output_tokens,
        )


if __name__ == "__main__":
    main()
