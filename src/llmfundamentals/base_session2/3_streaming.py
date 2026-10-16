"""
Streaming: print the answer piece by piece while the model is still writing it.
"""

import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 1000
PROMPT = "Draft a professional email to a client to postpone a project deadline by two weeks."

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def main():
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)  # each piece is shown as soon as it arrives
        response = (
            stream.get_final_message()
        )  # the complete message, with the token counts
    print()
    logger.info(
        "{} input + {} output tokens",
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


if __name__ == "__main__":
    main()
