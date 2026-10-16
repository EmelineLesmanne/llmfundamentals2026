"""
Compute the cost of a request from the token counts returned by the API.
"""

import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
MAX_TOKENS = 1000
PROMPT = "Explain in one paragraph what a token is."

# USD per million tokens. Check the OpenRouter models page before relying on these.
PRICES = {
    "anthropic/claude-haiku-4.5": {"input": 1.00, "output": 5.00},
    "anthropic/claude-sonnet-5.5": {"input": 2.00, "output": 10.00},
}

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def cost_usd(model, input_tokens, output_tokens):
    price = PRICES.get(model)
    if price is None:
        return None
    return round(
        (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000, 3
    )


def main():
    response = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": PROMPT}],
    )
    usage = response.usage
    cost = cost_usd(MODEL, usage.input_tokens, usage.output_tokens)
    logger.info(
        "{} input + {} output tokens = ${:.3f}",
        usage.input_tokens,
        usage.output_tokens,
        cost,
    )


if __name__ == "__main__":
    main()
