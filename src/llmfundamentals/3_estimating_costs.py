"""
Estimate the cost of a request before sending it, then compute the real cost.
"""
import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MAX_TOKENS = 1000
PROMPT = "Say hello in French."

# Pricing page: https://platform.claude.com/docs/en/about-claude/pricing
# Prices in USD per million tokens (MTok)
PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}
# Same models, as named on OpenRouter
OPENROUTER_IDS = {
    "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
    "claude-sonnet-5-5": "anthropic/claude-sonnet-5.5",
}

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)
messages = [{"role": "user", "content": PROMPT}]


def cost_usd(model, input_tokens, output_tokens):
    price = PRICES[model]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1_000_000


def count_input_tokens(model):
    """Exact count with the token counting endpoint when available.

    OpenRouter does not expose that endpoint (404), so fall back to the usual
    rule of thumb: about 4 characters per token in English.
    """
    try:
        return client.messages.count_tokens(model=OPENROUTER_IDS[model], messages=messages).input_tokens, "exact"
    except anthropic.NotFoundError:
        characters = sum(len(m["content"]) for m in messages)
        return max(1, round(characters / 4)), "approx. 4 chars/token"


def main():
    for model in PRICES:
        logger.info("===== {} =====", model)

        # Input tokens cost
        input_tokens, method = count_input_tokens(model)
        input_cost = cost_usd(model, input_tokens, 0)
        max_output_cost = cost_usd(model, 0, MAX_TOKENS)
        logger.info("Estimated input: {} tokens ({}) -> ${:.8f}", input_tokens, method, input_cost)
        logger.info("Max output: {} tokens -> ${:.8f}", MAX_TOKENS, max_output_cost)
        logger.info("Worst case before sending: ${:.8f}", input_cost + max_output_cost)

        # Actual cost
        response = client.messages.create(model=OPENROUTER_IDS[model], max_tokens=MAX_TOKENS, messages=messages)
        usage = response.usage
        actual_cost = cost_usd(model, usage.input_tokens, usage.output_tokens)
        logger.info("Reply: {}", response.content[0].text)
        logger.info(
            "Actual: {} input + {} output tokens -> ${:.8f}",
            usage.input_tokens, usage.output_tokens, actual_cost,
        )


if __name__ == "__main__":
    main()
