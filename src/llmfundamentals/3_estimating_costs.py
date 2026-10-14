"""
Estimate the cost of a request before sending it, then compute the real cost.
"""
import os
import anthropic
from dotenv import load_dotenv
from loguru import logger

MAX_TOKENS = 1000
PROMPT = "Say hello in French."

# Prices are in USD per million tokens (MTok)
PRICES = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5-5": {"input": 3.00, "output": 15.00},
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
    # Price per million tokens divided by 1,000,000
    input_cost = (input_tokens * price["input"]) / 1_000_000
    output_cost = (output_tokens * price["output"]) / 1_000_000
    return input_cost + output_cost


def estimate_input_tokens(model, messages):
    try:
        # Works on native Anthropic API
        return client.messages.count_tokens(model=model, messages=messages).input_tokens
    except anthropic.NotFoundError:
        # Fallback when routed through OpenRouter (which doesn't host /count_tokens)
        total_chars = sum(len(m["content"]) for m in messages)
        # Standard rule of thumb: ~4 characters per token
        return max(1, total_chars // 4)


def main():
    model_price_key = "claude-haiku-4-5"
    api_model = "anthropic/claude-haiku-4.5"

    # Input tokens count before calling the model
    input_tokens = estimate_input_tokens(model=api_model, messages=messages)
    logger.info(f"Estimated input tokens: {input_tokens}")

    # 1. Calculate input message cost
    input_cost = cost_usd(model_price_key, input_tokens=input_tokens, output_tokens=0)
    logger.info(f"Estimated input message cost: ${input_cost:.8f}")

    # 2. Calculate output max cost
    max_output_cost = cost_usd(model_price_key, input_tokens=0, output_tokens=MAX_TOKENS)
    logger.info(f"Estimated max output cost (for {MAX_TOKENS} tokens): ${max_output_cost:.8f}")
    logger.info(f"Estimated max total cost: ${input_cost + max_output_cost:.8f}")

    # Actual request
    response = client.messages.create(model=api_model, max_tokens=MAX_TOKENS, messages=messages)

    actual_input_tokens = response.usage.input_tokens
    actual_output_tokens = response.usage.output_tokens
    logger.info(f"Actual tokens: {actual_input_tokens} input, {actual_output_tokens} output")

    # 3. Calculate actual cost
    actual_cost = cost_usd(model_price_key, actual_input_tokens, actual_output_tokens)
    logger.info(f"Actual total cost: ${actual_cost:.8f}")
    logger.info(f"Response: {response.content[0].text.strip()}")


if __name__ == "__main__":
    main()