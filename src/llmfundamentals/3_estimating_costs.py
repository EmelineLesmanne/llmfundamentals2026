"""
Estimate the cost of a request before sending it, then compute the real cost.
"""
import anthropic
from dotenv import load_dotenv
from loguru import logger

MAX_TOKENS = 1000
PROMPT = "Say hello in French."

# Pricing page: https://platform.claude.com/docs/en/about-claude/pricing
# Fill in with the right prices
PRICES = {
    "claude-haiku-4-5": {"input": 0.00, "output": 0.00},
    "claude-sonnet-5-5": {"input": 0.00, "output": 00.00},
}

load_dotenv()
client = anthropic.Anthropic()
messages = [{"role": "user", "content": PROMPT}]


def cost_usd(model, input_tokens, output_tokens):
    price = PRICES[model]
    return # TODO calculate price


def main():
    model = "claude-haiku-4-5"

    # Input tokens cost
    input_tokens = client.messages.count_tokens(model=model, messages=messages).input_tokens

    logger.info("TODO calculate input message cost")
    logger.info("TODO calculate output max cost, using MAX_TOKENS")


    # Actual cost
    response = client.messages.create(model=model, max_tokens=MAX_TOKENS, messages=messages)

    logger.info("TODO calculate actual cost")


if __name__ == "__main__":
    main()
