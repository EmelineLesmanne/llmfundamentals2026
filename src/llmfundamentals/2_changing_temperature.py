"""
Using model Haiku 4.5, change temperature then top_p.
"""

import os
import time
import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
# PROMPT = "Describe winter in one poetic sentence."
PROMPT = "What cartoon character would make the best president?"
TEMPERATURES = [0.0, 0.5, 1.0]
TOP_PS = [0.1, 0.5, 1.0]

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def ask_temperature(prompt, temperature):
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": temperature},
    )
    return response.content[0].text


def ask_top_p(prompt, top_p):
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"top_p": top_p},
    )
    return response.content[0].text


def main():
    # TODO change temperature
    logger.info("--- Varying Temperature ---")
    for temp in TEMPERATURES:
        logger.info(f"Temperature: {temp}")
        logger.info(ask_temperature(PROMPT, temp))
        time.sleep(2)

    # TODO change top p
    logger.info("--- Varying Top P ---")
    for p in TOP_PS:
        logger.info(f"Top P: {p}")
        logger.info(ask_top_p(PROMPT, p))
        time.sleep(2)


if __name__ == "__main__":
    main()