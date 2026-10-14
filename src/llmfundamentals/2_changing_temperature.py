"""
Using model Haiku 4.5, change temperature then top_p.
"""

import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"  # OpenRouter model ID
#PROMPT = "Describe winter in one poetic sentence.""
PROMPT = "What cartoon character would make the best president?"
TEMPERATURES = [0.0, 0.5, 1.0]
TOP_PS = [0.1, 0.5, 1.0]
RUNS_PER_VALUE = 3  # same prompt several times to see how much answers vary

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def ask(prompt, sampling):
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
        extra_body=sampling,  # {"temperature": ...} or {"top_p": ...}, not both
    )
    return response.content[0].text


def main():
    for temperature in TEMPERATURES:
        logger.info(f"===== temperature = {temperature} =====")
        for run in range(1, RUNS_PER_VALUE + 1):
            logger.info(f"[run {run}] {ask(PROMPT, {'temperature': temperature})}")

    for top_p in TOP_PS:
        logger.info(f"===== top_p = {top_p} =====")
        for run in range(1, RUNS_PER_VALUE + 1):
            logger.info(f"[run {run}] {ask(PROMPT, {'top_p': top_p})}")


if __name__ == "__main__":
    main()
