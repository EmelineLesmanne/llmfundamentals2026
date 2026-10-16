"""
Compare LLMs from Anthropic and OpenAI on the same prompts, using your LLMWrapper.
"""

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so that "common" can be imported
from common.llmwrapper import LLMWrapper  # noqa: E402

# (provider, model)
MODELS = [
    ("anthropic", "anthropic/claude-haiku-4.5"),
    ("anthropic", "anthropic/claude-sonnet-5.5"),
    ("openai", "openai/gpt-4.1-mini"),
    # TODO add one more model, from either provider
    # Its price must be in PRICES_PER_MTOK (src/common/llmwrapper.py), otherwise its cost is None.
    # Check the price on the OpenRouter models page and add a line there.
]
PROMPTS = [
    "Explain in two sentences what a balance sheet is.",
    "Write a short professional email asking a professor for an extension on a report.",
    "A company has 120 employees and 35% work remotely. How many work on site?",
]
OUTPUT_FILE = Path(__file__).parent / "comparing_llms.csv"
HEADER = ["model", "prompt", "answer", "input_tokens", "output_tokens", "cost_usd", "latency_seconds"]

load_dotenv()  # reads the .env file


def main():
    rows = []
    for provider, model in MODELS:
        llm = LLMWrapper(provider, model=model)
        for prompt in PROMPTS:
            # TODO ask the model with your wrapper
            # Hint: response = llm.ask(prompt)
            #   response.text is the answer, and response.input_tokens, response.output_tokens,
            #   response.cost_usd and response.latency_seconds are the numbers you need.
            response = ...
            logger.info("{} | {!r}: {}", model, prompt, response.text)
            # TODO add a row, in the order of HEADER: model, prompt, answer, input tokens, output tokens, cost, latency
            rows.append([])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADER)
        writer.writerows(rows)
    logger.info("Saved {} rows to {}", len(rows), OUTPUT_FILE)

    # TODO log one line per model: average cost and average latency
    # Which model is the cheapest? The fastest? Is the answer of the cheapest one good enough?


if __name__ == "__main__":
    main()
