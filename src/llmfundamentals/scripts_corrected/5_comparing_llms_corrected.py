"""
Teacher solution of src/llmfundamentals/5_comparing_llms.py.
Compare LLMs from Anthropic and OpenAI on the same prompts, using your LLMWrapper.
"""

import csv
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # so that "common" can be imported
from common.llmwrapper import LLMWrapper

# (provider, model)
MODELS = [
    ("anthropic", "anthropic/claude-haiku-4.5"),
    ("anthropic", "anthropic/claude-sonnet-5.5"),
    ("openai", "openai/gpt-4.1-mini"),
    ("openai", "mistralai/mistral-nemo"),  # the extra model: its price is already in PRICES_PER_MTOK
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
            response = llm.ask(prompt)
            logger.info("{} | {!r}: {}", model, prompt, response.text)
            rows.append([model, prompt, response.text, response.input_tokens, response.output_tokens, response.cost_usd, response.latency_seconds])

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(HEADER)
        writer.writerows(rows)
    logger.info("Saved {} rows to {}", len(rows), OUTPUT_FILE)

    # One line per model: average cost and average latency over its prompts (rows are in the order of HEADER).
    for _, model in MODELS:
        model_rows = [row for row in rows if row[0] == model]
        costs = [row[5] for row in model_rows if row[5] is not None]  # cost_usd is None when the price is unknown
        average_cost = sum(costs) / len(costs) if costs else None
        average_latency = sum(row[6] for row in model_rows) / len(model_rows)
        logger.info("{}: average cost {} USD, average latency {:.1f} s", model, average_cost, average_latency)


if __name__ == "__main__":
    main()
