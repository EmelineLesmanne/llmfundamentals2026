"""
Prompting patterns
"""
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "claude-haiku-4-5"

load_dotenv()
client = anthropic.Anthropic()


def ask(prompt, system=None, max_tokens=300):
    """Send one prompt. Returns (reply text, input tokens, output tokens)."""
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


# Zero-shot: just the instruction, no worked example.
ZERO_SHOT = """Classify the message as: complaint, question or praise.

Message: Thanks, the support team was great!
Category:"""

# Few-shot: the same task with several worked examples.
FEW_SHOT = """Classify the message as: complaint, question or praise.

Message: My parcel arrived broken.
Category: complaint

Message: Where can I find my invoice?
Category: question

Message: I love the new app design.
Category: praise

Message: Thanks, the support team was great!
Category:"""

# Role/system: a system prompt sets the role and the rules; the user turn is only the input.
SYSTEM = """You are a customer-support triage assistant.
Classify each message as exactly one of: complaint, question, praise.
Reply with the category only, in lowercase."""
ROLE_PROMPT = "Message: Thanks, the support team was great!"

# Chain of thought: ask the model to reason step by step before giving the answer.
CHAIN_OF_THOUGHT = """A train leaves at 14:45 and the trip takes 2 hours 50 minutes.
A taxi from the station to the hotel takes 25 minutes.
At what time do I reach the hotel?

Think step by step, then end with a line of the form 'Final answer: <time>'."""

# name -> (system prompt, user prompt, max_tokens)
PROMPTS = {
    "zero-shot": (None, ZERO_SHOT, 10),
    "few-shot": (None, FEW_SHOT, 10),
    "role/system": (SYSTEM, ROLE_PROMPT, 10),
    "chain of thought": (None, CHAIN_OF_THOUGHT, 300),
}


DATA = Path(__file__).resolve().parents[2] / "docs" / "session1"


def load(name):
    return json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))


CLS = load("classification")
ARI = load("arithmetic")
STY = load("style")


def main():
    # EXAMPLE
    for name, (system, prompt, max_tokens) in PROMPTS.items():
        text, tokens_in, tokens_out = ask(prompt, system, max_tokens)
        logger.info("=== {} ({} tokens in, {} tokens out) ===\n{}", name, tokens_in, tokens_out, text)

    # TODO compare the LLM results for the three tasks and different prompting strategies.

    


if __name__ == "__main__":
    main()
