"""
Prompting patterns
"""
import json
import os
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def ask(prompt, system=None, max_tokens=300):
    """Send one prompt. Returns (reply text, input tokens, output tokens)."""
    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    return response.content[0].text.strip(), response.usage.input_tokens, response.usage.output_tokens


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
    path = DATA / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    logger.warning(f"File not found: {path}. Using empty list.")
    return []


CLS = load("classification")
ARI = load("arithmetic")
STY = load("style")


def main():
    # 1. Base demo comparing prompting strategies on single examples
    logger.info("=== Running Base Prompting Strategies Demo ===")
    for name, (system, prompt, max_tokens) in PROMPTS.items():
        text, tokens_in, tokens_out = ask(prompt, system, max_tokens)
        logger.info("=== {} ({} tokens in, {} tokens out) ===\n{}", name, tokens_in, tokens_out, text)
        time.sleep(1.5)  # Rate limit safety

    # 2. Compare strategies across datasets
    logger.info("\n=== Comparing Strategies Across Datasets ===")

    # Task 1: Classification (Testing Zero-Shot vs Few-Shot format compliance)
    if CLS:
        sample = CLS[0]
        text_input = sample.get("text") or sample.get("message") or str(sample)

        zero_shot_prompt = f"Classify as complaint, question, or praise:\n{text_input}"
        system_role = "Classify as complaint, question, or praise. Reply with category only in lowercase."

        res_zero, in_z, out_z = ask(zero_shot_prompt, max_tokens=10)
        time.sleep(1.5)
        res_sys, in_s, out_s = ask(text_input, system=system_role, max_tokens=10)
        time.sleep(1.5)

        logger.info("[CLS] Zero-shot result: '{}' (in: {}, out: {})", res_zero, in_z, out_z)
        logger.info("[CLS] System-role result: '{}' (in: {}, out: {})", res_sys, in_s, out_s)

    # Task 2: Arithmetic (Testing Direct vs Chain of Thought reasoning)
    if ARI:
        problem = ARI[0].get("question") or str(ARI[0])

        direct_prompt = f"{problem}\nGive only the final answer."
        cot_prompt = f"{problem}\nThink step by step and give the final answer."

        res_direct, in_d, out_d = ask(direct_prompt, max_tokens=20)
        time.sleep(1.5)
        res_cot, in_c, out_c = ask(cot_prompt, max_tokens=300)
        time.sleep(1.5)

        logger.info("[ARI] Direct answer: '{}' (tokens out: {})", res_direct, out_d)
        logger.info("[ARI] Chain of thought:\n{}\n(tokens out: {})", res_cot, out_c)

    # Task 3: Style Transfer
    if STY:
        text_to_style = STY[0].get("text") or str(STY[0])
        style_prompt = f"Rewrite in a formal, professional tone:\n{text_to_style}"

        res_sty, in_st, out_st = ask(style_prompt, max_tokens=100)
        logger.info("[STY] Transformed text:\n{}", res_sty)


if __name__ == "__main__":
    main()