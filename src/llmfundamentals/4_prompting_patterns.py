"""
Prompting patterns
"""
import json
import os
import re
import time
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"  # OpenRouter model ID
# New OpenRouter accounts are limited to 20 requests per minute, so leave 3 s between calls
SECONDS_BETWEEN_CALLS = 3.1

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


_last_call = 0.0


def ask(prompt, system=None, max_tokens=300):
    """Send one prompt. Returns (reply text, input tokens, output tokens)."""
    global _last_call
    time.sleep(max(0.0, _last_call + SECONDS_BETWEEN_CALLS - time.monotonic()))
    _last_call = time.monotonic()

    kwargs = {"system": system} if system else {}
    response = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": 0.0},  # same prompt -> same answer, so strategies are compared fairly
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

TASKS = {"classification": CLS, "arithmetic": ARI, "style": STY}
STRATEGIES = ["zero-shot", "few-shot", "role/system", "chain of thought"]

# How each task names its input and its expected output, in the JSON and in the prompt
INPUT_KEY = {"classification": "input", "arithmetic": "question", "style": "input"}
OUTPUT_KEY = {"classification": "label", "arithmetic": "answer", "style": "output"}
INPUT_LABEL = {"classification": "Message", "arithmetic": "Problem", "style": "Notice"}
OUTPUT_LABEL = {"classification": "Category", "arithmetic": "Answer", "style": "Rewrite"}
ANSWER_ONLY = {
    "classification": "Reply with the category name only.",
    "arithmetic": "Reply with the final answer only: a number, or a time as HH:MM. No units, no explanation.",
    "style": "Reply with the rewritten notice only.",
}
ROLE = {
    "classification": "a customer-support triage assistant",
    "arithmetic": "a careful maths tutor",
    "style": "the copywriter who writes the Harbour Cafe chalkboard",
}
SHORT_MAX_TOKENS = {"classification": 10, "arithmetic": 20, "style": 60}
COT_MAX_TOKENS = 800


def brief(task):
    """Instruction plus rules, shared by every strategy so only the pattern changes."""
    data = TASKS[task]
    return f"{data['instruction']}\n{data['rules']}" if "rules" in data else data["instruction"]


def build_prompt(task, strategy, item):
    """Returns (system prompt, user prompt, max_tokens) for one item."""
    question = f"{INPUT_LABEL[task]}: {item[INPUT_KEY[task]]}"
    answer_slot = f"{OUTPUT_LABEL[task]}:"

    if strategy == "zero-shot":
        return None, f"{brief(task)}\n{ANSWER_ONLY[task]}\n\n{question}\n{answer_slot}", SHORT_MAX_TOKENS[task]

    if strategy == "few-shot":
        examples = "\n\n".join(
            f"{INPUT_LABEL[task]}: {ex[INPUT_KEY[task]]}\n{OUTPUT_LABEL[task]}: {ex[OUTPUT_KEY[task]]}"
            for ex in TASKS[task]["examples"]
        )
        prompt = f"{brief(task)}\n{ANSWER_ONLY[task]}\n\n{examples}\n\n{question}\n{answer_slot}"
        return None, prompt, SHORT_MAX_TOKENS[task]

    if strategy == "role/system":
        system = f"You are {ROLE[task]}.\n{brief(task)}\n{ANSWER_ONLY[task]}"
        return system, question, SHORT_MAX_TOKENS[task]

    # chain of thought
    prompt = (
        f"{brief(task)}\n\n{question}\n\n"
        "Think step by step, then end with a line of the form 'Final answer: <answer>'."
    )
    return None, prompt, COT_MAX_TOKENS


def extract_answer(strategy, reply):
    """Keep only the answer: the 'Final answer:' line for chain of thought, the whole reply otherwise."""
    if strategy == "chain of thought":
        matches = re.findall(r"final answer:\s*(.+)", reply, flags=re.IGNORECASE)
        reply = matches[-1] if matches else reply
    return reply.replace("**", "").strip()


def style_problems(text):
    """List the Harbour Cafe rules that the rewrite breaks (empty list = all rules respected)."""
    problems = []
    if not text.endswith(" ~ hc"):
        problems.append("signature")
    body = text.removesuffix(" ~ hc")
    if text != text.lower():
        problems.append("not lowercase")
    if "\n" in text:
        problems.append("several lines")
    if len(body.split()) > 12:
        problems.append(f"{len(body.split())} words")
    if "." in body:
        problems.append("full stop")
    return problems


def normalize_number(text):
    """Last number or HH:MM time in the text, e.g. 'The price is 103 euros.' -> '103'."""
    found = re.findall(r"\d{1,2}:\d{2}|\d+(?:\.\d+)?", text.replace(",", ""))
    if not found:
        return None
    value = found[-1]
    if ":" in value:
        hours, minutes = value.split(":")
        return f"{int(hours):02d}:{minutes}"
    return f"{float(value):g}"


def is_correct(task, item, answer):
    if task == "classification":
        return answer.lower().strip(" .") == item["label"]
    if task == "arithmetic":
        return normalize_number(answer) == normalize_number(item["answer"])
    return not style_problems(answer)  # style: no reference answer, check the rules instead


def compare_strategies():
    """Run every task with every strategy and return {(task, strategy): stats}."""
    results = {}
    for task, data in TASKS.items():
        for strategy in STRATEGIES:
            stats = {"correct": 0, "total": 0, "tokens_in": 0, "tokens_out": 0}
            logger.info("===== {} / {} =====", task, strategy)
            for item in data["items"]:
                system, prompt, max_tokens = build_prompt(task, strategy, item)
                reply, tokens_in, tokens_out = ask(prompt, system, max_tokens)
                answer = extract_answer(strategy, reply)
                correct = is_correct(task, item, answer)

                stats["correct"] += correct
                stats["total"] += 1
                stats["tokens_in"] += tokens_in
                stats["tokens_out"] += tokens_out

                expected = item.get(OUTPUT_KEY[task]) or ", ".join(style_problems(answer)) or "all rules OK"
                logger.info("{} {!r}  (expected: {})", "OK " if correct else "KO ", answer[:100], expected)
            results[(task, strategy)] = stats
    return results


def print_summary(results):
    lines = [f"{'task':<15}{'strategy':<18}{'score':>8}{'tokens in':>11}{'tokens out':>12}"]
    for (task, strategy), s in results.items():
        score = f"{s['correct']}/{s['total']}"
        lines.append(f"{task:<15}{strategy:<18}{score:>8}{s['tokens_in']:>11}{s['tokens_out']:>12}")
    logger.info("Summary (style score = rewrites that respect every rule)\n{}", "\n".join(lines))


def main():
    # EXAMPLE
    for name, (system, prompt, max_tokens) in PROMPTS.items():
        text, tokens_in, tokens_out = ask(prompt, system, max_tokens)
        logger.info("=== {} ({} tokens in, {} tokens out) ===\n{}", name, tokens_in, tokens_out, text)

    # Compare the LLM results for the three tasks and different prompting strategies.
    print_summary(compare_strategies())


if __name__ == "__main__":
    main()
