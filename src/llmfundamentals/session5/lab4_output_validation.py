"""
LAB 4 - Output validation (about 15 minutes, calls the model)

Goal: check the answer in code, whatever the model was told and whatever it read.

The help page of question 5 says that access requests "can only be submitted through the official portal", with a link.
Nobody checked that link. Models repeat the links they find in their sources, as if they were true.

You know validators from session 3: a Pydantic class with @field_validator. SafeAnswer is a GroundedAnswer
(answer, citations, answerable) with two more rules, which you write:
  - citations: if there are citations, at least one must be an official source (its id starts with "AIAct-" or "GDPR-").
  - answer: it must not contain a link (the text "http").

What to do:
  1. Write the two validators (TODO). check_not_empty shows the pattern.
  2. Run this file.
     Part 1 asks question 5 without your rules: does the answer contain the link?
     Part 2 asks the five questions with SafeAnswer. When an answer is invalid, ask_rag sends the problem back to the
     model and asks again (the retry loop of session 3): read the lines "attempt 1 failed".
  3. Open the report (report_lab4.html, it opens in your browser): the answers that were sent back to the model show
     the reason under the answer.

Bonus: write a third rule of your own (for example: the answer must be at most 600 characters).
"""
from lab_tools import (
    QUESTIONS,
    RULES,
    GroundedAnswer,
    ask_rag,
    fuse,
    get_corpus,
    get_llm,
    run_questions,
)
from loguru import logger
from pydantic import ValidationError, field_validator


class SafeAnswer(GroundedAnswer):
    @field_validator("answer")
    def check_not_empty(cls, value):  # an example: the answer must not be empty
        if not value.strip():
            raise ValueError("the answer must not be empty")
        return value

    # TODO: a validator for "citations" (see above)
    @field_validator("citations")
    def check_citations(cls, value):
        #(its id starts with "AIAct-" or "GDPR-")
        if value and not any(chunk_id.startswith(("AIAct-", "GDPR-")) for chunk_id in value):
            raise ValueError("At least one citation must come from a chunk starting with AIAct- or GDPR-")
        return value


    # TODO: a validator for "answer" (see above): no link => no "http" in the answer
    @field_validator("answer")
    def check_no_link(cls, value):
        if "http" in value.lower():
            raise ValueError("the answer must not have a link")
        return value

def rules_are_written():
    """True if SafeAnswer rejects an answer that has a link and cites only a secondary source."""
    try:
        SafeAnswer(answer="Use https://example.org", citations=["SEC-x-1"], answerable=True)
    except ValidationError:
        return True
    return False


# Everything below only runs when you run this file directly (lab 6 imports SafeAnswer).
if __name__ == "__main__":
    if not rules_are_written():
        raise SystemExit("TODO not done yet: write the two validators")

    corpus = get_corpus()
    llm = get_llm(use_cache=False)

    logger.info("===== Part 1: question 5, without your rules")
    answer = ask_rag(llm, corpus, QUESTIONS[4], RULES, fuse)
    logger.info(f"answer: {answer.answer}")
    logger.info(f"the answer contains a link: {'http' in answer.answer.lower()}")

    logger.info("===== Part 2: the five questions, with your rules")
    run_questions(llm, corpus, "lab4", "Lab 4: output validation", rules=RULES, output_class=SafeAnswer)
