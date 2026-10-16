"""
LAB 6 - The retry loop

Goal: when an answer is rejected, send the error back to the model and ask again.

The loop is written for you. You write two things:
  TODO 1: the new prompt used for the next attempt. It must contain
          - the original prompt
          - the previous answer (error.answer)
          - what was wrong (error.problems)
          - a request for a corrected answer
  TODO 2: after the loop (every attempt failed): raise RuntimeError with a message.

Bonus: print current_prompt at attempt 2 and improve the wording.
"""

from lab_tools import (
    DATA,
    StructuredOutputError,
    clear_fakes,
    fake_next,
    get_llm,
    load_notice,
)
from loguru import logger


def ask_with_retry(llm, prompt, output_class, max_attempts=3):
    current_prompt = prompt
    for attempt in range(1, max_attempts + 1):
        try:
            result = llm.ask_structured(current_prompt, output_class)
            logger.info(f"attempt {attempt}: OK")
            return result
        except StructuredOutputError as error:
            logger.info(f"attempt {attempt}: failed")
            # Tip: an f-string with {prompt}, {error.answer} and {error.problems}
            current_prompt = f"""
                Your prompt was: {prompt}.
                Your answer was: {error.answer}, but it has an error {error.problems}.
                Correct it.
            """
            pass  # noqa: PIE790

    raise RuntimeError(
        f"We tried {max_attempts} times and it still failed to give a correct json."
    )


# Everything below only runs when you run this file directly (lab 7 imports ask_with_retry).
if __name__ == "__main__":
    from lab5_business_rules import CheckedNotice

    llm = get_llm()
    prompt = f"Extract the data of this product notice.\n\n{load_notice('kettle')}"

    logger.info("===== Step 1: one bad fake answer, then the real model")
    clear_fakes()
    fake_next(DATA / "fakes" / "kettle_power_zero.json")
    result = ask_with_retry(llm, prompt, CheckedNotice)
    if result is not None:
        logger.info(f"power_watts = {result.power_watts}")

    logger.info("===== Step 2: three bad fake answers")
    clear_fakes()
    for _ in range(3):
        fake_next(DATA / "fakes" / "kettle_power_zero.json")
    try:
        ask_with_retry(llm, prompt, CheckedNotice)
        logger.info("no error was raised")
    except RuntimeError as error:
        logger.info(f"final error: {error}")
