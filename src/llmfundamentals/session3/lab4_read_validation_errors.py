"""
LAB 4 - Read validation errors

Goal: learn to read the message that says why an answer was rejected.

Four answers of a "model" are in data/broken/. Each one has exactly one problem.
The code below gives each answer to ask_structured() (no model is called) and prints what was wrong.

What to do:
  1. Run this file
  2. Match each answer (1 to 4) with its cause, and write your four letters in the comment at the bottom.
       A. a field is missing
       B. a text where a number is expected
       C. not valid JSON
       D. a text where an array (a list) is expected

Bonus: open the four files and fix each one by hand until all four print "validation: OK".
"""

from lab2_write_the_class import ProductNotice
from lab_tools import DATA, StructuredOutputError, fake_next, get_llm
from loguru import logger

llm = get_llm()

for number in [1, 2, 3, 4]:
    logger.info(f"----- answer {number}")
    fake_next(DATA / "broken" / f"answer_{number}.json")
    try:
        llm.ask_structured("unused prompt", ProductNotice)
        logger.info("validation: OK")
    except StructuredOutputError as error:
        logger.info(error.problems)

# My answers
# answer 1 =
# answer 2 =
# answer 3 =
# answer 4 =
