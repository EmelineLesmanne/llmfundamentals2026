"""
LAB 5 - Business rules with field_validator

Goal: a schema checks types, not common sense. Add rules that reject impossible values.

Below, CheckedNotice is your ProductNotice plus rules. One rule is written for min_age.

What to do:
  1. Write the rule for power_watts: when it is not None, it must be between 1 and 5000.
     Imitate check_min_age. The message must be: "power_watts must be between 1 and 5000".
  2. Run this file
     Part 1 uses a fake answer with power_watts = 0 (no model call): you should see your message.
     Part 2 asks the model about the five notices.

Bonus: write a rule for warranty_years (between 0 and 10, or None) and look at the drill notice.
"""

from lab2_write_the_class import ProductNotice
from lab_tools import DATA, StructuredOutputError, fake_next, get_llm, load_notice
from loguru import logger
from pydantic import field_validator

if len(ProductNotice.model_fields) < 7:
    raise SystemExit(
        "Finish lab2 first (or copy the class from the display), then run this file again."
    )


class CheckedNotice(ProductNotice):
    @field_validator("min_age")
    def check_min_age(cls, value):
        if value is not None and not (0 <= value <= 18):
            raise ValueError("min_age must be between 0 and 18")
        return value

    # TODO: write check_power for "power_watts", like check_min_age above. between 0 and 8000
    @field_validator("power_watts")
    def check_power(cls, value):
        if value is not None and not (10 <= value <= 8000):
            raise ValueError("power_watts must be between 100 and 8000")
        return value

    # TODO (bonus): write check_warranty for "warranty_years" less than 100


# Everything below only runs when you run this file directly.
if __name__ == "__main__":
    llm = get_llm()

    logger.info("===== Part 1: fake answer with power_watts = 0")
    fake_next(DATA / "fakes" / "kettle_power_zero.json")
    try:
        llm.ask_structured("unused prompt", CheckedNotice)
        logger.info("validation: OK  <- the rule is missing or wrong")
    except StructuredOutputError as error:
        logger.info(error.problems)

    logger.info("===== Part 2: the five notices")
    for name in ["kettle", "drill", "toy", "lamp", "hair_dryer"]:
        prompt = f"Extract the data of this product notice.\n\n{load_notice(name)}"
        try:
            result = llm.ask_structured(prompt, CheckedNotice)
            logger.info(
                f"{name} -> power: {result.power_watts} warranty: {result.warranty_years} min_age: {result.min_age}"
            )
        except StructuredOutputError as error:
            logger.info(f"{name} -> FAILED")
            logger.info(error.problems)
