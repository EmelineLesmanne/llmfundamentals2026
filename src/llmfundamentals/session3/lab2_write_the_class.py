"""
Write the Pydantic class

Goal: describe the structure you expect, as a Python class.

What to do:
  1. Open data/kettle.json and look at its keys and values.
  2. Complete the class ProductNotice below: one line per key of kettle.json, with the right type.
     The first line is written for you. Use the same names as the keys.
     Rules: if a value can be null (nothing), write `int | None` (or `str | None`).
            a list of texts is written `list[str]`.
  3. Run this file.

Bonus: log json.dumps(ProductNotice.model_json_schema(), indent=2) and find each of your fields in it.
This is a JSON Schema: a JSON that describes what another JSON must look like.
"""
from lab_tools import DATA
from loguru import logger
from pydantic import BaseModel


class ProductNotice(BaseModel):
    product_name: str
    # TODO: write the 6 other fields here


# Everything below only runs when you run this file directly (the next labs import the class above).
if __name__ == "__main__":
    if len(ProductNotice.model_fields) < 7:
        logger.info("TODO not done yet: the class needs 7 fields")
    else:
        kettle_json = (DATA / "kettle.json").read_text(encoding="utf-8")
        notice = ProductNotice.model_validate_json(kettle_json)
        logger.info(notice)
