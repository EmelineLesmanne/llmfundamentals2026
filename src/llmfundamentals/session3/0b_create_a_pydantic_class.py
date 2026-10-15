"""
Step 0 - Create a Pydantic class (demo, no model call)

A Pydantic class describes the data you expect: one line per field, with its type.
Pydantic then checks that some JSON follows this description.

"""
from loguru import logger
from pydantic import BaseModel, ValidationError

import lab_tools  # noqa: F401  (only sets up the logger so the output looks the same on every screen)


class Book(BaseModel):
    title: str
    pages: int
    subtitle: str | None  # may be null, but the key must be there


good_json = '{"title": "Dune", "pages": 612, "subtitle": null}'
book = Book.model_validate_json(good_json)
logger.info(f"title: {book.title}")
logger.info(f"pages: {book.pages}")
logger.info(f"subtitle: {book.subtitle}")

# "pages" is a text here, and "title" is missing: Pydantic says what is wrong
bad_json = '{"pages": "many", "subtitle": null}'
try:
    Book.model_validate_json(bad_json)
except ValidationError as error:
    logger.info("--- the answer is not valid ---")
    logger.info(error)
