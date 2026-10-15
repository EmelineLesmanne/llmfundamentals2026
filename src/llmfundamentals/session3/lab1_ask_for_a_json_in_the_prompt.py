"""
LAB 1 - Ask for JSON in a plain prompt (about 10 minutes, calls the model)

Goal: see what happens when you simply ask a model for JSON.

"""
import json

from lab_tools import get_llm, load_notice
from loguru import logger

llm = get_llm()
notice = load_notice("kettle")
prompt = f"Extract the product information from this notice as JSON.\n\n{notice}."

answer = llm.ask(prompt).text
logger.info("--- answer of the model ---")
logger.info(answer)
logger.info("--- json.loads ---")
try:
    data = json.loads(answer)
    logger.info(f"keys: {list(data.keys())}")
except json.JSONDecodeError:
    logger.info("json.loads failed")

