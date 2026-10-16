"""
LAB 3 - Your first ask_structured() call

Goal: get an object with the right fields from a notice, without parsing anything yourself.

What to do:
  1. Complete the TODO: call llm.ask_structured(prompt, ProductNotice, verbose=True)
  2. Run this file
  3. Read the lines printed by verbose=True. Compare your power_watts and warranty_years with the class.

"""

from lab2_write_the_class import (
    ProductNotice,
)
from lab_tools import get_llm, load_notice
from loguru import logger

llm = get_llm()
notice = load_notice("kettle")
prompt = f"Extract the data of this product notice.\n\n{notice}"

result = llm.ask_structured(prompt, ProductNotice, verbose=True)

if result is None:
    logger.info("TODO not done yet: call ask_structured")
else:
    logger.info(f"power_watts = {result.power_watts}")
    logger.info(f"warranty_years = {result.warranty_years}")
