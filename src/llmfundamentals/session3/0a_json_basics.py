import json

from lab_tools import DATA
from loguru import logger

text = (DATA / "kettle.json").read_text(encoding="utf-8")
logger.info("--- the JSON text ---")
logger.info(text)

# loading a json in Python: it becomes a dict
data = json.loads(text)

logger.info("--- Reading the fields ---")

logger.info(f"brand: {data['brand']}")
logger.info(f"power_watts: {data['power_watts']}")
logger.info(f"min_age: {data['min_age']}")  # null in JSON becomes None in Python
logger.info(
    f"first warning: {data['warnings'][0]}"
)  # a list: the first item is number 0

# To see real JSON again, use json.dumps:
logger.info("--- back to JSON ---")
logger.info(json.dumps(data, indent=2))
