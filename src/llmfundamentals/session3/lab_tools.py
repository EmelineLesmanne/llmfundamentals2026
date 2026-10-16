"""Small helpers shared by the labs of session 3. You do not need to read or change this file."""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Which provider the labs use: "openai" or "anthropic". Your code does not change when you switch.
PROVIDER = "openai"

DATA = Path(__file__).resolve().parent / "data"

# makes `from common.llmwrapper import ...` work when you run a lab file
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()  # reads OPENROUTER_API_KEY from the .env file


from common.llmwrapper import (  # noqa: F401  (after the sys.path line above; the labs import these from here)
    LLMWrapper,
    StructuredOutputError,
    clear_fakes,
    fake_next,
)

# Show only the message (no time, no level) and hide the debug lines, so every screen looks the same.
logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


def get_llm():
    return LLMWrapper(PROVIDER)


def load_notice(name):
    """Return the text of a product notice: "kettle", "drill", "toy", "lamp" or "hair_dryer"."""
    notices = json.loads((DATA / "notices.json").read_text(encoding="utf-8"))["items"]
    for notice in notices:
        if notice["id"] == name:
            return notice["text"]
    raise ValueError(f"No notice called {name!r}")
