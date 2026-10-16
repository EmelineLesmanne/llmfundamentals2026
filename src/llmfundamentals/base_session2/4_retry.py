"""
Retry a request with exponential backoff, but only for errors that can go away by themselves.
"""

import functools
import os
import time

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"
OPENROUTER_URL = "https://openrouter.ai/api"

# Temporary errors worth retrying. A BadRequestError (400) is not here:
# the same request would fail again, so it is raised immediately.
RETRYABLE_ERRORS = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
)

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")


def retry_with_backoff(max_retries=3, base_delay=1.0):
    """Retry the function on RETRYABLE_ERRORS, waiting base_delay * 2**attempt seconds each time."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except RETRYABLE_ERRORS as error:
                    if attempt == max_retries:
                        raise
                    delay = base_delay * 2**attempt
                    logger.warning(
                        "{} (retry {}/{} in {} s)",
                        type(error).__name__,
                        attempt + 1,
                        max_retries,
                        delay,
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


@retry_with_backoff(max_retries=3, base_delay=0.5)
def ask(client, messages):
    response = client.messages.create(model=MODEL, max_tokens=100, messages=messages)
    return response.content[0].text


def main():
    # max_retries=0: the SDK must not retry by itself, otherwise we would not see our own retries
    good_client = anthropic.Anthropic(
        base_url=OPENROUTER_URL, auth_token=OPENROUTER_API_KEY, max_retries=0
    )
    # nothing listens on this address: every request fails with an APIConnectionError
    broken_client = anthropic.Anthropic(
        base_url="http://localhost:9", auth_token="none", max_retries=0
    )
    hello = [{"role": "user", "content": "Say hello in French."}]
    wrong_role = [{"role": "robot", "content": "Say hello in French."}]

    logger.info("1. a working request: {}", ask(good_client, hello))

    logger.info("2. a connection error is retried, then raised")
    try:
        ask(broken_client, hello)
    except anthropic.APIConnectionError:
        logger.error("gave up after 3 retries")

    logger.info("3. an invalid request is not retried")
    try:
        ask(good_client, wrong_role)
    except anthropic.APIStatusError as error:
        logger.error("{} raised immediately", type(error).__name__)


if __name__ == "__main__":
    main()
