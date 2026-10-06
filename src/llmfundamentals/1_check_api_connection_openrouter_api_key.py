import os

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "anthropic/claude-haiku-4.5"  # OpenRouter model ID

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = anthropic.Anthropic(
    base_url="https://openrouter.ai/api",
    auth_token=OPENROUTER_API_KEY,
)


def main():
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": "Say hello in French"}],
    )
    # response.content[0].text gives the response
    # response.usage.input_tokens gives the number of input tokens
    # response.usage.output_tokens gives the number of output tokens
    logger.info(response.content[0].text)
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens



if __name__ == "__main__":
    main()
