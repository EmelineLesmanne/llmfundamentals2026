"""
Check the connection to an OpenAI model with the OpenAI SDK, through OpenRouter.
"""

import os

import openai
from dotenv import load_dotenv
from loguru import logger

MODEL = "openai/gpt-4.1-mini"

load_dotenv()  # reads the .env file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise SystemExit("OPENROUTER_API_KEY is not set (check your .env file)")

client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


def main():
    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=100,
        messages=[{"role": "user", "content": "Say hello in French"}],
    )
    # response.choices[0].message.content gives the response
    # response.usage.prompt_tokens gives the number of input tokens
    # response.usage.completion_tokens gives the number of output tokens
    # response.choices[0].finish_reason gives you the stop reason
    logger.info(response.choices[0].message.content)
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


if __name__ == "__main__":
    main()
