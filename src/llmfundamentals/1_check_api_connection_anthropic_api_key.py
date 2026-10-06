
import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "claude-haiku-4-5"

load_dotenv()  # reads the .env file

client = anthropic.Anthropic()


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
