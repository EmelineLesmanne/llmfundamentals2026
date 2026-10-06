"""
Using model Haiku 4.5, change temperature then top_p.
"""

import anthropic
from dotenv import load_dotenv
from loguru import logger

MODEL = "claude-haiku-4-5"
#PROMPT = "Describe winter in one poetic sentence.""
PROMPT = "What cartoon character would make the best president?"
TEMPERATURES = [0.0, 0.5, 1.0]
TOP_PS = [0.1, 0.5, 1.0]

load_dotenv() 
client = anthropic.Anthropic()


def ask(prompt):
    response = client.messages.create(
        model=MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": 0.5},  # or {"top_p": 0.9}, not both
    )
    return response.content[0].text


def main():
    # TODO change temperature
    logger.info(ask(PROMPT))

    # TODO change top p



if __name__ == "__main__":
    main()
