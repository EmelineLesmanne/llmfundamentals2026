"""
LAB 1 - The problem (about 10 minutes, calls the model)

Goal: see what a model answers about a recent law when it has no sources.

What to do:
  1. Run this file. The model answers from memory. Below its answer, the official text is printed
     (Article 113 of the AI Act, as amended in July 2026: its second and third paragraphs).
  2. Find the date in the official text, compare it with the date given by the model,
     and write both in the comment at the bottom.

Bonus: ask the model another question about the AI Act and check its answer in the official text.
"""
from lab_tools import get_corpus, get_llm
from loguru import logger

llm = get_llm(use_cache=False)  # no cache: asking again gives a new answer
corpus = get_corpus()

question = "From what date does the AI Act apply to the high-risk AI systems of Annex III?"
logger.info("--- answer of the model, without any source ---")
logger.info(llm.ask(question).text)

logger.info("--- official text ---")
for chunk_id in ["AIAct-Art113-2", "AIAct-Art113-3"]:
    logger.info(corpus.by_id[chunk_id]["text"])

# My notes
# date given by the model =
# date in the official text =
