"""
LAB 2 - Dense search (about 15 minutes, no model call)

Goal: find the chunks of the AI Act and the GDPR whose meaning is closest to a question.

Each chunk is a paragraph of an article, with an id such as GDPR-Art17-1 (GDPR, article 17, paragraph 1).
Its embedding was computed in advance, and so was the embedding of the question below.

DOCUMENTATION

  dense_search(corpus, llm, question, k=5)
      Finds the k chunks whose embedding is closest to the embedding of the question.
        corpus    the corpus loaded below
        llm       the model wrapper loaded below (only used to embed a question that is not known in advance)
        question  the question, as a text
        k         how many chunks to return (5 if you do not say)
      Returns a list of (chunk_id, score) pairs, best first. The score is a cosine similarity (1 = same meaning).

  corpus.by_id
      A dictionary: chunk id -> chunk. A chunk is itself a dictionary, with the keys
      "id", "regulation", "article", "title" and "text".

What to do (three TODOs):
  1. Get the 5 best chunks for the question.
  2. For each hit, log one line with the score (3 decimals), two spaces, the chunk id, two spaces,
     and then "regulation, article: title" of the chunk. For example:
         0.123  GDPR-Art99-1  GDPR, Article 99: Entry into force and application
  3. Log the text of the best chunk.
Then run this file. Which chunk comes first?

Bonus: ask your own question (this one calls the model once, to embed it). Try k=10 or a question in French.
"""
from lab_tools import (
    dense_search,  # noqa: F401  (used once you complete the TODO)
    get_corpus,
    get_llm,
)
from loguru import logger

corpus = get_corpus()
llm = get_llm()

question = "What is the right to be forgotten?"
logger.info(f"Question: {question}")

hits = dense_search(corpus, llm, question, k=5)

if not hits:
    logger.info("TODO not done yet: call dense_search")

for chunk_id, score in hits:
    chunk = corpus.by_id[chunk_id]
    logger.info(f"{score:.3f}  {chunk_id}  {chunk['regulation']}, {chunk['article']}: {chunk['title']}")

best_id = hits[0][0]
logger.info(corpus.by_id[best_id]["text"])