"""
LAB 3 - Hybrid search (about 12 minutes, no model call)

Goal: understand when each way of searching works, and why we merge them.

THREE WAYS TO SEARCH THE CHUNKS

  DENSE   Compares MEANINGS. The question and every chunk were turned into embeddings (lists of numbers); the chunks
          whose embedding is closest to the question's win. Good with other words for the same idea
          ("right to be forgotten" finds "erasure"). Can miss exact numbers, codes and names.

  BM25    Compares WORDS. A chunk scores high when it contains the words of the question, and rare words
          (a number, a code, a name) count more than common ones. Good for exact terms. Blind to synonyms.

  HYBRID  Runs both, then merges the two rankings with Reciprocal Rank Fusion (RRF): a chunk at rank r in a ranking
          gets the score 1 / (60 + r) from it, and the scores of the two rankings are added. A chunk that is high in
          both rankings wins; a chunk that is first in only one can still come out near the top.

What to do:
  1. Read the function fuse below: it is the RRF merge, already written. Which rank gives the biggest score?
  2. BEFORE running, guess in the comment at the bottom, for each of the three questions, which method will put the
     right chunk first: A dense, B BM25, C hybrid. Read the questions: does each one rely on meaning, or on exact words?
  3. Run this file. For each question it prints the rank of the right chunk with each method (1 = best).
     Compare with your guesses. Do you see a method that wins every time?

Optional: set SHOW_FIRST_CHUNKS to True below to also see the FIRST chunk found by each method, and read it.
When a method does not put the right chunk first, what did it find instead? Why does that chunk look like a good answer
to that method?

Bonus: in fuse, replace 60 by 1 and run again. Do the hybrid ranks change? Then try 1000.
"""
from lab_tools import (
    bm25_search,
    dense_search,
    get_corpus,
    get_llm,
    hybrid_search,
    rank_of,
)
from loguru import logger

SHOW_FIRST_CHUNKS = False  # OPTIONAL: set to True to see the first chunk found by each method


def show_first_chunk(corpus, method, hits, right_chunk):
    """Log the best chunk found by one method: its id, its place in the law, and the start of its text.
    hits is a list of (chunk id, score) pairs, best first."""
    chunk = corpus.by_id[hits[0][0]]  # hits[0] is the best pair, hits[0][0] its chunk id
    note = "  <- the right chunk" if chunk["id"] == right_chunk else ""
    logger.info(f"first chunk with {method}: {chunk['id']} ({chunk['article']}: {chunk['title']}){note}")
    start = chunk["text"][:150].replace("\n", " ")  # the first 150 characters, on one line
    logger.info(f"    {start}...")


def fuse(rankings):
    """Reciprocal Rank Fusion.
    rankings: a list of rankings, each one a list of chunk ids, best first.
    Returns a dictionary {chunk id: score}: the higher the score, the better the chunk."""
    scores = {}
    for ranking in rankings:
        # enumerate(..., start=1) gives the rank of each chunk id in this ranking: 1, 2, 3, ...
        for rank, chunk_id in enumerate(ranking, start=1):
            # the better the rank, the bigger the score; a chunk found by both searches receives both scores
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
    return scores


# Everything below only runs when you run this file directly (lab 4 imports fuse).
if __name__ == "__main__":
    corpus = get_corpus()
    llm = get_llm()

    # (question, id of the chunk that answers it)
    cases = [
        ("What is the right to be forgotten?", "GDPR-Art17-1"),
        ("Which infringements are fined up to EUR 7 500 000?", "AIAct-Art99-5"),
        ("What are the obligations of deployers of high-risk AI systems?", "AIAct-Art26-1"),
    ]
    for question, right_chunk in cases:
        # dense: the 10 chunks with the closest MEANING (embeddings)
        dense = dense_search(corpus, llm, question, 10)
        # bm25: the 10 chunks sharing the most (rare) WORDS with the question
        bm25 = bm25_search(corpus, question, 10)
        # hybrid: dense and bm25 merged by fuse
        hybrid = hybrid_search(corpus, llm, question, fuse, k=10)

        logger.info(f"--- {question}")
        logger.info(f"right chunk: {right_chunk}")
        logger.info(f"rank with dense: {rank_of(right_chunk, dense)}")
        logger.info(f"rank with BM25: {rank_of(right_chunk, bm25)}")
        logger.info(f"rank with hybrid: {rank_of(right_chunk, hybrid)}")

        if SHOW_FIRST_CHUNKS:
            show_first_chunk(corpus, "dense", dense, right_chunk)
            show_first_chunk(corpus, "BM25", bm25, right_chunk)
            show_first_chunk(corpus, "hybrid", hybrid, right_chunk)

# My guesses BEFORE running (A dense, B BM25, C hybrid)
# question 1 =
# question 2 =
# question 3 =

# My conclusion AFTER running: which method should we use, and why?
