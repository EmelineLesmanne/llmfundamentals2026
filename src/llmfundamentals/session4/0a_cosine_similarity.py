"""
Step 0 - Embeddings and cosine similarity (demo)

An embedding turns a text into a list of numbers. Texts with similar meanings get similar lists.
The embeddings of the sentences below were computed in advance (by a model), so the first part needs no model call.

The cosine similarity measures how much two lists "point in the same direction":
1 means the same meaning, 0 means unrelated.

Run:  uv run python src/llmfundamentals/session4/0a_cosine_similarity.py

Bonus: write your own sentence in MY_TEXT below and run again. This one calls the model to compute its embedding.
"""
import json

import numpy as np
from lab_tools import DATA, get_llm
from loguru import logger

MY_TEXT = ""  # BONUS: write your own sentence here


def cosine_similarity(list_a, list_b):
    """1 = same direction (same meaning), 0 = unrelated."""
    a = np.array(list_a)
    b = np.array(list_b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def show_similarities(vector, sentences):
    for item in sentences:
        similarity = cosine_similarity(vector, item["embedding"])
        logger.info(f"{similarity:.3f}  {item['text']}")


data = json.loads((DATA / "sentences.json").read_text(encoding="utf-8"))
sentences = data["sentences"]
query = data["query"]["embedding"]

logger.info(f"Query: {data['query']['text']!r} is a list of {len(query)} numbers")
logger.info(f"The first five: {np.array(query[:5]).round(3)}")

logger.info("--- similarity of each sentence with the query ---")
show_similarities(query, sentences)

if MY_TEXT:
    logger.info(f"--- bonus: similarity of each sentence with {MY_TEXT!r} ---")
    my_embedding = get_llm().embed([MY_TEXT])[0]  # one call to the embedding model
    logger.info(f"The first five numbers of your text: {np.array(my_embedding[:5]).round(3)}")
    show_similarities(my_embedding, sentences)
