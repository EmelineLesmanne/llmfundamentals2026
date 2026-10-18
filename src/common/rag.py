"""Retrieval-augmented generation (RAG) tools, shared by the sessions on RAG.

The corpus is a list of chunks (one per article paragraph), each with an id such as "GDPR-Art17-1", and one embedding
per chunk. Searching returns lists of (chunk id, score). ask_rag() puts the best chunks in a prompt and returns an
answer whose citations have been checked in code.
"""
import json
import re
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
from loguru import logger
from pydantic import BaseModel
from rank_bm25 import BM25Okapi

from common.llmwrapper import StructuredOutputError


@dataclass
class Corpus:
    chunks: list  # one dict per chunk: id, regulation, article, title, text, ...
    by_id: dict  # chunk id -> chunk
    index: object  # FAISS index: finds the chunks whose embedding is closest to a question
    bm25: object  # BM25 index: finds the chunks that share words with a question
    query_vectors: dict  # embeddings of fixed questions, computed in advance (so no model call is needed)
    model: str  # the embedding model that built the index: questions must be embedded with the same one


def tokenize(text):
    """Lowercase words and numbers: "Article 22" -> ["article", "22"]."""
    return re.findall(r"\w+", text.lower())


def load_corpus(folder, suffix=""):
    """Load chunks{suffix}.jsonl, embeddings{suffix}.npy and embedding_model.txt from folder, and build the two search indexes."""
    folder = Path(folder)
    lines = (folder / f"chunks{suffix}.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines]
    vectors = np.load(folder / f"embeddings{suffix}.npy")  # one normalised row per chunk
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of normalised vectors = cosine similarity
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(f"{c['article']} {c['title']} {c['text']}") for c in chunks])
    query_file = folder / "query_embeddings.json"
    query_vectors = json.loads(query_file.read_text(encoding="utf-8")) if query_file.exists() else {}
    model = (folder / "embedding_model.txt").read_text(encoding="utf-8").strip()
    return Corpus(chunks, {c["id"]: c for c in chunks}, index, bm25, query_vectors, model)


def embed_question(corpus, llm, question):
    """The normalised embedding of a question: precomputed if the question is a known one, else asked to the model."""
    vector = corpus.query_vectors.get(question)
    if vector is None:
        vector = llm.embed([question], model=corpus.model)[0]
    vector = np.array([vector], dtype="float32")
    return vector / np.linalg.norm(vector)


def dense_search(corpus, llm, question, k=5):
    """The k chunks whose meaning is closest to the question: a list of (chunk id, cosine similarity)."""
    scores, positions = corpus.index.search(embed_question(corpus, llm, question), k)
    return [(corpus.chunks[p]["id"], float(s)) for s, p in zip(scores[0], positions[0], strict=True)]


def bm25_search(corpus, question, k=5):
    """The k chunks sharing the most (rare) words with the question: a list of (chunk id, BM25 score)."""
    scores = corpus.bm25.get_scores(tokenize(question))
    best = np.argsort(-scores, kind="stable")[:k]
    return [(corpus.chunks[p]["id"], float(scores[p])) for p in best]


def hybrid_search(corpus, llm, question, fusion, k=5, depth=20):
    """Dense and BM25 searches merged by `fusion` (a function: list of rankings -> {chunk id: score})."""
    rankings = [
        [chunk_id for chunk_id, _ in dense_search(corpus, llm, question, depth)],
        [chunk_id for chunk_id, _ in bm25_search(corpus, question, depth)],
    ]
    scores = fusion(rankings)
    best = sorted(scores, key=scores.get, reverse=True)[:k]  # sorted() is stable: ties keep the dense order
    return [(chunk_id, scores[chunk_id]) for chunk_id in best]


class GroundedAnswer(BaseModel):
    answer: str
    citations: list[str]  # ids of the sources the answer relies on
    answerable: bool  # False when the sources do not contain the answer


def build_prompt(question, chunks, rules):
    """The rules, then the sources inside tags, then the question."""
    sources = ""
    for chunk in chunks:
        sources += (
            f'<source id="{chunk["id"]}" regulation="{chunk["regulation"]}" article="{chunk["article"]}" '
            f'title="{chunk["title"]}">\n{chunk["text"]}\n</source>\n'
        )
    return f"{rules}\n\n<sources>\n{sources}</sources>\n\nQuestion: {question}"


def check_citations(answer, retrieved_ids):
    """What is wrong with the citations of a GroundedAnswer, one problem per line (empty text if nothing)."""
    problems = []
    if answer.answerable and not answer.citations:
        problems.append("citations: an answerable question needs at least one citation")
    for chunk_id in answer.citations:
        if chunk_id not in retrieved_ids:
            problems.append(f"citations: {chunk_id!r} is not one of the sources")
    return "\n".join(problems)


def ask_rag(llm, corpus, question, rules, fusion, k=5, max_attempts=3, verbose=False):
    """Retrieve the k best chunks (hybrid search with `fusion`), ask for a GroundedAnswer, check its citations in code.
    If the answer or its citations are invalid, the problems are sent back to the model (at most max_attempts tries)."""
    hits = hybrid_search(corpus, llm, question, fusion, k)
    retrieved_ids = [chunk_id for chunk_id, _ in hits]
    if verbose:
        logger.info(f"[ask_rag] retrieved: {', '.join(retrieved_ids)}")
    prompt = build_prompt(question, [corpus.by_id[chunk_id] for chunk_id in retrieved_ids], rules)
    current_prompt = prompt
    last_problems = ""
    for attempt in range(1, max_attempts + 1):
        try:
            answer = llm.ask_structured(current_prompt, GroundedAnswer, verbose=verbose)
            problems = check_citations(answer, retrieved_ids)
            if problems:
                raise StructuredOutputError(answer.model_dump_json(), problems)
            if verbose:
                logger.info("[ask_rag] citations: OK")
            return answer
        except StructuredOutputError as error:
            last_problems = error.problems
            if verbose:
                logger.info(f"[ask_rag] attempt {attempt} failed:\n{error.problems}")
            current_prompt = (
                f"{prompt}\n\nYour previous answer was:\n{error.answer}\n\n"
                f"It was rejected for these reasons:\n{error.problems}\n\nGive a corrected answer."
            )
    raise RuntimeError(f"No valid answer after {max_attempts} attempts:\n{last_problems}")

