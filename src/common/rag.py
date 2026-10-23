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


def read_chunks(folder, suffix=""):
    """The chunks and embeddings of one folder: chunks{suffix}.jsonl, embeddings{suffix}.npy, embedding_model.txt."""
    folder = Path(folder)
    lines = (folder / f"chunks{suffix}.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = [json.loads(line) for line in lines]
    vectors = np.load(folder / f"embeddings{suffix}.npy")  # one normalised row per chunk
    model = (folder / "embedding_model.txt").read_text(encoding="utf-8").strip()
    query_file = folder / "query_embeddings.json"
    query_vectors = json.loads(query_file.read_text(encoding="utf-8")) if query_file.exists() else {}
    return chunks, vectors, model, query_vectors


def load_corpus(folder, suffix="", extra=None):
    """Load a folder (see read_chunks) and build the two search indexes. If extra is the path of a second folder,
    its chunks are added to the first one: it must use the same embedding model."""
    chunks, vectors, model, query_vectors = read_chunks(folder, suffix)
    if extra is not None:
        more_chunks, more_vectors, more_model, more_query_vectors = read_chunks(extra)
        assert more_model == model, f"the two folders use different embedding models: {model} and {more_model}"
        chunks = chunks + more_chunks
        vectors = np.vstack([vectors, more_vectors])
        query_vectors = {**query_vectors, **more_query_vectors}
    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product of normalised vectors = cosine similarity
    index.add(vectors)
    bm25 = BM25Okapi([tokenize(f"{c['article']} {c['title']} {c['text']}") for c in chunks])
    return Corpus(chunks, {c["id"]: c for c in chunks}, index, bm25, query_vectors, model)


def read_document(path):
    """(metadata, paragraphs) of a text file: "key: value" lines, a line "---", then paragraphs separated by blank lines."""
    header, body = Path(path).read_text(encoding="utf-8").split("\n---\n", 1)
    metadata = {}
    for line in header.splitlines():
        if not line.startswith("#"):  # a first line starting with # is a label, not data
            key, value = line.split(": ", 1)
            metadata[key] = value
    paragraphs, heading = [], ""
    for paragraph in (p.strip() for p in body.split("\n\n") if p.strip()):
        if "." not in paragraph and len(paragraph) < 120:  # a heading: it goes with the paragraph that follows
            heading += paragraph + "\n"
        else:
            paragraphs.append(heading + paragraph)
            heading = ""
    return metadata, paragraphs


def document_chunks(path, folder):
    """The chunks of a document file. The source of a chunk is where the file came from (folder/file name):
    it is decided here, when the file is loaded, never read from the text of the document."""
    path = Path(path)
    metadata, paragraphs = read_document(path)
    return [
        {
            "id": f"SEC-{path.stem}-{number}",
            "regulation": metadata["publisher"],  # described by who published it, like any source: not by its category
            "article": metadata["type"],
            "title": metadata["title"],
            "paragraph": str(number),
            "text": text,
            "source": f"{folder}/{path.name}",
            "author": metadata["author"],
            "publisher": metadata["publisher"],
            "date": metadata["date"],
        }
        for number, text in enumerate(paragraphs, start=1)
    ]


def embedding_text(chunk):
    """What the embedding model reads: the place of the chunk, then its text."""
    return f"{chunk['regulation']}, {chunk['article']}: {chunk['title']}\n{chunk['text']}"


def add_chunks(corpus, chunks, vectors):
    """Add chunks (with their normalised embeddings, one row each) to a loaded corpus."""
    corpus.index.add(np.array(vectors, dtype="float32"))
    corpus.chunks.extend(chunks)
    corpus.by_id.update({chunk["id"]: chunk for chunk in chunks})
    corpus.bm25 = BM25Okapi([tokenize(f"{c['article']} {c['title']} {c['text']}") for c in corpus.chunks])


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


def build_prompt(question, chunks, rules, source_text=None):
    """The rules, then the sources inside tags, then the question.
    A chunk may have a "tier" (official or secondary): it then appears in the tag of its source.
    source_text, if given, is a function chunk -> the text shown inside the tag (default: the text of the chunk)."""
    sources = ""
    for chunk in chunks:
        tier = f' tier="{chunk["tier"]}"' if "tier" in chunk else ""
        text = source_text(chunk) if source_text else chunk["text"]
        sources += (
            f'<source id="{chunk["id"]}" regulation="{chunk["regulation"]}" article="{chunk["article"]}" '
            f'title="{chunk["title"]}"{tier}>\n{text}\n</source>\n'
        )
    return f"{rules}\n\n<sources>\n{sources}</sources>\n\nQuestion: {question}"


def check_citations(answer, retrieved_ids):
    """What is wrong with the citations of a GroundedAnswer, one problem per line (empty text if nothing).
    Blank citations (some weaker models write [""] for "no citation") count as no citation."""
    citations = [chunk_id for chunk_id in answer.citations if chunk_id.strip()]
    problems = []
    if answer.answerable and not citations:
        problems.append("citations: an answerable question needs at least one citation")
    for chunk_id in citations:
        if chunk_id not in retrieved_ids:
            problems.append(f"citations: {chunk_id!r} is not one of the sources")
    return "\n".join(problems)


def ask_rag(
    llm,
    corpus,
    question,
    rules,
    fusion,
    k=5,
    max_attempts=3,
    verbose=False,
    output_class=GroundedAnswer,
    tier_of=None,
    spotlight=None,
    on_problem=None,
    check=True,
):
    """Retrieve the k best chunks (hybrid search with `fusion`), ask for an answer of type output_class
    (GroundedAnswer by default), check its citations in code.
    If the answer or its citations are invalid, the problems are sent back to the model (at most max_attempts tries).

    Optional defences:
      tier_of    a function chunk -> "official" or "secondary". The tier is shown in the prompt and the official
                 chunks come first (the order of the search is kept inside each tier).
      spotlight  a function text -> text, applied to the text of the secondary chunks in the prompt.
      on_problem a function (attempt number, problems) called each time an answer is rejected.
      check      False: do not check the citations (to see what a model writes whatever it cites)."""
    hits = hybrid_search(corpus, llm, question, fusion, k)
    chunks = [corpus.by_id[chunk_id] for chunk_id, _ in hits]
    if tier_of is not None:
        chunks = [{**chunk, "tier": tier_of(chunk)} for chunk in chunks]
        chunks.sort(key=lambda chunk: chunk["tier"] != "official")  # sort() is stable: official first, same order inside
    retrieved_ids = [chunk["id"] for chunk in chunks]
    if verbose:
        logger.info(f"[ask_rag] retrieved: {', '.join(retrieved_ids)}")

    def source_text(chunk):
        if spotlight is not None and chunk.get("tier") == "secondary":
            return spotlight(chunk["text"])
        return chunk["text"]

    prompt = build_prompt(question, chunks, rules, source_text)
    current_prompt = prompt
    last_problems = ""
    for attempt in range(1, max_attempts + 1):
        try:
            answer = llm.ask_structured(current_prompt, output_class, verbose=verbose)
            problems = check_citations(answer, retrieved_ids) if check else ""
            if problems:
                raise StructuredOutputError(answer.model_dump_json(), problems)
            if verbose:
                logger.info("[ask_rag] citations: OK")
            return answer
        except StructuredOutputError as error:
            last_problems = error.problems
            if on_problem is not None:
                on_problem(attempt, error.problems)
            if verbose:
                logger.info(f"[ask_rag] attempt {attempt} failed:\n{error.problems}")
            current_prompt = (
                f"{prompt}\n\nYour previous answer was:\n{error.answer}\n\n"
                f"It was rejected for these reasons:\n{error.problems}\n\nGive a corrected answer."
            )
    raise RuntimeError(f"No valid answer after {max_attempts} attempts:\n{last_problems}")
