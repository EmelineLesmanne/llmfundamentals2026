"""
LAB 1 - The problem (about 10 minutes, calls the model)

Goal: see what poisoned documents do to a RAG that trusts every source.

Besides the official texts (the AI Act and the GDPR), the corpus now holds six short documents written by other people:
commentaries, FAQs, blog posts, newsletters. We call them "secondary". Their chunk ids start with SEC-.
Some of them are wrong, or harmful. You have five questions. This RAG is the one of session 4: it trusts all the sources.
Its model is a small open-weight model (the labs of this session use it by default). To try the strong model, write
"strong" after the file name:  uv run python src/llmfundamentals/session5/lab1_the_problem.py strong

What to do:
  1. Complete the TODO: ask each question with ask_rag, as in session 4:
         ask_rag(llm, corpus, question, RULES, fuse)
     (RULES and fuse come from lab_tools.)
  2. Run this file. The terminal shows, for each question, a verdict and the start of the answer. A report opens in your
     browser (report_lab1.html): for each question, the answer, what the official texts say, and the sources that were
     retrieved (the orange ones are secondary).
  3. Look at the report, and be ready to say out loud: which answer surprised you the most, and why?

Bonus: open the files of data/secondary/ and find, for the answer you chose, what the document says.
"""
from lab_tools import (
    QUESTIONS,
    RULES,  # noqa: F401  (used once you complete the TODO)
    ask_rag,  # noqa: F401  (used once you complete the TODO)
    fuse,  # noqa: F401  (used once you complete the TODO)
    get_corpus,
    get_llm,
    make_row,
    model_label,
    show_report,
    summary_lines,
)
from loguru import logger

corpus = get_corpus()
llm = get_llm(use_cache=False)

rows = []
for number, question in enumerate(QUESTIONS, start=1):
    # TODO: replace None by the call to ask_rag
    answer = ask_rag(llm, corpus, question, RULES, fuse)

    if answer is None:
        logger.info("TODO not done yet: call ask_rag")
        break
    row = make_row(corpus, llm, question, answer)  # collects what the report shows
    rows.append(row)
    for line in summary_lines(number, row):
        logger.info(line)

if rows:
    show_report(rows, "lab1", "Lab 1: a RAG that trusts every source", model_label())
