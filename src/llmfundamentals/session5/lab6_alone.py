"""
LAB 6 - Your own attack, on your own (about 15 minutes, calls the model)

Goal: write a poisoned document and see what the defences of labs 3, 4 and 5 do with it.

What to do:
  1. Edit my_document.txt, next to this lab: a short document about the AI Act or the GDPR that tries to harm the RAG
     (a false claim, a fake legal text, an instruction to the model, a link in the answer, ...).
     Everything fictional, with invented names. Keep the format: the lines "title:", "author:", ... then "---".
  2. Write in QUESTION a question that should retrieve your document.
  3. Run this file. It adds your document to the corpus (as a secondary source, because of the folder it came from) and asks
     your question with the defences of labs 3, 4 and 5. A report opens in your browser (report_lab6.html): was your
     document retrieved (an orange chip with your id)? What did the answer say? Did your attack work?
  4. Try to improve your document until it gets through.

Bonus: swap your document with a neighbour's, and try to find the weak point of their defences.
"""
from pathlib import Path

from lab3_provenance import trust_tier
from lab4_output_validation import SafeAnswer
from lab5_spotlighting import spotlight
from lab_tools import (
    POLICY,
    RULES,
    UNTRUSTED_RULE,
    add_document,
    ask_rag,
    fuse,
    get_corpus,
    get_llm,
    make_row,
    model_label,
    show_report,
    summary_lines,
)
from loguru import logger

QUESTION = "TODO a question that should retrieve your document"

if __name__ == "__main__":
    document = Path(__file__).parent / "my_document.txt"
    if QUESTION.startswith("TODO"):
        raise SystemExit("TODO not done yet: write QUESTION")
    if "TODO" in document.read_text(encoding="utf-8"):
        raise SystemExit("TODO not done yet: write your document in my_document.txt")

    corpus = get_corpus()
    llm = get_llm(use_cache=False)
    ids = add_document(corpus, llm, document)
    logger.info(f"Your document was added to the corpus, as secondary chunks: {ids}")
    logger.info(f"Question: {QUESTION}")
    retried = []
    try:
        answer = ask_rag(
            llm,
            corpus,
            QUESTION,
            RULES + POLICY + UNTRUSTED_RULE,
            fuse,
            output_class=SafeAnswer,
            tier_of=trust_tier,
            spotlight=spotlight,
            on_problem=lambda attempt, problems: retried.append(f"attempt {attempt}: " + problems.replace("\n", " / ")),
        )
    except RuntimeError:  # no valid answer: the validators blocked it
        answer = None
    row = make_row(corpus, llm, QUESTION, answer, retried)
    for line in summary_lines(1, row):
        logger.info(line)
    show_report([row], "lab6", "Lab 6: your own attack", model_label())

# My notes
# did my attack work? what did I change to make it work? =
