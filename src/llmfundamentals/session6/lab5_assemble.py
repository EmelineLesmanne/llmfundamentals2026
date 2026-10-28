"""
LAB 5 - Assemble the evaluation (about 25 minutes, one question is evaluated, a few model calls)

Goal: see the whole pipeline of an evaluation: its stages, in which order they run, and what each one measures.

An evaluation of a RAG goes through these stages, for each question of the golden dataset:
    1. search            the application's search finds chunks
    2. generate          the application answers, from those chunks
    3. recall            did the search find the gold articles?                        (your recall_at_k, lab 2)
    4. faithfulness      is the answer faithful to the chunks that the application received?  (your judge, lab 3)
    5. correctness       is the answer correct, compared with the reference answer?    (a judge written by the kit)
    6. diagnose          in which cell of the grid does this question fall?            (your diagnose, below)
    7. collect           keep the results of this question
and then, once, for all the questions: aggregate (means and counts) and report.

The grid (retrieval good: all the gold articles were found among the K best chunks):

                          answer good                    answer bad
    retrieval good        everything works               the search found it, the model failed (or was misled)
    retrieval bad         right by luck or memory        the search failed, and so did the answer

What to do:
  1. Complete diagnose (TODO): it receives two booleans, retrieval_ok and answer_ok, and returns the label of the cell:
     "good retrieval, good answer", "good retrieval, bad answer", "bad retrieval, good answer" or "bad retrieval, bad answer".
  2. Assemble the loop. The fragments are listed below, in no particular order. Each fragment ends with its label,
     like  # (m)  . Copy the fragments you keep into the loop, below the markers, in the right order:
       - between "# >>> EACH" and "# >>> AFTER": the fragments that run once for EACH question (indented by 4 spaces),
       - between "# >>> AFTER" and "# >>> END": the fragments that run ONCE, after the loop (no indentation).
     Remove the "#" at the start of the line you copy, and remove the line "pass".
     Some fragments are traps: they look right and contain a mistake in how an evaluation is done. Leave them out.
  3. Run this file. It first checks your assembly (without calling any model) and tells you what is wrong. Then it
     evaluates ONE question (QUESTIONS_TO_RUN = 1). A report opens in your browser. The instructor will run all 13
     (write "all" after the file name: about 3 minutes).

To use another judge, write its name after the file name (gpt-4.1 by default; sonnet, haiku, gpt-mini).
Bonus: run all 13 (about 3 minutes); change RETRIEVER to "bm25" or "dense".
"""
# ruff: noqa: F401
import sys

from lab2_recall import recall_at_k
from lab3_faithfulness import Verdict, faithfulness_prompt
from lab_tools import (
    ask_judge,
    build_grid,
    check_assembly,
    chunks_of,
    format_sources,
    fusion_for,
    generate,
    get_corpus,
    get_judge,
    get_weak_llm,
    judge_correctness,
    load_golden,
    logger,
    make_row,
    retrieve,
    show_evaluation,
    summarise,
)

RETRIEVER = "hybrid"  # "dense", "bm25" or "hybrid"
K = 5
QUESTIONS_TO_RUN = 13 if "all" in sys.argv else 1  # 1 to test your loop; write "all" after the file name for the 13 questions


def diagnose(retrieval_ok, answer_ok):
    if retrieval_ok and answer_ok:
        return "good retrieval, good answer"
    if retrieval_ok:
        return "good retrieval, bad answer"
    if answer_ok:
        return "bad retrieval, good answer"
    return "bad retrieval, bad answer"

if diagnose(True, True) == "TODO":
    raise SystemExit("TODO not done yet: complete diagnose")
if recall_at_k(["GDPR-Art17-2"], ["GDPR-Art17"], 1) != 1 or not Verdict.model_fields or faithfulness_prompt("a", "b") == "TODO":
    raise SystemExit("Finish labs 2 and 3 first (or copy recall_at_k, Verdict and faithfulness_prompt from the display).")
problems = check_assembly(__file__)
if problems:
    for problem in problems:
        logger.info(problem)
    raise SystemExit("The assembly is not right yet: read the lines above, fix, and run again.")

corpus, app, judge = get_corpus(), get_weak_llm(), get_judge("gpt-4.1")
fusion = fusion_for(RETRIEVER)
rows = []
for item in load_golden()[:QUESTIONS_TO_RUN]:
    # >>> EACH: the fragments that run once for each question go here (indented by 4 spaces)
    question = item["question"]  # (m) the question to ask
    retrieved = retrieve(corpus, app, question, fusion, K)  # (c) stage 1, the search: the best chunks found
    answer = generate(app, corpus, question, fusion, K)  # (t) stage 2: the application answers from those chunks
    recall = recall_at_k(retrieved, item["gold_articles"], K) if item["gold_articles"] else None  # (e) stage 3: recall at article level, None when there is no gold article
    sources = chunks_of(corpus, retrieved)  # (h) the chunks the application received: the faithfulness is checked against them
    verdict = ask_judge(judge, faithfulness_prompt(answer.answer, format_sources(sources)), Verdict)  # (q) stage 4: faithfulness, answer against the received chunks
    result = judge_correctness(judge, question, item["reference"], answer.answer)  # (a) stage 5: correctness, answer against the reference answer
    correct = result is not None and result.correct  # (r) turn the judge's result into True or False
    cell = diagnose(recall is None or recall >= 0.99, correct)  # (x) stage 6: retrieval good if every gold article was found
    rows.append(make_row(item, answer, retrieved, recall, verdict, correct, cell))  # (w) stage 7: keep the results of this question
# >>> AFTER: the fragments that run once, after the loop, go here (no indentation)
summary_lines = summarise(rows)  # (s) aggregate: means and counts over all the questions
grid = build_grid(rows)  # (g) group the questions by cell of the grid
show_evaluation(rows, summary_lines, grid, RETRIEVER, judge[0])  # (p) the report
# >>> END
