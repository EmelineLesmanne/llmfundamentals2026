"""
LAB 3 - Write a judge, and compare it with your labels (about 20 minutes)

Goal: use a model to check answers, and check the model against yourself. A judge is only useful if you know how often it
agrees with a human. In lab 1 you labelled the 13 answers of the application by hand (faithful? correct?). Now a model
does the same, and the kit shows where you and the model disagree.

STEP 1 - YOU WRITE THE JUDGE
  - Verdict: the Pydantic class of the judge's answer, with these three fields (the names are imposed, the types are yours):
        supported           true if every claim of the answer is supported by the chunks
        unsupported_claims  the claims that are not supported (an empty list if none)
        reason              a short explanation
  - faithfulness_prompt(answer, sources): the text sent to the judge. It must say what to check, and that the judge must use
    only the sources (not what it knows or what is true in general). answer and sources are texts.

STEP 2 - RUN
  Run this file. The judge (a model of another family than the application) judges the 13 answers for faithfulness, and a
  second judge written by the kit judges their correctness. A report opens: how often you and the judge agree, and the answers
  where you disagree. Who is right?

To use another judge, write its name after the file name: gpt-4.1 (default), sonnet, haiku, gpt-mini.
Bonus: change your prompt and run again. Does the agreement change?
"""
from lab_tools import (
    ask_judge,
    format_sources,
    get_corpus,
    get_judge,
    judge_correctness,
    load_golden,
    load_json,
    logger,
    read_labels,
    report_agreement,
    show_report,
)
from pydantic import BaseModel


class Verdict(BaseModel):
    # The structure of the judge's answer, as in session 3: one line per field, "name: type".
    # Hints:
    #   a true/false field has the type bool; a list of texts has the type list[str]; a text has the type str.
    #   The names must be exactly: supported, unsupported_claims, reason.
    #   The kit uses them to compute the verdict (supported and no unsupported claim = faithful) and to show the reason.
    pass  # TODO: the three fields (supported, unsupported_claims, reason)


def faithfulness_prompt(answer, sources):
    # answer: the text of the answer to check. sources: the chunks the application received, as one text
    # (one block per chunk, headed by its id in brackets).
    # Hints:
    #   An f-string puts the variables in the text: f"Sources:\n{sources}". "\n" is a line break.
    #   A good judge prompt says: what the judge is (its role), what to check, how to decide, and what to return.
    #   Say what "faithful" means: each claim of the answer is stated in at least one of the sources.
    #   Say that the judge must NOT use what it knows or what is true in general: only the sources count.
    #   Name the three fields to return (supported, unsupported_claims, reason) and what each one contains.
    # TODO: write the prompt of the judge. Return it as a text.
    return "TODO"


# Everything below only runs when you run this file directly (lab 5 imports Verdict and faithfulness_prompt).
if __name__ == "__main__":
    corpus = get_corpus()
    # Your labels from lab 1 (a dictionary: question id -> (faithful?, correct?)). Without them, nothing to compare.
    labels, problems = read_labels()
    if problems:
        raise SystemExit("Finish lab 1 first: your labels in my_labels.txt are not complete (or copy the labels shown on the display).")
    if not Verdict.model_fields or faithfulness_prompt("a", "b") == "TODO":
        raise SystemExit("STEP 1 not done yet: complete Verdict and faithfulness_prompt")

    judge = get_judge("gpt-4.1")  # (name, model): chosen by the word after the file name, gpt-4.1 by default
    answers = load_json("answers.json")  # the 13 answers of the application, frozen in advance (same for everybody)
    retrieval = load_json("retrieval.json")  # the chunks found by each search for each question, best first
    rows = []  # one dictionary per question, for the report
    for item in load_golden():
        answer = answers[item["id"]]
        sources = [corpus.by_id[chunk_id] for chunk_id in retrieval[item["id"]]["hybrid"][:5]]  # what the application received
        # Your judge: your prompt, and the answer must have the shape of your Verdict class. None if the judge answered badly.
        verdict = ask_judge(judge, faithfulness_prompt(answer["answer"], format_sources(sources)), Verdict)
        # The kit's judge for correctness (answer compared with the reference answer of the golden dataset).
        correct = judge_correctness(judge, item["question"], item["reference"], answer["answer"])
        rows.append({"item": item, "answer": answer["answer"], "verdict": verdict, "correct": correct})
        # One line per question: your label next to the judge's ("yes" or "NO"; "-" if the judge gave no valid answer).
        mine = labels[item["id"]]
        logger.info(f"{item['id']}: faithful {'yes' if mine[0] else 'NO'} (you) | judge: {'-' if verdict is None else ('yes' if verdict.supported and not verdict.unsupported_claims else 'NO')}"
                    f"   correct {'yes' if mine[1] else 'NO'} (you) | judge: {'-' if correct is None else ('yes' if correct.correct else 'NO')}")
    # The report counts the questions where you and the judge give the same label, and lists the others with the judge's reason.
    body, agree_faithful, agree_correct = report_agreement(rows, judge[0], labels)
    logger.info(f"You agree with the judge on {agree_faithful} of {len(rows)} (faithfulness) and {agree_correct} of {len(rows)} (correctness).")
    show_report("lab3", "Lab 3: you against the judge", body, f"agreement with {judge[0]}: faithfulness {agree_faithful}/{len(rows)}, correctness {agree_correct}/{len(rows)}", f"Judge: {judge[0]}")
