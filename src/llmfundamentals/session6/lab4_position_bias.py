"""
LAB 4 - Position bias (about 15 minutes, calls a judge model)

Goal: check whether a judge is fair. A judge that compares two answers should choose the same one whether it
is shown first or second.

The kit gives you 13 pairs of real answers to the same question: the answer of the strong model and the answer of
the weak model. judge_pair(judge, question, first, second) asks the judge which is better.
  - It sends the judge a prompt that shows the two answers under the names "Answer A" (the one you pass as `first`) and
    "Answer B" (the one you pass as `second`), and asks which is better.
  - It returns "A" if the judge prefers the first answer shown, "B" if it prefers the second.
The letter is a POSITION, not an answer: the same answer is A when shown first and B when shown second.
You must call it twice for each pair: with the strong answer first, then with the weak answer first.

What to do:
  1. Complete the loop (TODOs): the second call, and the decision "did the judge flip?". The judge flips when it does not
     choose the same answer after the swap. Hint: in the first call the strong answer is shown first, so it wins if the
     result is "A"; in the second call the weak answer is shown first, so the strong answer is B and wins if the result is "B".
  2. Run this file (about two minutes: 26 calls, spaced by a few seconds). A report opens in your browser.
  3. How many pairs flip? Is the judge biased towards the first answer or the second?

To use another judge, write its name after the file name: haiku (default), gpt-4.1, sonnet, gpt-mini.
Bonus: run it with another judge (see above) and compare the number of flips.
"""
from lab_tools import (
    get_judge,
    judge_pair,
    load_golden,
    load_json,
    logger,
    report_flips,
    show_report,
)

judge = get_judge("haiku")  # (name, model): chosen by the word after the file name, haiku by default
pairs = load_json("pairs.json")  # for each question id: the answer of the strong model and the answer of the weak model
rows = []  # one dictionary per pair, for the report
flips = 0  # the number of pairs where the judge changes its mind when the order is swapped
first_chosen = 0  # how many times the judge chose the answer shown first (a judge with no position bias: about half)
calls = 0  # the number of verdicts counted
for item in load_golden():
    pair = pairs[item["id"]]
    if pair["weak"] is None or pair["strong"] is None:
        continue  # one of the models produced no valid answer
    strong, weak = pair["strong"]["answer"], pair["weak"]["answer"]
    # judge_pair(judge, question, first, second) returns "A" if the judge prefers the answer given as `first`,
    # "B" if it prefers the one given as `second` (None if the judge gave no valid answer: not handled here).
    winner_1 = judge_pair(judge, item["question"], strong, weak)  # the strong answer is shown first (A)
    #   A judge without position bias picks the same answer in both orders.
    #   A comparison such as (winner_1 == "A") gives True or False.
    # TODO: ask the judge again, with the weak answer shown first
    winner_2 = judge_pair(judge, item["question"], weak, strong)
    # TODO: flipped is True if the judge did not choose the same answer in the two calls, False otherwise
    flipped = (winner_1 == "A") != (winner_2 == "B")

    # Stops with a message while the TODOs are not done.
    if winner_2 is None or flipped is None:
        logger.info("TODO not done yet: complete the loop")
        break
    calls += 2
    first_chosen += (winner_1 == "A") + (winner_2 == "A")  # True counts as 1: how often the first answer was chosen
    flips += flipped  # True counts as 1
    logger.info(f"{item['id']}: {'FLIP' if flipped else 'stable'}")
    rows.append({"question": item["question"], "winner_1": winner_1, "winner_2": winner_2, "flipped": flipped})

# The report: for each pair, the two verdicts and whether the judge flipped; and whether it tends to choose the first answer.
if rows:
    logger.info(f"{flips} of {len(rows)} pairs flip when the order is swapped (judge: {judge[0]})")
    body = report_flips(rows, flips, len(rows), judge[0], first_chosen, calls)
    summary = f"{flips} of {len(rows)} pairs flip ({judge[0]})"
    show_report("lab4", "Lab 4: is the judge fair?", body, summary, f"Judge: {judge[0]}")
