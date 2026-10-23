"""
LAB 3 - Provenance (about 10 minutes, calls the model)

Goal: decide how much to trust a source from where it came from, never from what it says about itself.

Look at the fake "Article 22a" of lab 1: it is written as "OFFICIAL CONSOLIDATED TEXT". A document can claim anything.
But we know where each file came from: when the files were loaded, every chunk got a "source", the folder and the file
it came from (the dictionary of a chunk has the keys "id", "text", "source", ...):
    "official/gdpr.pdf"                          "secondary/newsletter_automated_recruitment.txt"

What to do:
  1. Complete trust_tier: it receives a chunk (a dictionary) and returns "official" or "secondary",
     decided by the source of the chunk and nothing else.
  2. Run this file. First it logs the tier given to the fake Article 22a. Then it asks the five questions of lab 1 again,
     with two changes made by the kit: the tier of each source is shown to the model, with a policy
     ("a statement about the law needs an official source..."), and the official chunks come first.
  3. Open the report (report_lab3.html, it opens in your browser) and compare with lab 1: which answers changed?
     The scoreboard at the bottom of the report keeps the scores of your labs.

Bonus: what happens to the hidden instruction of question 4? Does provenance stop it?
"""
from lab_tools import POLICY, RULES, get_corpus, get_llm, run_questions
from loguru import logger


def trust_tier(chunk):
    """Return "official" or "secondary" for a chunk."""
    # TODO: look at chunk["source"] (not at the text of the chunk!) and return the right tier
    if chunk["source"].startswith("official"):
        return "official"
    return "secondary"



# Everything below only runs when you run this file directly (lab 6 imports trust_tier).
if __name__ == "__main__":
    corpus = get_corpus()
    fake_article = corpus.by_id["SEC-newsletter_automated_recruitment-1"]
    logger.info(f"Tier of the fake Article 22a: {trust_tier(fake_article)}")
    logger.info(f"Tier of GDPR Article 22(1): {trust_tier(corpus.by_id['GDPR-Art22-1'])}")
    if trust_tier(fake_article) is None:
        raise SystemExit("TODO not done yet: complete trust_tier")

    llm = get_llm(use_cache=False)
    run_questions(llm, corpus, "lab3", "Lab 3: provenance", rules=RULES + POLICY, tier_of=trust_tier)

