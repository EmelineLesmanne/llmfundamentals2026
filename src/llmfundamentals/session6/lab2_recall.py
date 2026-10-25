"""
LAB 2 - Recall@k

Goal: measure whether the search finds the right articles, with a number.

The golden dataset has 13 questions. For each one we know the gold articles: the articles that contain the answer
(for example ["AIAct-Art27", "GDPR-Art35"]). The kit gives you, for each question, the chunks found by three searches
(dense, BM25, hybrid), best first.

DOCUMENTATION

  recall_at_k(ranked_ids, gold_articles, k)
      ranked_ids     the ids of the chunks found by a search, best first, for example ["GDPR-Art17-2", "GDPR-Art5-1", ...]
      gold_articles  the gold articles of the question, for example ["GDPR-Art17"]
      k              only the k best chunks count
      returns        the share of the gold articles that appear among the k best chunks (between 0 and 1)

  article_of(chunk_id)   the article of a chunk id:  article_of("GDPR-Art17-2") is "GDPR-Art17"

  Example: gold articles ["AIAct-Art27", "GDPR-Art35"], and the 5 best chunks belong to AIAct-Art27 (four of them) and
  to nothing else that is gold: the recall is 0.5 (one gold article out of two was found).

What to do:
  1. Complete recall_at_k (TODO).
  2. Run this file. It computes the recall of the three searches on all the questions that have an answer, and a report
     opens in your browser (report_lab2.html).
  3. Which search is the best? Is the difference big?

Bonus: change K (3, 10) and look at what happens to the means.
"""
from lab_tools import (
    article_of,  # noqa: F401  (used once you complete the TODO)
    load_golden,
    load_json,
    logger,
    report_recall,
    show_report,
)

K = 5  # only the K best chunks of each search count


def recall_at_k(ranked_ids, gold_articles, k):
    # Hints:
    #   ranked_ids[:k] is the list of the k best chunk ids (a slice: the first k elements).
    #   article_of(chunk_id) tells which article a chunk belongs to: article_of("GDPR-Art17-2") is "GDPR-Art17".
    #   A gold article is "found" if at least one of the k best chunks belongs to it (compare with ==).
    #   Count each gold article at most once, even if several chunks belong to it (a `break` stops a loop early).
    #   Do not forget to divide by the number of gold articles: len(gold_articles).
    # TODO: count the gold articles found among the k best chunks, and divide by the number of gold articles
    return 0.0


# Everything below only runs when you run this file directly (lab 5 imports recall_at_k).
if __name__ == "__main__":
    # A quick test of your function: the only chunk found belongs to the only gold article, so the recall must be 1.
    if recall_at_k(["GDPR-Art17-2"], ["GDPR-Art17"], 1) != 1:
        raise SystemExit("TODO not done yet: complete recall_at_k")

    # retrieval.json was computed in advance: for each question id, the chunk ids found by each search, best first.
    # For example retrieval["g06"]["dense"] is the list of chunk ids found by the dense search for question g06.
    retrieval = load_json("retrieval.json")
    rows = []  # one (item, recalls) pair per question, for the report
    totals = {"dense": 0, "bm25": 0, "hybrid": 0}  # the sum of the recalls of each search, to compute its mean
    for item in load_golden():
        if not item["gold_articles"]:
            continue  # the unanswerable question has no gold article
        recalls = {}
        for search in ["dense", "bm25", "hybrid"]:
            # The recall of this search for this question, with your function.
            recalls[search] = recall_at_k(retrieval[item["id"]][search], item["gold_articles"], K)
            totals[search] += recalls[search]
        rows.append((item, recalls))
    # The mean recall of each search over the questions: this is the number used to compare the three searches.
    means = {}
    for search, total in totals.items():
        means[search] = total / len(rows)
    logger.info(f"Mean recall@{K} on {len(rows)} questions: dense {means['dense']:.2f} | BM25 {means['bm25']:.2f} | hybrid {means['hybrid']:.2f}")
    # The report: one line per question with the recall of each search (green = everything found, red = nothing found).
    summary = f"recall@{K}: dense {means['dense']:.2f}, BM25 {means['bm25']:.2f}, hybrid {means['hybrid']:.2f}"
    show_report("lab2", "Lab 2: recall of the three searches", report_recall(rows, means, K), summary)
