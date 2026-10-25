"""Small helpers shared by the labs of session 6. You do not need to read or change this file."""
import json
import os
import sys
import time
import webbrowser
from html import escape
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from loguru import logger
from pydantic import BaseModel

# The application that we evaluate answers with a weak open-weight model (the RAG of sessions 4 and 5).
WEAK_MODEL = "mistralai/mistral-nemo"
# The judges: models of other families than the application. Write the name of a judge after the file name to change it,
# for example:  uv run python src/llmfundamentals/session6/lab3_faithfulness.py sonnet
JUDGES = {
    "gpt-4.1": ("openai", "openai/gpt-4.1"),
    "haiku": ("anthropic", "anthropic/claude-haiku-4.5"),
    "gpt-mini": ("openai", "openai/gpt-4.1-mini"),
    "sonnet": ("anthropic", "anthropic/claude-sonnet-5.5"),
}
PAUSE = 1.0  # seconds between two judge calls (paid keys are not limited to 20 calls per minute; the wrapper retries on a 429)

DATA = Path(__file__).resolve().parent / "data"
SESSION4_DATA = Path(__file__).resolve().parents[1] / "session4" / "data"
SESSION5_DATA = Path(__file__).resolve().parents[1] / "session5" / "data"

# makes `from common.llmwrapper import ...` work when you run a lab file
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
load_dotenv()  # reads OPENROUTER_API_KEY from the .env file

from common.llmwrapper import LLMWrapper, StructuredOutputError
from common.rag import GroundedAnswer, ask_rag, hybrid_search, load_corpus

# Show only the message (no time, no level) and hide the debug lines, so every screen looks the same.
logger.remove()
logger.add(sys.stderr, format="{message}", level="INFO")


# ---------------------------------------------------------------------------------------------- data
def load_json(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_golden():
    """The golden dataset: a list of items with id, type, question, reference, gold_articles and expected."""
    return load_json("golden.json")["items"]


def article_of(chunk_id):
    """The article (or annex) of a chunk id: "GDPR-Art17-2" -> "GDPR-Art17"."""
    return "-".join(chunk_id.split("-")[:2])


def get_corpus():
    """The corpus of session 5: the official texts plus the secondary documents. Each chunk has a "source"."""
    corpus = load_corpus(SESSION4_DATA, extra=SESSION5_DATA)
    for chunk in corpus.chunks:
        if "source" not in chunk:
            chunk["source"] = "official/" + {"AI Act": "ai-act.pdf", "GDPR": "gdpr.pdf"}[chunk["regulation"]]
    corpus.query_vectors.update(load_json("query_embeddings.json"))  # the questions of this session: no embedding call
    return corpus


def format_sources(chunks):
    """The sources, as text: one block per chunk, with its id."""
    return "\n\n".join(f"[{chunk['id']}]\n{chunk['text']}" for chunk in chunks)


# ---------------------------------------------------------------------------------------------- models
def get_weak_llm(use_cache=False):
    return LLMWrapper("openai", model=WEAK_MODEL, use_cache=use_cache)


def get_judge(default):
    """The judge of the lab: (name, model). The name written after the file name replaces the default judge."""
    name = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in JUDGES else default
    provider, model = JUDGES[name]
    logger.info(f"[the judge is {name}]")
    return name, LLMWrapper(provider, model=model)


_last_call = [0.0]


def ask_judge(judge, prompt, output_class):
    """Ask the judge for an answer of type output_class (a Pydantic class). Returns None if the judge answers in the
    wrong format. Calls are spaced by PAUSE seconds."""
    wait = PAUSE - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    try:
        return judge[1].ask_structured(prompt, output_class)
    except StructuredOutputError:
        return None


class PairVerdict(BaseModel):
    winner: Literal["A", "B"]
    reason: str


def judge_pair(judge, question, first, second):
    """Which of two answers is better, according to the judge? Returns "A" (the first), "B" (the second) or None."""
    prompt = (
        "You compare two answers to a question about EU law.\n\n"
        f"Question: {question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}\n\n"
        "Which answer is better? Give the winner (A or B) and a short reason."
    )
    verdict = ask_judge(judge, prompt, PairVerdict)
    return verdict.winner if verdict else None


class Correctness(BaseModel):
    correct: bool
    reason: str


def judge_correctness(judge, question, reference, answer):
    """Is the answer correct, compared with the reference answer? Returns a Correctness, or None."""
    prompt = (
        f"Question: {question}\n\nReference answer (it is true):\n{reference}\n\nAnswer to evaluate:\n{answer}\n\n"
        "Is the answer correct? It is correct if it agrees with the reference answer on the facts that matter "
        "(dates, numbers, yes or no) and does not contradict it. Give a short reason."
    )
    return ask_judge(judge, prompt, Correctness)


# ---------------------------------------------------------------------------------------------- reports
CSS = """
:root { --bg:#fafaf7; --fg:#1d1d1b; --card:#fff; --line:#d8d6cf; --good:#1f7a3d; --bad:#b3261e; --neutral:#6b6b66; --sec:#c25a00; }
@media (prefers-color-scheme: dark) { :root { --bg:#161614; --fg:#ecebe6; --card:#201f1d; --line:#3a3936; --good:#5fc27e; --bad:#f2837c; --neutral:#a3a29c; --sec:#ffa14a; } }
body { background:var(--bg); color:var(--fg); font:16px/1.5 system-ui, sans-serif; margin:0 auto; max-width:64rem; padding:1rem 16px 3rem; }
h1 { font-size:1.4rem; } h2 { font-size:1.05rem; margin:.2rem 0 .6rem; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:.8rem 1rem; margin:1rem 0; }
.badge { display:inline-block; font-weight:700; border:2px solid; border-radius:6px; padding:0 .5rem; text-transform:uppercase; font-size:.85rem; margin-right:.4rem; }
.badge.good { color:var(--good); } .badge.bad { color:var(--bad); } .badge.neutral { color:var(--neutral); }
.chip { display:inline-block; border:1px solid var(--line); border-radius:4px; padding:0 .4rem; margin:0 .25rem .25rem 0; font:.8rem ui-monospace, monospace; }
.chip.gold { border-color:var(--good); color:var(--good); font-weight:700; } .chip.sec { border-color:var(--sec); color:var(--sec); }
table { border-collapse:collapse; margin:.5rem 0; } td, th { border:1px solid var(--line); padding:.2rem .7rem; text-align:left; vertical-align:top; }
td.good { background:color-mix(in srgb, var(--good) 22%, transparent); } td.bad { background:color-mix(in srgb, var(--bad) 22%, transparent); }
td.mid { background:color-mix(in srgb, var(--sec) 22%, transparent); }
.muted { color:var(--neutral); font-size:.9rem; } .truth { border-left:4px solid var(--good); padding-left:.6rem; }
"""


def badge(text, kind):
    return f'<span class="badge {kind}">{escape(text)}</span>'


def chips(chunk_ids, gold_articles=()):
    """Chunk ids as small labels: green when the chunk is in a gold article, orange when it is a secondary source."""
    out = ""
    for chunk_id in chunk_ids:
        kind = "gold" if article_of(chunk_id) in gold_articles else ("sec" if chunk_id.startswith("SEC-") else "")
        out += f'<span class="chip {kind}">{escape(chunk_id)}</span>'
    return out


def text(value):
    return escape(value or "").replace("\n", "<br>")


def show_report(lab, title, body, summary, note=""):
    """Write report_<lab>.html next to the labs, with the scoreboard of the labs already run, and open it in the browser."""
    kit = Path(__file__).resolve().parent
    scores_file = kit / "scores.json"
    scores = json.loads(scores_file.read_text(encoding="utf-8")) if scores_file.exists() else {}
    scores[lab] = summary
    scores_file.write_text(json.dumps(scores), encoding="utf-8")
    board = "".join(f"<tr><td>{escape(name)}</td><td>{escape(score)}</td></tr>" for name, score in sorted(scores.items()))
    page = (
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{escape(title)}</title>'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><style>{CSS}</style></head><body>'
        f"<h1>{escape(title)}</h1><p class=\"muted\">{escape(note)}</p>{body}"
        f"<h2>Scoreboard</h2><table><tr><th>Lab</th><th>Result</th></tr>{board}</table></body></html>"
    )
    path = kit / f"report_{lab}.html"
    path.write_text(page, encoding="utf-8")
    logger.info(f"The full report is in {path.name}, next to the labs: it opens in your browser (open the file if it does not).")
    if not os.environ.get("NO_BROWSER"):
        webbrowser.open(path.as_uri())


def cell(value):
    """A table cell coloured by a score between 0 and 1."""
    kind = "good" if value >= 0.99 else ("bad" if value <= 0.01 else "mid")
    return f'<td class="{kind}">{value:.2f}</td>'


# ---------------------------------------------------------------------------------------------- lab 2
def report_recall(rows, means, k):
    """rows: [(item, {"dense": r, "bm25": r, "hybrid": r})] for the answerable items; means: {"dense": m, ...}."""
    body = f"<div class=\"card\"><h2>Recall@{k} of each retriever</h2><table><tr><th>Question</th><th>Type</th><th>Gold articles</th><th>dense</th><th>BM25</th><th>hybrid</th></tr>"
    for item, recalls in rows:
        body += (
            f"<tr><td>{escape(item['question'])}</td><td>{escape(item['type'])}</td><td>{escape(', '.join(item['gold_articles']))}</td>"
            f"{cell(recalls['dense'])}{cell(recalls['bm25'])}{cell(recalls['hybrid'])}</tr>"
        )
    body += f"<tr><th colspan=\"3\">Mean over the {len(rows)} answerable questions</th>{cell(means['dense'])}{cell(means['bm25'])}{cell(means['hybrid'])}</tr></table>"
    body += '<p class="muted">Recall@k: the share of the gold articles that appear among the k best chunks found. Green = all found, red = none.</p></div>'
    return body


# ---------------------------------------------------------------------------------------------- lab 3
def report_faithfulness(rows, judge_name):
    """rows: [{"item", "answer", "sources", "verdict"}]; verdict is a Verdict (supported, unsupported_claims, reason) or None."""
    body = ""
    for row in rows:
        item, verdict = row["item"], row["verdict"]
        if verdict is None:
            tag = badge("no verdict", "neutral")
        elif verdict.supported and not verdict.unsupported_claims:
            tag = badge("faithful", "good")
        else:
            tag = badge("unfaithful", "bad")
        claims = ""
        if verdict is not None and verdict.unsupported_claims:
            claims = "<p><b>Claims the judge found unsupported:</b><br>" + "<br>".join(escape(c) for c in verdict.unsupported_claims) + "</p>"
        why = f'<p class="muted"><b>Reason:</b> {escape(verdict.reason)}</p>' if verdict else ""
        body += (
            f'<section class="card"><h2>{escape(item["question"])}</h2>{tag}'
            f'<p><b>Answer:</b><br>{text(row["answer"])}</p>{claims}{why}'
            f'<p><b>Sources given to the application:</b> {chips([c["id"] for c in row["sources"]], item["gold_articles"])}</p></section>'
        )
    return body


# ---------------------------------------------------------------------------------------------- lab 4
def report_flips(rows, flips, total, judge_name, first_chosen, calls):
    body = (
        f'<div class="card"><h2>{flips} of {total} pairs flip when the order of the answers is swapped</h2>'
        f'<p>Judge: {escape(judge_name)}. The first answer was chosen {first_chosen} times out of {calls}.</p>'
        "<table><tr><th>Question</th><th>Strong answer first: winner</th><th>Weak answer first: winner</th><th></th></tr>"
    )
    for row in rows:
        w1, w2 = row["winner_1"], row["winner_2"]
        tag = badge("flip", "bad") if row["flipped"] else badge("stable", "good")
        shown1 = {"A": "the strong answer", "B": "the weak answer"}.get(w1, "-")
        shown2 = {"A": "the weak answer", "B": "the strong answer"}.get(w2, "-")
        body += f"<tr><td>{escape(row['question'])}</td><td>{shown1}</td><td>{shown2}</td><td>{tag}</td></tr>"
    body += "</table><p class=\"muted\">A judge that is not biased by the position of the answers gives the same winner in both orders.</p></div>"
    return body


# ---------------------------------------------------------------------------------------------- lab 1
def report_examples(examples):
    """examples: [{"title", "item", "answer", "citations", "retrieved"}]"""
    body = '<p>Two real answers of the application. Which one is the better answer? Be ready to say why.</p>'
    for example in examples:
        item = example["item"]
        body += (
            f'<section class="card"><h2>{escape(example["title"])}: {escape(item["question"])}</h2>'
            f'<p class="truth"><b>Reference answer:</b> {text(item["reference"])}</p>'
            f'<p><b>The answer of the application:</b><br>{text(example["answer"])}</p>'
            f'<p><b>Cited:</b> {chips(example["citations"], item["gold_articles"])}</p>'
            f'<p><b>Chunks found by the search (green = in a gold article, orange = secondary document):</b><br>{chips(example["retrieved"], item["gold_articles"])}</p>'
            f'<p class="muted">Gold articles: {escape(", ".join(item["gold_articles"]))}</p></section>'
        )
    return body


# ---------------------------------------------------------------------------------------------- lab 5
def report_evaluation(rows, summary_lines, grid, retriever, judge_name):
    """rows: one dict per item (item, answer, recall, faithful, correct, cell); grid: {cell label: [questions]}."""
    body = f'<div class="card"><h2>Summary</h2><p>Retriever: {escape(retriever)} · judge: {escape(judge_name)}</p><ul>'
    body += "".join(f"<li>{escape(line)}</li>" for line in summary_lines) + "</ul></div>"
    body += '<div class="card"><h2>Diagnosis grid</h2><table><tr><th>Cell</th><th>Questions</th></tr>'
    for label, questions in grid.items():
        body += f"<tr><td>{escape(label)}</td><td>{'<br>'.join(escape(q) for q in questions) or '-'}</td></tr>"
    body += "</table></div>"
    for row in rows:
        item = row["item"]
        faithful = {True: badge("faithful", "good"), False: badge("unfaithful", "bad"), None: badge("not judged", "neutral")}[row["faithful"]]
        correct = {True: badge("correct", "good"), False: badge("wrong", "bad"), None: badge("not judged", "neutral")}[row["correct"]]
        recall = "" if row["recall"] is None else badge(f"recall {row['recall']:.2f}", "good" if row["recall"] >= 0.99 else "bad")
        body += (
            f'<section class="card"><h2>{escape(item["question"])}</h2>{recall}{faithful}{correct}'
            f'<p class="truth"><b>Reference:</b> {text(item["reference"])}</p><p><b>Answer:</b><br>{text(row["answer"])}</p>'
            f'<p><b>Cell of the grid:</b> {escape(row["cell"])}</p>'
            f'<p><b>Retrieved:</b> {chips(row["retrieved"], item["gold_articles"])}</p></section>'
        )
    return body


# ---------------------------------------------------------------------------------------------- the application
# The RAG of sessions 4 and 5: hybrid search and the rules of session 4.
def fuse(rankings):
    """Reciprocal Rank Fusion (session 4, lab 3)."""
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank)
    return scores


RULES = """Answer the question using only the sources below.
- Write the answer in a few sentences.
- Cite the id of every source you rely on, in "citations". Copy each id exactly as it is written in the id of the
  source tag (for example AIAct-Art113-3), without adding anything to it. Never cite an id that is not in the sources.
- If the sources do not contain the answer, set "answerable" to false, leave "citations" empty and say so in "answer".
- Do not use your own knowledge: the sources are more recent than what you know."""


def fusion_for(retriever):
    """A fusion function for hybrid_search that keeps only one search: "dense", "bm25", or both ("hybrid")."""
    if retriever == "dense":
        return lambda rankings: fuse([rankings[0]])
    if retriever == "bm25":
        return lambda rankings: fuse([rankings[1]])
    return fuse


# ---------------------------------------------------------------------------------------------- lab 3: your labels
LABELS_FILE = Path(__file__).resolve().parent / "my_labels.txt"
FAITHFUL_WORDS = {"faithful": True, "unfaithful": False}
CORRECT_WORDS = {"correct": True, "incorrect": False}


def read_labels():
    """The labels written by the student in my_labels.txt: ({id: (faithful, correct)}, [problems])."""
    labels, problems = {}, []
    known = [item["id"] for item in load_golden()]
    for line in LABELS_FILE.read_text(encoding="utf-8").splitlines():
        words = line.replace(":", " ").replace(",", " ").split()
        if not words or words[0].startswith("#"):
            continue
        if words[0] not in known or len(words) != 3:
            problems.append(f"I do not understand this line: {line!r}")
        elif words[1] not in FAITHFUL_WORDS or words[2] not in CORRECT_WORDS:
            problems.append(f"{words[0]}: write faithful or unfaithful, then correct or incorrect (not {words[1]!r} {words[2]!r})")
        else:
            labels[words[0]] = (FAITHFUL_WORDS[words[1]], CORRECT_WORDS[words[2]])
    for item_id in known:
        if item_id not in labels and not any(p.startswith(item_id) for p in problems):
            problems.append(f"{item_id}: no label yet")
    return labels, problems


def report_dataset(items):
    """The golden dataset as a table, and the number of questions of each type."""
    body = (
        '<div class="card"><h2>The golden dataset</h2><p>Each item has: an <b>id</b>, a <b>type</b>, a <b>question</b>, a '
        "<b>reference answer</b> (what a good answer says), the <b>gold articles</b> (the articles that contain the answer) and "
        "the <b>expected</b> behaviour (answer, or refuse because the corpus cannot answer).</p><table><tr><th>Id</th><th>Type</th>"
        "<th>Question</th><th>Reference answer</th><th>Gold articles</th><th>Expected</th></tr>"
    )
    for item in items:
        body += (
            f"<tr><td>{escape(item['id'])}</td><td>{escape(item['type'])}</td><td>{escape(item['question'])}</td>"
            f"<td>{text(item['reference'])}</td><td>{escape(', '.join(item['gold_articles']) or '-')}</td><td>{escape(item['expected'])}</td></tr>"
        )
    counts = {}
    for item in items:
        counts[item["type"]] = counts.get(item["type"], 0) + 1
    body += "</table><p><b>Questions by type:</b> " + ", ".join(f"{escape(name)}: {number}" for name, number in counts.items()) + "</p></div>"
    return body


def labelling_cards(corpus):
    """The cards of the labelling sheet: for each question, the reference, the answer and the chunks the application received."""
    answers, retrieval = load_json("answers.json"), load_json("retrieval.json")
    body = (
        '<div class="card"><h2>How to label the answers</h2><ul>'
        "<li><b>faithful</b>: every claim of the answer is stated in at least one of the chunks the application received "
        "(cited or not, even if another chunk disagrees, even if the chunk is wrong). <b>unfaithful</b>: a claim comes from nowhere.</li>"
        "<li><b>correct</b>: the answer gives the key facts of the reference answer and does not contradict it. "
        "<b>incorrect</b>: a key fact is missing or wrong.</li></ul>"
        "<p>Write one line per question in <code>my_labels.txt</code>, for example <code>g01  faithful  incorrect</code>.</p></div>"
    )
    for item in load_golden():
        answer = answers[item["id"]]
        retrieved = retrieval[item["id"]]["hybrid"][:5]
        cited = [chunk_id for chunk_id in retrieved if chunk_id in (answer["citations"] if answer else [])]
        chunks = ""
        for chunk_id in cited + [chunk_id for chunk_id in retrieved if chunk_id not in cited]:
            style, badge = ("", " <b>cited</b>") if chunk_id in cited else (' style="opacity:.55"', "")
            chunks += f'<p{style}><span class="chip">{escape(chunk_id)}</span>{badge} {text(corpus.by_id[chunk_id]["text"][:700])}</p>'
        body += (
            f'<section class="card"><h2>{escape(item["id"])}: {escape(item["question"])}</h2>'
            f'<p class="truth"><b>Reference answer:</b> {text(item["reference"])}</p>'
            f'<p><b>Answer of the application:</b><br>{text(answer["answer"] if answer else "(no valid answer)")}</p>'
            f"<p><b>Chunks the application received</b> (the ones the answer cites come first; the others are greyed but still count):</p>{chunks}</section>"
        )
    return body


def report_agreement(rows, judge_name, labels):
    """rows: {"item", "answer", "verdict" (Verdict or None), "correct" (Correctness or None)}; labels: the student's labels."""
    agree_faithful, agree_correct, disagreements = 0, 0, ""
    for row in rows:
        item, verdict, result = row["item"], row["verdict"], row["correct"]
        mine_faithful, mine_correct = labels[item["id"]]
        judge_faithful = None if verdict is None else verdict.supported and not verdict.unsupported_claims
        judge_correct = None if result is None else result.correct
        same_faithful, same_correct = judge_faithful == mine_faithful, judge_correct == mine_correct
        agree_faithful += same_faithful
        agree_correct += same_correct
        if same_faithful and same_correct:
            continue
        lines = ""
        for name, mine, judged, same, why in (
            ("faithful", mine_faithful, judge_faithful, same_faithful, verdict.reason if verdict else ""),
            ("correct", mine_correct, judge_correct, same_correct, result.reason if result else ""),
        ):
            if not same:
                shown = {True: name, False: "un" + name if name == "faithful" else "incorrect", None: "no verdict"}
                lines += (
                    f"<p>{badge('disagree: ' + name, 'bad')} you: <b>{shown[mine]}</b> · judge: <b>{shown[judged]}</b><br>"
                    f'<span class="muted">The judge says: {escape(why)}</span></p>'
                )
        disagreements += f'<section class="card"><h2>{escape(item["question"])}</h2><p>{text(row["answer"])}</p>{lines}</section>'
    total = len(rows)
    head = (
        f'<div class="card"><h2>Your labels against the judge ({escape(judge_name)})</h2>'
        f"<p>Faithfulness: you and the judge agree on <b>{agree_faithful} of {total}</b> answers.<br>"
        f"Correctness: you and the judge agree on <b>{agree_correct} of {total}</b> answers.</p>"
        "<p class=\"muted\">Below, the answers where you disagree. Who is right: you or the judge?</p></div>"
    )
    return head + (disagreements or '<p class="card">You and the judge agree on everything.</p>'), agree_faithful, agree_correct


# ---------------------------------------------------------------------------------------------- lab 5: the evaluation loop
def retrieve(corpus, app, question, fusion, k):
    """The ids of the k best chunks found by the search."""
    hits = hybrid_search(corpus, app, question, fusion, k=k)
    ids = []
    for chunk_id, _ in hits:
        ids.append(chunk_id)
    return ids


def generate(app, corpus, question, fusion, k):
    """The answer of the application (a GroundedAnswer). If no valid answer could be produced: an answer that says so."""
    try:
        return ask_rag(app, corpus, question, RULES, fusion, k=k)
    except RuntimeError:
        return GroundedAnswer(answer="NO VALID ANSWER", citations=[], answerable=True)


def chunks_of(corpus, chunk_ids):
    """The chunks (dictionaries) of a list of chunk ids."""
    chunks = []
    for chunk_id in chunk_ids:
        chunks.append(corpus.by_id[chunk_id])
    return chunks


def make_row(item, answer, retrieved, recall, verdict, correct, cell):
    """Everything the report shows about one question."""
    faithful = None if verdict is None else verdict.supported and not verdict.unsupported_claims
    return {"item": item, "answer": answer.answer, "retrieved": retrieved, "recall": recall, "faithful": faithful, "correct": correct, "cell": cell}


def summarise(rows):
    """The numbers of the evaluation, as lines of text."""
    recalls = [row["recall"] for row in rows if row["recall"] is not None]
    lines = [f"questions evaluated: {len(rows)}"]
    if recalls:
        lines.append(f"mean recall: {sum(recalls) / len(recalls):.2f}")
    lines.append(f"faithful answers: {sum(1 for row in rows if row['faithful'])} of {len(rows)}")
    lines.append(f"correct answers: {sum(1 for row in rows if row['correct'])} of {len(rows)}")
    return lines


def build_grid(rows):
    """{cell label: [questions]}"""
    grid = {}
    for row in rows:
        grid.setdefault(row["cell"], []).append(row["item"]["question"])
    return grid


def show_evaluation(rows, summary_lines, grid, retriever, judge_name):
    for line in summary_lines:
        logger.info(line)
    for label, questions in grid.items():
        logger.info(f"{label}: {len(questions)}")
    summary = f"{retriever}, {len(rows)} question(s): " + "; ".join(summary_lines[1:])
    show_report("lab5", "Lab 5: evaluation of the RAG", report_evaluation(rows, summary_lines, grid, retriever, judge_name), summary, f"Judge: {judge_name}")


# Each fragment of lab 5: what it needs (uses), what it creates (defines), where it goes (loop = once per question,
# after = once, after the loop, wrong = a plausible line that contains a mistake), and why.
FRAGMENTS = {
    "m": {"scope": "loop", "uses": [], "defines": ["question"], "why": ""},
    "c": {"scope": "loop", "uses": ["question"], "defines": ["retrieved"], "why": ""},
    "t": {"scope": "loop", "uses": ["question"], "defines": ["answer"], "why": ""},
    "e": {"scope": "loop", "uses": ["retrieved"], "defines": ["recall"], "why": ""},
    "h": {"scope": "loop", "uses": ["retrieved"], "defines": ["sources"], "why": ""},
    "q": {"scope": "loop", "uses": ["answer", "sources"], "defines": ["verdict"], "why": ""},
    "a": {"scope": "loop", "uses": ["question", "answer"], "defines": ["result"], "why": ""},
    "r": {"scope": "loop", "uses": ["result"], "defines": ["correct"], "why": ""},
    "x": {"scope": "loop", "uses": ["recall", "correct"], "defines": ["cell"], "why": ""},
    "w": {"scope": "loop", "uses": ["answer", "retrieved", "recall", "verdict", "correct", "cell"], "defines": [], "why": ""},
    "s": {"scope": "after", "uses": [], "defines": ["summary_lines"], "why": ""},
    "g": {"scope": "after", "uses": [], "defines": ["grid"], "why": ""},
    "p": {"scope": "after", "uses": ["summary_lines", "grid"], "defines": [], "why": ""},
    "j": {"scope": "wrong", "why": "it checks the faithfulness against the reference answer: faithfulness is checked against the chunks the application received"},
    "z": {"scope": "wrong", "why": "it compares chunk ids with article ids: they are never equal (the recall is always 0); recall is computed on articles"},
    "l": {"scope": "wrong", "why": "it gives the reference answer to the application: the application must not see the answer it is evaluated on"},
    "n": {"scope": "wrong", "why": "it gives the gold articles to the search: the search must not see what it is evaluated on"},
}
BASE_NAMES = ["item", "rows", "corpus", "app", "judge", "fusion", "K"]  # exist before the lines of the student


def check_assembly(path):
    """Check the loop assembled by the student, without running it. Returns a list of problems (empty if all is right)."""
    import re

    lines = Path(path).read_text(encoding="utf-8").splitlines()
    marks = {}
    for number, line in enumerate(lines):
        for name in ("EACH", "AFTER", "END"):
            if line.strip().startswith(f"# >>> {name}"):
                marks[name] = number
    if len(marks) < 3:
        return ["I cannot find the markers '# >>> EACH...', '# >>> AFTER...' and '# >>> END' of the file: do not delete them."]
    found, problems, defined = [], [], list(BASE_NAMES)
    if not any(re.search(r"#\s*\((\w+)\)", line) for line in lines[marks["EACH"] + 1 : marks["END"]]):
        return ["Nothing assembled yet: copy the fragments you keep between the markers '# >>> EACH...' and '# >>> END'."]
    regions = [("EACH", marks["EACH"] + 1, marks["AFTER"], 4, "loop"), ("AFTER", marks["AFTER"] + 1, marks["END"], 0, "after")]
    for _, start, end, indent, scope in regions:
        for line in lines[start:end]:
            code = line.strip()
            if not code or code.startswith("#") or code == "pass":
                continue
            tag = re.search(r"#\s*\((\w+)\)\s*(.*)$", line)
            if tag is None or tag.group(1) not in FRAGMENTS:
                problems.append(f"This line is not one of the fragments: {code[:60]!r}")
                continue
            name, comment = tag.group(1), tag.group(2).strip()
            info = FRAGMENTS[name]
            found.append(name)
            if info["scope"] == "wrong":
                problems.append(f"({name}) is a distractor, remove it: {info['why']}.")
                continue
            if info["scope"] != scope:
                where = "once for each question (inside the loop, indented by 4 spaces)" if info["scope"] == "loop" else "once, after the loop (no indentation)"
                problems.append(f"({name}) is in the wrong place: it belongs {where}.")
                continue
            if len(line) - len(line.lstrip()) != indent:
                problems.append(f"({name}) has the wrong indentation: {indent} spaces are expected here.")
            if not comment:
                problems.append(f"({name}) has no comment: add, after the label, one line that says what this line measures or why it is there.")
            for used in info["uses"]:
                if used not in defined:
                    problems.append(f"({name}) uses `{used}`, which is not created yet: a line that creates it must come before.")
            defined.extend(info["defines"])
    for name, info in FRAGMENTS.items():
        if info["scope"] != "wrong" and name not in found:
            problems.append(f"({name}) is missing.")
    for name in set(found):
        if found.count(name) > 1:
            problems.append(f"({name}) appears twice.")
    return problems
