"""
eval.py — retrieval + abstention evaluation over qa.jsonl.

Reports, before vs after:
  * Retrieval (answerable questions, abstention gate OFF): Hit@1, Recall@3, MRR,
    computed on the raw ranking so downstream dedup never distorts them.
    Multi-gold labels (list of acceptable doc_ids) make Recall meaningful.
  * Abstention (all questions): correct-abstention rate on unanswerable,
    false-abstention rate on answerable.

Usage:
  python eval.py --system baseline
  python eval.py --system improved [--model all-MiniLM-L6-v2|e5-small-v2|gte-small]
"""
import argparse
import json

import numpy as np

import baseline_rag as base
from improved_rag import ImprovedRetriever, Retrieved, load_docs

K_RECALL = 3


# --------------------------------------------------------------------------
# Baseline adapter: reuse the baseline's own index/scoring, but expose a full
# doc-level ranking (baseline_rag only ever returns top-1) so we can measure it
# fairly. The baseline has NO abstention, so it never abstains — by design.
# --------------------------------------------------------------------------
class BaselineSystem:
    name = "baseline"

    def __init__(self):
        self.model = base.SentenceTransformer(base.EMBED_MODEL)

    def build(self, docs):
        self.chunks, self.vectors = base.build_index(docs, self.model)
        return self

    def rank(self, query):
        q = self.model.encode([query])[0].astype("float32")
        q /= np.linalg.norm(q)
        sims = self.vectors @ q
        # Collapse chunks to best score per doc, then order by score.
        best = {}
        for c, s in zip(self.chunks, sims):
            if c["doc_id"] not in best or s > best[c["doc_id"]][1]:
                best[c["doc_id"]] = (c, float(s))
        ordered = sorted(best.values(), key=lambda cs: -cs[1])
        return [Retrieved(c["doc_id"], c["title"], c["text"], s, s, 0.0) for c, s in ordered]

    def answer(self, query, k=3):
        # Baseline always returns its top-1; it cannot abstain.
        top = self.rank(query)[0]
        return type("A", (), {"abstained": False, "results": [top]})()


class ImprovedSystem:
    def __init__(self, model_name):
        self.name = f"improved({model_name})"
        self.retriever = ImprovedRetriever(model_name=model_name)

    def build(self, docs):
        self.retriever.build(docs)
        return self

    def rank(self, query):
        return self.retriever.rank(query)

    def answer(self, query, k=3):
        return self.retriever.answer(query, k=k)


def load_qa(path="qa.jsonl"):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def evaluate(system, qa):
    answerable = [q for q in qa if q["answerable"]]
    unanswerable = [q for q in qa if not q["answerable"]]

    hit1 = recall3 = mrr = 0.0
    per_q = []
    by_style = {}  # style -> [hits, count]
    for q in answerable:
        ranked = system.rank(q["question"])
        ids = [r.doc_id for r in ranked]
        gold = set(q["gold_doc_ids"])
        h1 = 1.0 if ids[0] in gold else 0.0
        rec = len(set(ids[:K_RECALL]) & gold) / len(gold)
        rr = next((1.0 / (i + 1) for i, d in enumerate(ids) if d in gold), 0.0)
        hit1 += h1; recall3 += rec; mrr += rr
        style = q.get("style", "nl")
        s = by_style.setdefault(style, [0.0, 0])
        s[0] += h1; s[1] += 1
        per_q.append((q["id"], style, ids[0], h1, rec, round(rr, 3)))
    n = len(answerable)
    hit1, recall3, mrr = hit1 / n, recall3 / n, mrr / n

    # Abstention
    false_abstain = sum(1 for q in answerable if system.answer(q["question"]).abstained)
    correct_abstain = sum(1 for q in unanswerable if system.answer(q["question"]).abstained)

    return {
        "hit1": hit1, "recall3": recall3, "mrr": mrr,
        "correct_abstain": correct_abstain, "n_unans": len(unanswerable),
        "false_abstain": false_abstain, "n_ans": n,
        "by_style": by_style, "per_q": per_q,
    }


def print_report(name, m):
    print(f"\n=== {name} ===")
    print(f"  Retrieval (answerable, n={m['n_ans']}):")
    print(f"    Hit@1    : {m['hit1']:.3f}")
    print(f"    Recall@{K_RECALL} : {m['recall3']:.3f}")
    print(f"    MRR      : {m['mrr']:.3f}")
    print(f"    Hit@1 by query style: " +
          ", ".join(f"{st}={h/c:.3f} (n={c})" for st, (h, c) in sorted(m["by_style"].items())))
    print(f"  Abstention:")
    print(f"    Correct abstention (unanswerable): {m['correct_abstain']}/{m['n_unans']}")
    print(f"    False   abstention (answerable)  : {m['false_abstain']}/{m['n_ans']}")
    print(f"  Per-question (id, style, top1, hit1, recall@{K_RECALL}, rr):")
    for row in m["per_q"]:
        print(f"    {row}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["baseline", "improved"], default="improved")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    docs = load_docs()
    qa = load_qa()
    if args.system == "baseline":
        system = BaselineSystem().build(docs)
        name = "baseline_rag.py"
    else:
        system = ImprovedSystem(args.model).build(docs)
        name = f"improved_rag.py [{args.model}]"

    print_report(name, evaluate(system, qa))


if __name__ == "__main__":
    main()
