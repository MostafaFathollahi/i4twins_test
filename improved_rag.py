"""
improved_rag.py — improved retrieval over corpus.jsonl.

Improvements vs. baseline_rag.py:
  * Whole-record encoding WITH the title ("title — text") — restores the most
    discriminative tokens (equipment IDs) the baseline discarded. No fixed-size
    character windows (unnecessary: every record fits the model context; see
    analyze_corpus.py).
  * Hybrid retrieval: dense cosine + BM25 lexical, fused with Reciprocal Rank
    Fusion (RRF). Dense handles paraphrase/semantics; BM25 nails exact codes
    (E-207, DN65, BRG-4410). RRF needs no score normalization or tuned weights.
  * Abstention: if the top candidate's dense cosine (with a BM25 exact-match
    override) is below a calibrated threshold, return "not found in documents"
    instead of a fabricated best-of-a-bad-lot chunk.
  * Near-duplicate dedup with a NUMERIC GUARD, and conflicting-value flagging:
    semantically similar candidates whose numbers agree are collapsed; when the
    numbers differ, both are kept and a conflict is flagged.

Fully offline, CPU-only. Requires: sentence-transformers, rank_bm25, numpy.
"""
import json
import re
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CORPUS_PATH = "corpus.jsonl"
SEP = " — "  # joins title and text into one retrievable unit

# MiniLM abstention floor, calibrated on qa.jsonl: it sits in the gap between the
# cosine-separable unanswerable questions (<=0.654) and the lowest answerable
# question (0.686). Small-sample; see README §6 for the honesty caveat.
ABSTAIN_COSINE = 0.67

# Embedding models we support. E5 REQUIRES query:/passage: prefixes — omitting
# them silently degrades quality; GTE and MiniLM do not use prefixes.
# abstain_cosine is calibrated PER MODEL: cosine scales are not comparable across
# embedding models. E5/GTE compress similarities into a narrow high band where
# answerable and unanswerable questions overlap, so their floors sit high and buy
# little separation (see README §8) — MiniLM separates best on this corpus.
MODELS = {
    "all-MiniLM-L6-v2": {
        "hf": "sentence-transformers/all-MiniLM-L6-v2",
        "query_prefix": "", "passage_prefix": "", "abstain_cosine": ABSTAIN_COSINE,
    },
    "e5-small-v2": {
        "hf": "intfloat/e5-small-v2",
        "query_prefix": "query: ", "passage_prefix": "passage: ", "abstain_cosine": 0.837,
    },
    "gte-small": {
        "hf": "thenlper/gte-small",
        "query_prefix": "", "passage_prefix": "", "abstain_cosine": 0.859,
    },
}

RRF_K = 60          # standard RRF constant
DEDUP_SIM = 0.90    # cosine above which two docs are "the same topic"
ABSTAIN_COVERAGE = 0.8  # min query-term coverage to override a low semantic score

# Equipment/error codes like C-100, M-50, P-200, F-30, E-207, DN65, BRG-4410.
_CODE = re.compile(r"\b[a-z]{1,4}-?\d{2,4}\b", re.IGNORECASE)
# Function words stripped before measuring lexical coverage of a query.
_STOP = set("what is the of a an to for be should how are was were does do did in on "
            "at and or with which from into per each when where why who a".split())


def equipment_codes(text):
    return {m.group(0).lower().replace("-", "") for m in _CODE.finditer(text)}


def lexical_coverage(query, doc_text):
    """Fraction of the query's content words that appear in the doc (1.0 if none)."""
    q_terms = [t for t in tokenize(query) if t not in _STOP and not t.isdigit()]
    if not q_terms:
        return 1.0
    doc_terms = set(tokenize(doc_text))
    return sum(1 for t in q_terms if t in doc_terms) / len(q_terms)


def load_docs(path=CORPUS_PATH):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def tokenize(text):
    """Lexical tokens for BM25: lowercase alphanumerics, keep codes like e-207, dn65."""
    return re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower())


# Numbers with an optional unit token immediately after (e.g. "16 bar", "4.5 mm/s",
# "40 m3/h"). Used by the dedup numeric guard and conflict flag.
_NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*([a-z][a-z0-9/]*)?", re.IGNORECASE)

# Physical units that actually appear in this corpus. The conflict detector
# only compares these, so the word that happens to follow a bare number
# (e.g. "2000 operating", "207 indicates") is never mistaken for a unit.
KNOWN_UNITS = {"bar", "kw", "m3/h", "m3/min", "mm/s", "degrees",
               "celsius", "hours", "months", "v", "hz", "rms", "kg"}


def numeric_pairs(text):
    """Set of (value, following-token) pairs. Used by the dedup numeric guard."""
    pairs = set()
    for value, unit in _NUM_UNIT.findall(text):
        pairs.add((value, (unit or "").lower()))
    return pairs


def unit_values(text):
    """Map real physical unit -> set of values found with it (e.g. {'bar': {'16'}})."""
    out = {}
    for value, unit in numeric_pairs(text):
        if unit in KNOWN_UNITS:
            out.setdefault(unit, set()).add(value)
    return out


@dataclass
class Retrieved:
    doc_id: str
    title: str
    text: str
    rrf_score: float
    cosine: float
    bm25: float


@dataclass
class Answer:
    abstained: bool
    results: list = field(default_factory=list)  # list[Retrieved]
    conflict: bool = False
    conflict_note: str = ""
    text: str = ""


class ImprovedRetriever:
    def __init__(self, model_name="all-MiniLM-L6-v2", abstain_cosine=None):
        if model_name not in MODELS:
            raise ValueError(f"Unknown model {model_name}; choose from {list(MODELS)}")
        self.cfg = MODELS[model_name]
        self.model_name = model_name
        self.model = SentenceTransformer(self.cfg["hf"])
        self.abstain_cosine = (self.cfg["abstain_cosine"] if abstain_cosine is None
                               else abstain_cosine)
        self.docs = []
        self.vectors = None
        self.bm25 = None

    # ---- indexing -------------------------------------------------------
    def build(self, docs):
        self.docs = docs
        passages = [self.cfg["passage_prefix"] + d["title"] + SEP + d["text"] for d in docs]
        vecs = self.model.encode(passages, convert_to_numpy=True).astype("float32")
        self.vectors = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.bm25 = BM25Okapi([tokenize(d["title"] + " " + d["text"]) for d in docs])
        return self

    # ---- channels -------------------------------------------------------
    def _dense(self, query):
        q = self.model.encode([self.cfg["query_prefix"] + query], convert_to_numpy=True)[0]
        q = q.astype("float32")
        q /= np.linalg.norm(q)
        return self.vectors @ q  # cosine per doc, shape (n_docs,)

    def _bm25(self, query):
        scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype="float32")
        return scores

    @staticmethod
    def _ranks(scores):
        """Map doc index -> 0-based rank (0 = best) by descending score."""
        order = np.argsort(-scores)
        rank = np.empty(len(scores), dtype=int)
        for r, idx in enumerate(order):
            rank[idx] = r
        return rank

    # ---- fused ranking (metrics use THIS, before dedup) -----------------
    def rank(self, query, k=None):
        cos = self._dense(query)
        bm = self._bm25(query)
        dr, br = self._ranks(cos), self._ranks(bm)
        rrf = 1.0 / (RRF_K + dr) + 1.0 / (RRF_K + br)
        bm_norm = bm / bm.max() if bm.max() > 0 else bm
        order = np.argsort(-rrf)
        out = [
            Retrieved(
                doc_id=self.docs[i]["id"], title=self.docs[i]["title"],
                text=self.docs[i]["text"], rrf_score=float(rrf[i]),
                cosine=float(cos[i]), bm25=float(bm_norm[i]),
            )
            for i in order
        ]
        return out[:k] if k else out

    # ---- dedup + conflict (presentation layer, used by answer()) --------
    def _dedup(self, ranked, id_to_idx):
        """Drop near-duplicates: high semantic similarity AND identical numbers.
        Different numbers => NOT a duplicate (it may be a conflict) — the numeric
        guard keeps the DOC-01/DOC-02 pressure disagreement from being merged."""
        kept = []
        for cand in ranked:
            dup = False
            for keep in kept:
                sim = float(
                    self.vectors[id_to_idx[cand.doc_id]] @ self.vectors[id_to_idx[keep.doc_id]]
                )
                if sim >= DEDUP_SIM and numeric_pairs(cand.text) == numeric_pairs(keep.text):
                    dup = True
                    break
            if not dup:
                kept.append(cand)
        return kept

    def _detect_conflict(self, kept):
        """Flag conflicting values only among docs about the SAME entity (shared
        equipment code) and only for real physical units."""
        top = kept[0]
        top_codes = equipment_codes(top.title + " " + top.text)
        if not top_codes:
            return False, "", [top]
        peers = [top] + [r for r in kept[1:]
                         if equipment_codes(r.title + " " + r.text) & top_codes]
        agg = {}
        for r in peers:
            for unit, values in unit_values(r.text).items():
                agg.setdefault(unit, set()).update(values)
        conflicting = {u: v for u, v in agg.items() if len(v) > 1}
        if conflicting:
            note = "; ".join(f"{u}={'/'.join(sorted(v))}" for u, v in conflicting.items())
            return True, note, peers
        return False, "", [top]

    # ---- product behavior: abstention + dedup + conflict ----------------
    def _abstain(self, query, top):
        """Grounding gate. Abstain when the corpus does not support the query.

        1. Code grounding: if the query names an equipment/error code that the
           best doc does not contain, abstain (catches near-misses like the
           C-200 vs C-100 adversarial case).
        2. Semantic floor: if the top dense cosine is high enough, answer.
        3. Coverage override: below the floor, answer only if the doc lexically
           grounds (nearly) all query content words — this rescues terse but
           valid keyword queries whose absolute cosine is low.
        """
        doc_text = top.title + " " + top.text
        q_codes = equipment_codes(query)
        if q_codes and not (q_codes & equipment_codes(doc_text)):
            return True
        if top.cosine >= self.abstain_cosine:
            return False
        return lexical_coverage(query, doc_text) < ABSTAIN_COVERAGE

    def answer(self, query, k=3):
        ranked = self.rank(query)
        id_to_idx = {d["id"]: i for i, d in enumerate(self.docs)}
        top = ranked[0]
        if self._abstain(query, top):
            return Answer(abstained=True, results=[],
                          text="Not found in the documents.")
        kept = self._dedup(ranked[:6], id_to_idx)[:k]
        conflict, note, peers = self._detect_conflict(kept)
        if conflict:
            body = " || ".join(f"[{r.doc_id}] {r.text}" for r in peers)
            text = f"[CONFLICTING VALUES — {note}] {body}"
            return Answer(abstained=False, results=peers, conflict=True,
                          conflict_note=note, text=text)
        top = kept[0]
        return Answer(abstained=False, results=[top], text=f"[{top.doc_id}] {top.text}")


def calibrate_threshold(retriever, qa_path="qa.jsonl"):
    """Report top-cosine for answerable vs unanswerable questions.

    Honest note: with ~15 questions we cannot hold out a clean split, so we
    inspect the separation and pick a threshold in the gap. This is illustrative,
    not a generalization claim (README §6).
    """
    qa = [json.loads(line) for line in open(qa_path, encoding="utf-8") if line.strip()]
    ans, un = [], []
    for q in qa:
        top = retriever.rank(q["question"])[0]
        (ans if q["answerable"] else un).append(top.cosine)
    print("answerable top-cosine:  ", [round(x, 3) for x in sorted(ans)])
    print("unanswerable top-cosine:", [round(x, 3) for x in sorted(un)])
    if ans and un:
        print(f"min(answerable)={min(ans):.3f}  max(unanswerable)={max(un):.3f}")


if __name__ == "__main__":
    docs = load_docs()
    r = ImprovedRetriever().build(docs)
    for q in [
        "What is the rated output of the C-100 compressor?",
        "What is the maximum operating pressure of the P-200 pump?",  # conflict
        "What is the acceptable vibration limit for F-30 fans?",       # duplicate
        "What is the warranty period of the M-50 motor?",              # unanswerable
    ]:
        a = r.answer(q)
        print("Q:", q)
        print("A:", a.text)
        print("-" * 70)
