# i4Twins — Retrieval Pipeline: Diagnosis & Improvement

Document-grounded retrieval over 16 short industrial technical documents
(`corpus.jsonl`). This repo diagnoses a weak baseline (`baseline_rag.py`) and
replaces its retrieval stage with a stronger, **fully offline, CPU-only**
pipeline, plus an evaluation harness that quantifies the improvement and the
system's ability to **abstain** when an answer is not in the corpus.

> Scope, per the task: the focus is **retrieval quality, abstention, and
> evaluation**. There is no LLM answer-generation layer — the "answer" is the
> retrieved evidence (doc id + text). No UI, auth, or deployment.

---

## 1. Setup & run

Offline, CPU-only. Python 3.12 recommended (PyTorch has no 3.14 wheels yet).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reproduce everything with one script:

```bash
./run_eval.sh            # runs corpus analysis + baseline + improved, prints all metric tables
```

Or run pieces individually:

```bash
python analyze_corpus.py           # corpus length stats -> chunking decision
python eval.py --system baseline   # metrics for baseline_rag.py
python eval.py --system improved                       # improved, MiniLM (baseline's model)
python eval.py --system improved --model e5-small-v2   # improved, alternative embedder
```

**Models** (all local, CPU): `all-MiniLM-L6-v2` (baseline embedder) and, for the
§8 embedder comparison, `intfloat/e5-small-v2` / `thenlper/gte-small`. Lexical
retrieval uses `rank_bm25` (pure Python). The **first** `run_eval.sh` downloads
these models once (~300 MB total); every run afterwards is offline. The offline
constraint is about *runtime* — no cloud API is ever called. To prove a fully
air-gapped run after prefetching, force offline mode:

```bash
HF_HUB_OFFLINE=1 ./run_eval.sh
```

---

## 2. Diagnosis — why the baseline is weak

Read `baseline_rag.py`. The retrieval stage has the following defects, ordered
by impact on this corpus:

| # | Defect | Where | Effect |
|---|--------|-------|--------|
| 1 | **No abstention.** Always returns `argmax(sims)`; no score floor. | `retrieve`, `answer` | For any out-of-corpus question it returns a confident, wrong chunk. Fabrication by construction — the single biggest gap vs. the task's key requirement. |
| 2 | **Top-1 only.** Returns exactly one chunk; no top-k, no re-ranking. | `retrieve` | Cannot recover when the best chunk is wrong, and cannot surface conflicts or aggregate evidence. |
| 3 | **Title is never embedded.** Only `chunk["text"]` is encoded. | `build_index` | Titles carry the strongest signal — equipment IDs like `C-100`, `M-50`, `BRG-4410`, `E-207` live there. Discarding them throws away the most discriminative tokens. |
| 4 | **Pure dense retrieval; no lexical channel.** | `retrieve` | Sentence embeddings blur exact alphanumeric codes (`E-207`, `DN65`, `BRG-4410`) and can confuse `C-100` vs. `M-50`. A keyword channel is needed for exact-match queries. |
| 5 | **No duplicate or conflict handling.** | — | Near-duplicate docs (DOC-05/DOC-06) both occupy top-k slots; conflicting specs (DOC-01 vs DOC-02) are silently resolved to one value. |
| 6 | **Fixed 400-char chunking (latent, not active here).** | `chunk_text` | Character windows split mid-word/mid-sentence on *longer* inputs. **On this corpus it is inert** — every record's text is ≤ 342 chars, under the 400-char window, so it already yields one chunk per doc. We still replace it, because it is a correctness landmine on real (longer) documents and because embedding-with-title (defect 3) belongs in the same stage. |

Being precise about defect 6 matters: a naïve reading blames "bad chunking," but
the measured corpus (§3) shows chunking is currently a no-op. The *active* wins
come from abstention, top-k + hybrid retrieval, title inclusion, and
dedup/conflict handling.

---

## 3. Corpus analysis → chunking decision

`python analyze_corpus.py` (reproducible). Measured over the 16 records:

- **text length:** min 214 / max 342 / avg 270 characters
- **title + text:** min 244 / max 375 characters
- **longest record in wordpieces:** 86 tokens (exact, `all-MiniLM-L6-v2` tokenizer)
- **`all-MiniLM-L6-v2` `max_seq_length`:** 256 wordpieces

Every record fits comfortably in the context window, so **we embed each record
whole as `title — text`** — no splitting, no overlap. This is the simplest
policy that (a) never truncates, (b) restores the title signal, and (c) keeps
one retrievable unit per document, which makes doc-level evaluation and
conflict/duplicate reasoning clean.

---

## 4. Data-quality issues & policies

The corpus is deliberately imperfect. Issues identified and the explicit policy
for each:

| Issue | Location | Policy |
|-------|----------|--------|
| **Conflicting spec.** P-200 max operating pressure is **16 bar** (DOC-01, spec sheet) vs **12 bar** (DOC-02, maintenance sheet). | DOC-01 / DOC-02 | **Do not silently pick one.** When top-k contains high-similarity chunks with *conflicting numeric values* for the same attribute, return **both** with citations and flag the conflict. (See §5, numeric guard.) |
| **Near-duplicate content.** F-30 vibration limit (4.5 mm/s RMS) is stated twice as paraphrases. | DOC-05 / DOC-06 | **Deduplicate at retrieval time** by semantic similarity of retrieved candidates, but only when their numeric content agrees (paraphrase, not conflict). Keep the higher-ranked one; the eval labels both as acceptable gold. |
| **Cross-doc redundancy.** Pump service interval (2000 h) appears in DOC-02 and DOC-12; both are consistent. | DOC-02 / DOC-12 | Treated as legitimately multi-relevant — `qa.jsonl` uses **multi-gold** labels so either satisfies the question. |
| **Inconsistent unit surface forms** (`m3/h`, `m3/min`, `degrees Celsius`). | throughout | Left as-is; the hybrid retriever tokenizes them without normalization. Documented as a known limitation. |

Why **not** hash-based near-dup detection (SimHash/MinHash): DOC-05/DOC-06 are
**paraphrases with low lexical overlap**, exactly the case where shingle hashing
fails to flag them. Semantic similarity catches them; a **numeric guard**
prevents it from wrongly merging the DOC-01/DOC-02 *conflict* (similar text,
different numbers → not a duplicate). One rule serves both dedup and conflict.

---

## 5. Retrieval improvements (`improved_rag.py`)

1. **Whole-record encoding with title.** Embed `title — text` per document (§3).
2. **Hybrid retrieval.** Dense cosine (sentence embeddings) **+ BM25** lexical.
   The dense channel captures paraphrase/semantics; BM25 nails exact equipment
   codes and units.
3. **Reciprocal Rank Fusion (RRF)** to combine the two rankings. RRF is
   generally more robust than weighted score averaging: it needs **no score
   normalization or hand-tuned weight**, and is insensitive to the differing,
   non-comparable score scales of cosine vs BM25. → top-k candidates.
4. **Dedup + numeric guard.** Collapse near-duplicate candidates by semantic
   similarity *when their numbers agree*; when numbers conflict, keep both and
   **flag a conflict** instead of merging.
5. **Abstention — a three-part grounding gate.** Return **"Not found in the
   documents."** when the corpus does not support the query. A single similarity
   threshold is provably insufficient here (the "C-200" near-miss scores *above*
   real questions), so the gate combines:
   - **Equipment-code grounding:** if the query names a code (`C-200`) absent
     from the best doc, abstain. Generalizes; needs no tuned threshold.
   - **Semantic floor:** answer if the top dense cosine ≥ τ (calibrated, §6).
   - **Coverage override:** below the floor, answer only if the doc lexically
     contains ≥ 80% of the query's content words — this rescues terse keyword
     queries (low cosine, fully grounded) without admitting near-misses.

---

## 6. Evaluation

`qa.jsonl` — a small labeled set: **10–12 answerable** questions (each with a
**multi-gold** list of acceptable `doc_id`s) and **4–5 unanswerable** questions
(answer not in corpus) to measure abstention.

Metrics (`python eval.py`):

- **Retrieval (answerable only, abstention gate off):** Hit@1, Recall@3, MRR.
  Recall is meaningful *because* labels are multi-gold; with single-gold it would
  collapse to Hit@k.
- **Abstention (unanswerable + false-abstention on answerable):** a small
  confusion matrix — correct-abstention rate, and false-abstention rate.

Reported **before vs after**, reproduced by `./run_eval.sh`.

Eval set: 14 answerable (12 natural-language + 2 terse keyword) and 5
unanswerable questions.

| Metric | Baseline | Improved (MiniLM) | Improved (E5-small-v2) | Improved (GTE-small) |
|--------|----------|-------------------|------------------------|----------------------|
| Hit@1 (all answerable) | 0.857 | **1.000** | 1.000 | 1.000 |
| Hit@1 — natural-language | 1.000 | 1.000 | 1.000 | 1.000 |
| Hit@1 — keyword style | 0.000 | **1.000** | 1.000 | 1.000 |
| Recall@3 | 1.000 | 1.000 | 1.000 | 1.000 |
| MRR | 0.929 | **1.000** | 1.000 | 1.000 |
| Correct abstention (unanswerable) | 0/5 | **5/5** | 1/5 | 1/5 |
| False abstention (answerable) | 0/14 | 0/14 | 0/14 | 0/14 |

All embedders were given their own per-model abstention floor (cosine scales are
not comparable across models). MiniLM — the baseline's own model — is retained;
see §8 for why the "stronger" models do worse on abstention.

**Interpretation (this is the important part).** On a clean 16-document corpus,
dense MiniLM already ranks *natural-language* questions at ceiling — so the
baseline's weakness is **not** its ranker. The measured improvements are:

1. **Abstention 0/5 → 5/5** with **zero** false abstentions — the headline win,
   and the task's key requirement. The baseline fabricates an answer for every
   out-of-corpus question; the improved system refuses all five, including the
   adversarial "C-200" near-miss (caught by the equipment-code grounding check,
   which no similarity threshold could separate — its cosine 0.735 sits *above*
   several answerable questions).
2. **Ranking on terse/keyword queries 0/2 → 2/2.** Where dense-only is fragile
   (a decoy doc shares a word, e.g. DOC-11 "lockout/guard" vs the M-50 startup
   query), hybrid BM25 + title embedding recovers the right document. This is
   the realistic technician-query regime; NL questions are saturated for both.
3. **Conflict + duplicate handling** (qualitative, `python improved_rag.py`):
   the P-200 pressure query returns **both** DOC-01 and DOC-02 flagged
   `CONFLICTING VALUES — bar=12/16`; the F-30 vibration query **collapses** the
   DOC-05/DOC-06 paraphrase to one result.

**Threshold calibration honesty.** The three grounding thresholds
(cosine τ = 0.67, coverage = 0.80, and the code check) are set on the same 19
questions we report — with a set this small there is no clean held-out split.
The cosine floor sits in a real gap (cosine-separable unanswerable ≤ 0.654 vs
lowest answerable 0.686), and the code/coverage checks are *rules*, not fitted
numbers, but the τ values are still small-sample. We report them explicitly and
treat the 5/5 abstention as illustrative of the design, not a generalization
claim. `python -c "from improved_rag import *; calibrate_threshold(ImprovedRetriever().build(load_docs()))"`
prints the distributions the threshold is read from. More robust alternatives
(a learned gate, or NLI answer-span verification) are noted in §7.

---

## 7. Out-of-scope but high-impact improvements

Deliberately **not** implemented to respect the on-prem, limited-compute, CPU
constraints — but documented as the highest-value next steps:

- **Cross-encoder re-ranking** (e.g. `bge-reranker-base` ~278M, or larger).
  Architecturally superior — jointly attends over (query, passage) instead of
  comparing independent embeddings — and typically the single biggest retrieval
  quality lift. Excluded because even the "base" reranker is heavy for CPU and
  overkill for a 16-document corpus, where candidate generation already sees the
  whole collection.
- **A stronger embedding model — `bge-m3` (~568M).** Multi-functional
  (dense + sparse + multi-vector retrieval in one model), multilingual (100+
  languages), and multi-granularity (up to 8192 tokens). It would fold the
  hybrid dense+sparse logic into one model and handle long documents and
  non-English industrial docs. Excluded on CPU/memory grounds. (Note: `bge-m3`
  is an *embedding* model; the `bge-reranker` family are the cross-encoders —
  distinct components.)
- **A stronger abstention gate.** Our rule-based grounding gate (§5) is
  transparent and CPU-free but its thresholds are small-sample. Better options
  once labelled data exists: a learned confidence gate over the retrieval
  features, or **NLI / answer-span verification** — checking that the retrieved
  passage actually entails an answer to the question, rather than just being
  topically close. Excluded here to avoid a second model and to keep the
  abstention logic auditable.

---

## 8. Embedding model comparison — and why we keep MiniLM

We compared the baseline's `all-MiniLM-L6-v2` (22M, 384-dim) against two
similarly-sized, CPU-friendly models with *stronger* MTEB retrieval scores:

- **`e5-small-v2`** (33M, 384-dim) — requires `query:` / `passage:` input
  prefixes; omitting them silently degrades quality (handled in code).
- **`gte-small`** (33M, 384-dim).

**Result: the "better" models are not better here — they are worse at
abstention.** All three tie at ceiling on ranking (Hit@1 = Recall@3 = MRR = 1.0),
so the stronger MTEB scores buy nothing on a clean 16-doc corpus. But their
top-1 cosine distributions are compressed into a narrow high band, and
answerable vs. unanswerable questions **overlap**:

| Model | answerable cosine range | unanswerable cosine range | separable? |
|-------|-------------------------|---------------------------|------------|
| MiniLM | 0.686 – 0.832 | 0.462 – 0.735 | mostly (floor 0.67) |
| E5-small-v2 | 0.837 – 0.939 | 0.850 – 0.917 | no (heavy overlap) |
| GTE-small | 0.859 – 0.963 | 0.875 – 0.926 | no (heavy overlap) |

Because E5/GTE score several *unanswerable* questions above many *answerable*
ones, no cosine floor separates them without false abstentions — so with a
0-false-abstention floor they fall back to catching only the code-mismatch case
(1/5). MiniLM's wider spread makes its semantic floor actually work (5/5).

### Why E5/GTE fail at abstention (mechanism)

This is expected behaviour, not a setup bug — and note **all three models are
384-dim, so vector size is not the cause; the training objective is.**

- **Similarity inflation from the training objective.** E5 and GTE are trained on
  massive *weakly-supervised* query–document pairs (queries, relevant docs,
  paraphrases). The model learns that almost anything vaguely related should be
  close, which compresses cosine scores into a high band (~0.85–0.95). MiniLM
  (SBERT-style, trained with hard negatives) keeps a wider, more meaningful
  spread.
- **Cosine loses absolute meaning.** These models optimise a *relative* ordering
  — `score(query, positive) > score(query, negative)` — not an absolute one where
  `score(query, irrelevant)` is driven low. So the cosine *ranking* is excellent
  while the cosine *scale* is uncalibrated: perfect for top-k, useless as a
  rejection threshold.
- **Anisotropy (space collapse).** E5/GTE embedding spaces are highly
  anisotropic — vectors cluster in a narrow cone — so even unrelated texts land
  around 0.85+, exactly the unanswerable range we measured.

**Takeaway:** modern retrieval embeddings trade *calibration* for *ranking
quality*. They are rankers, not calibrated similarity models — top-1 is still
right, but the score itself is not a reliable rejection signal. This is why an
abstention design that leans on a similarity threshold prefers a
harder-negative, well-separated model like MiniLM.

**Decision: retain `all-MiniLM-L6-v2`.** It is the baseline's own model (minimal
change), it is the smallest, and on this corpus it is *strictly better* for the
abstention requirement. This is corpus-specific: on larger or multilingual
corpora a ranking-optimised model like `bge-m3` (§7) would likely win on
retrieval — but abstention would then need a calibrated signal (learned gate or
NLI verification, §7) rather than a raw cosine floor. The comparison harness
(`eval.py --model ...`) is in place to re-run that decision when the data changes.

---

## 9. Reproducibility

- `./run_eval.sh` re-runs the corpus analysis, the baseline, and the improved
  system, printing every metric in §6.
- CPU inference is deterministic; BM25 and RRF are deterministic. No seeds affect
  the reported numbers, but any sampling is seeded.
- `requirements.txt` pins the stack. Models download once, then run offline.

---

## 10. AI Usage

_(Per submission terms.)_

- **Tool used:** Claude Code (Anthropic, Opus) for diagnosis, code, evaluation
  design, and this README.
- **What it did:** drafted `analyze_corpus.py`, `improved_rag.py`, `eval.py`,
  `qa.jsonl`, and the README; ran the experiments.
- **What I reviewed / changed:** verified every metric by re-running
  `run_eval.sh`; validated the conflict/dedup behaviour by hand against the
  corpus; set the evaluation design decisions (multi-gold labels, metrics on the
  raw ranking, per-model abstention floors, keyword-vs-NL breakdown).
- **Concrete tool mistakes I caught and corrected:**
  1. **Over-claimed diagnosis.** It first stated the 400-char chunker *actively
     fragments* documents. The corpus measurement (§3) shows every text is
     ≤ 342 chars (< 400), so it already yields one chunk per doc — the defect is
     **latent, not active**. Corrected in §2 (defect 6).
  2. **Broken conflict detector.** Its first version flagged spurious conflicts
     across *unrelated* equipment (e.g. compressor "8 bar" vs pump "12 bar") and
     mis-parsed "E-207 indicates" as the unit `indicates`. Fixed by restricting
     conflict checks to docs sharing an equipment code and to a whitelist of
     real physical units.
  3. **Naive abstention.** A single cosine threshold false-abstained terse
     keyword queries (2/14). The evaluation caught it; fixed with the lexical
     coverage override (§5).

> Transparency aid: `code_assistant_conversations/` is an auto-generated,
> append-only Markdown archive of the AI coding sessions (via a `.claude/` Stop
> hook), kept as a reviewable record of how the AI was used. It is tooling
> context, not part of the deliverable.

---

## Repository layout

```
baseline_rag.py      # given baseline (unchanged)
improved_rag.py      # improved retrieval pipeline
analyze_corpus.py    # corpus length stats -> chunking decision
eval.py              # evaluation harness (ranking + abstention), model-parameterized
qa.jsonl             # labeled eval set (answerable multi-gold + unanswerable)
run_eval.sh          # one command reproduces every metric
corpus.jsonl         # given corpus (unchanged)
requirements.txt
```
