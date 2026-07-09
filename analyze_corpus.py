"""
analyze_corpus.py — corpus statistics that motivate the chunking decision.

Reports per-record character lengths and (if the embedding model's tokenizer is
available locally) exact wordpiece token counts, then compares the longest
record against the model's max sequence length to decide whether whole-record
encoding is safe.

Run:  python analyze_corpus.py
"""
import json
import statistics as st

CORPUS_PATH = "corpus.jsonl"
EMBED_MODEL = "all-MiniLM-L6-v2"
MODEL_MAX_SEQ = 256  # all-MiniLM-L6-v2 truncates inputs beyond 256 wordpieces
SEP = " — "          # separator used when we embed "title — text" as one unit


def load_docs(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def summarize(values, label):
    print(
        f"  {label:20s} min={min(values):4d}  max={max(values):4d}  "
        f"avg={sum(values) / len(values):6.1f}  median={int(st.median(values)):4d}"
    )


def real_token_counts(docs):
    """Exact wordpiece counts via the model tokenizer; None if unavailable offline."""
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBED_MODEL)
        counts = []
        for d in docs:
            enc = model.tokenize([d["title"] + SEP + d["text"]])
            counts.append(int(enc["input_ids"].shape[1]))  # single input, no padding
        return counts
    except Exception as exc:  # model/lib not installed — fall back to estimate
        print(f"  (tokenizer unavailable: {exc}; using ~4 chars/token estimate)")
        return None


def main():
    docs = load_docs(CORPUS_PATH)
    print(f"Corpus: {len(docs)} records\n")

    title_len = [len(d["title"]) for d in docs]
    text_len = [len(d["text"]) for d in docs]
    combo_len = [len(d["title"]) + len(SEP) + len(d["text"]) for d in docs]

    print("Character lengths:")
    summarize(title_len, "title")
    summarize(text_len, "text")
    summarize(combo_len, "title+text")

    print("\nToken lengths (title+text):")
    tokens = real_token_counts(docs)
    if tokens is None:
        tokens = [c // 4 for c in combo_len]
        source = "estimated (chars/4)"
    else:
        source = "exact wordpieces"
    summarize(tokens, f"tokens [{source}]")

    longest = max(tokens)
    print(f"\nModel '{EMBED_MODEL}' max_seq_length = {MODEL_MAX_SEQ} wordpieces")
    print(f"Longest record = {longest} tokens")
    if longest <= MODEL_MAX_SEQ:
        print(
            "=> Every record fits well within the context window. "
            "Chunking is unnecessary: embed each record whole (title + text)."
        )
    else:
        print("=> Some records exceed the window; sentence-aware chunking is needed.")


if __name__ == "__main__":
    main()
