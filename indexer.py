"""
Part A - Corpus & Pre-processing
Part C - Positional Index

Parses corpus_100.txt (SGML-ish <DOC>...</DOC> records), builds:
  1. An inverted index: term -> df -> {docID: tf}
  2. A positional index: term -> df -> {docID: [tf, [p1, p2, ...]]}
  3. Document-length norms needed later for cosine normalization (Part B).

Design decision (documented for the report):
  Token *positions* are assigned over the fully processed token stream
  (after lower-casing, punctuation removal, stop-word removal, and
  stemming) rather than over the raw text. This keeps position numbering
  consistent with the vocabulary used everywhere else in the project, and
  is the standard simplification used in this kind of assignment. It is
  called out explicitly in the report because it does affect phrase
  results: e.g. "cotton and shirt" collapses to positions {cotton:0,
  shirt:1} once the stop-word "and" is removed, so it WOULD satisfy the
  phrase query "cotton shirt". This is discussed in Part E.
"""

import re
import json
import os
from collections import defaultdict
from preprocess import preprocess, tokenize, STOPWORDS

CORPUS_PATH = os.path.join(os.path.dirname(__file__), "corpus_100.txt")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

DOC_RE = re.compile(
    r"<DOC>\s*<DOCID>(.*?)</DOCID>\s*<CATEGORY>(.*?)</CATEGORY>\s*"
    r"<TITLE>(.*?)</TITLE>\s*<TEXT>(.*?)</TEXT>\s*</DOC>",
    re.DOTALL,
)


def load_corpus(path=CORPUS_PATH):
    """Read corpus_100.txt and return a list of doc dicts."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    docs = []
    for m in DOC_RE.finditer(raw):
        docid, category, title, text = m.groups()
        docs.append({
            "docid": docid.strip(),
            "category": category.strip(),
            "title": title.strip(),
            "text": text.strip(),
        })
    return docs


def build_indexes(docs):
    """
    Returns:
      inverted_index: {term: {docid: tf}}
      positional_index: {term: {docid: [tf, [positions]]}}
      doc_lengths: {docid: L2 norm of (1+log10(tf)) weights}  -- for Part B
      doc_meta: {docid: {"title":..., "category":..., "text":...}}
      N: number of documents
      word_frequencies: {real_word: total_count} -- UNSTEMMED words (with
        stopwords still removed), used only to drive autocomplete so
        suggestions are real dictionary words instead of stemmed roots
        like "washabl" or "featur".
    """
    import math

    inverted_index = defaultdict(lambda: defaultdict(int))
    positional_index = defaultdict(lambda: defaultdict(lambda: [0, []]))
    doc_meta = {}
    word_frequencies = defaultdict(int)

    for doc in docs:
        docid = doc["docid"]
        doc_meta[docid] = {
            "title": doc["title"],
            "category": doc["category"],
            "text": doc["text"],
        }
        # Index title + category + text together, as the assignment's
        # sample queries (e.g. "cotton shirt", "festive wear") match
        # against the full product description.
        full_text = f'{doc["title"]} {doc["category"]} {doc["text"]}'
        tokens = preprocess(full_text)

        for pos, term in enumerate(tokens):
            inverted_index[term][docid] += 1
            entry = positional_index[term][docid]
            entry[0] += 1
            entry[1].append(pos)

        for word in tokenize(full_text):
            if word not in STOPWORDS:
                word_frequencies[word] += 1

    # Document-length norms for Inc.ltc cosine normalization (Part B).
    doc_lengths = {}
    doc_term_weights = defaultdict(dict)  # docid -> {term: wd,t}
    for term, postings in inverted_index.items():
        for docid, tf in postings.items():
            w = 1 + math.log10(tf) if tf > 0 else 0.0
            doc_term_weights[docid][term] = w

    for docid, weights in doc_term_weights.items():
        doc_lengths[docid] = math.sqrt(sum(w * w for w in weights.values()))

    N = len(docs)
    return inverted_index, positional_index, doc_lengths, doc_meta, N, dict(word_frequencies)


def save_dictionary_and_postings(inverted_index, N, path=None):
    path = path or os.path.join(OUTPUT_DIR, "dictionary_postings.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Vocabulary size: {len(inverted_index)}   |   N documents: {N}\n")
        f.write("=" * 70 + "\n")
        for term in sorted(inverted_index.keys()):
            postings = inverted_index[term]
            df = len(postings)
            f.write(f"{term}  (df={df})\n")
            plist = ", ".join(f"{docid}:{tf}" for docid, tf in sorted(postings.items()))
            f.write(f"    postings: {plist}\n")
    return path


def save_positional_index(positional_index, path=None):
    path = path or os.path.join(OUTPUT_DIR, "positional_index.txt")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(positional_index.keys()):
            postings = positional_index[term]
            df = len(postings)
            f.write(f"{term}  (df={df})\n")
            for docid, (tf, positions) in sorted(postings.items()):
                f.write(f"    {docid} -> tf={tf}, positions={positions}\n")
    return path


def save_json_indexes(inverted_index, positional_index, doc_lengths, doc_meta, N, word_frequencies):
    """Persist everything as JSON so app.py / vsm.py can load quickly
    without re-parsing the corpus and re-running the stemmer each time."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = {
        "N": N,
        "inverted_index": {t: dict(p) for t, p in inverted_index.items()},
        "positional_index": {t: {d: v for d, v in p.items()} for t, p in positional_index.items()},
        "doc_lengths": doc_lengths,
        "doc_meta": doc_meta,
        "word_frequencies": word_frequencies,
    }
    path = os.path.join(OUTPUT_DIR, "index_store.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    return path


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents from {CORPUS_PATH}")
    inverted_index, positional_index, doc_lengths, doc_meta, N, word_frequencies = build_indexes(docs)
    print(f"Vocabulary size (after stopword removal + stemming): {len(inverted_index)}")

    p1 = save_dictionary_and_postings(inverted_index, N)
    p2 = save_positional_index(positional_index)
    p3 = save_json_indexes(inverted_index, positional_index, doc_lengths, doc_meta, N, word_frequencies)

    print(f"Wrote dictionary/postings -> {p1}")
    print(f"Wrote positional index    -> {p2}")
    print(f"Wrote JSON index store    -> {p3}")


if __name__ == "__main__":
    main()
