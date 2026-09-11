"""
Part B - Vector Space Model (lnc.ltc weighting scheme)

Document weight:  wd,t = 1 + log10(tf)         for tf > 0   (NO idf)
Query weight:     wq,t = (1 + log10(tf)) * log10(N / df)
Both vectors are cosine-normalized before the dot product is taken
(equivalently: dot product of raw weights, divided by the product of the
two vector norms -- that's what we implement, it's mathematically
identical and avoids re-normalizing every document on every query).
"""

import json
import math
import os
from collections import defaultdict

from preprocess import preprocess_query

STORE_PATH = os.path.join(os.path.dirname(__file__), "output", "index_store.json")


class VSMSearcher:
    def __init__(self, store_path=STORE_PATH):
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.N = data["N"]
        self.inverted_index = data["inverted_index"]      # term -> {docid: tf}
        self.doc_lengths = data["doc_lengths"]             # docid -> norm
        self.doc_meta = data["doc_meta"]                   # docid -> {title, category, text}

    def df(self, term):
        return len(self.inverted_index.get(term, {}))

    def search(self, query_text, top_k=10):
        """
        Returns a list of (docid, score, title, category) sorted by
        decreasing cosine similarity, ties broken by increasing docID.
        Terms with df == 0 (out-of-vocabulary) are reported separately
        and simply contribute nothing to the score.
        """
        q_tokens = preprocess_query(query_text)
        q_tf = defaultdict(int)
        for t in q_tokens:
            q_tf[t] += 1

        oov_terms = [t for t in q_tf if self.df(t) == 0]

        # ---- query weights (ltc) ----
        q_weights = {}
        for term, tf in q_tf.items():
            df = self.df(term)
            if df == 0:
                continue  # term not in corpus vocabulary at all
            idf = math.log10(self.N / df)
            q_weights[term] = (1 + math.log10(tf)) * idf

        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0:
            return [], oov_terms  # nothing matchable

        # ---- accumulate dot products doc-by-doc using postings ----
        scores = defaultdict(float)
        for term, qw in q_weights.items():
            postings = self.inverted_index.get(term, {})
            for docid, tf in postings.items():
                dw = 1 + math.log10(tf)  # lnc: log-tf, no idf on doc side
                scores[docid] += dw * qw

        # ---- normalize by doc norm * query norm (cosine) ----
        results = []
        for docid, dot in scores.items():
            dnorm = self.doc_lengths.get(docid, 0.0)
            if dnorm == 0:
                continue
            cos = dot / (dnorm * q_norm)
            results.append((docid, cos))

        results.sort(key=lambda x: (-x[1], x[0]))
        results = results[:top_k]

        enriched = [
            (docid, score, self.doc_meta[docid]["title"], self.doc_meta[docid]["category"])
            for docid, score in results
        ]
        return enriched, oov_terms


if __name__ == "__main__":
    searcher = VSMSearcher()
    for q in ["cotton shirt", "festive kurta", "winter jacket"]:
        print(f"\nQuery: {q!r}")
        results, oov = searcher.search(q)
        if oov:
            print(f"  (out-of-vocabulary terms ignored: {oov})")
        for docid, score, title, cat in results:
            print(f"  {docid}  {score:.4f}  [{cat}]  {title}")
