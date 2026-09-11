"""
Part C - Phrase & Proximity search using the positional index.

Phrase search: 'cotton shirt' must occur as EXACT consecutive positions
    (pos, pos+1, pos+2, ...) in some document -- not just "both terms
    occur somewhere in the document" (that's what ordinary VSM gives you).

Proximity search: 'term1 WITHIN/k term2' -- the two terms must occur at
    positions whose absolute difference is <= k, in EITHER order.
"""

import json
import os
import re

from preprocess import preprocess_query

STORE_PATH = os.path.join(os.path.dirname(__file__), "output", "index_store.json")


class PositionalSearcher:
    def __init__(self, store_path=STORE_PATH):
        with open(store_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.positional_index = data["positional_index"]  # term -> {docid: [tf, [positions]]}
        self.doc_meta = data["doc_meta"]

    def _postings(self, term):
        return self.positional_index.get(term, {})

    def phrase_search(self, phrase_text):
        """
        Returns (results, oov_terms) where results is a list of
        (docid, title, category, matched_positions) for docs containing
        the exact phrase, sorted by increasing docID.
        """
        terms = preprocess_query(phrase_text)
        if not terms:
            return [], []

        oov = [t for t in terms if t not in self.positional_index]
        if oov:
            return [], oov

        # Candidate docs = docs containing the first term.
        first_postings = self._postings(terms[0])
        candidate_docs = set(first_postings.keys())
        for t in terms[1:]:
            candidate_docs &= set(self._postings(t).keys())

        results = []
        for docid in sorted(candidate_docs):
            positions_per_term = [self._postings(t)[docid][1] for t in terms]
            base_positions = positions_per_term[0]
            for start in base_positions:
                if all((start + offset) in positions_per_term[offset]
                       for offset in range(len(terms))):
                    matched = list(range(start, start + len(terms)))
                    results.append((docid, self.doc_meta[docid]["title"],
                                     self.doc_meta[docid]["category"], matched))
                    break  # one match per doc is enough evidence
        return results, []

    def proximity_search(self, term1, term2, k):
        """
        term1 WITHIN/k term2: both terms occur in the same doc with
        |pos1 - pos2| <= k, order-independent.
        Returns (results, oov_terms); results = list of
        (docid, title, category, [(pos1, pos2, distance), ...]).
        """
        t1_list = preprocess_query(term1)
        t2_list = preprocess_query(term2)
        t1 = t1_list[0] if t1_list else term1
        t2 = t2_list[0] if t2_list else term2

        oov = [t for t in (t1, t2) if t not in self.positional_index]
        if oov:
            return [], oov

        docs1 = self._postings(t1)
        docs2 = self._postings(t2)
        common_docs = set(docs1.keys()) & set(docs2.keys())

        results = []
        for docid in sorted(common_docs):
            pos1_list = docs1[docid][1]
            pos2_list = docs2[docid][1]
            matches = []
            for p1 in pos1_list:
                for p2 in pos2_list:
                    if abs(p1 - p2) <= k:
                        matches.append((p1, p2, abs(p1 - p2)))
            if matches:
                results.append((docid, self.doc_meta[docid]["title"],
                                 self.doc_meta[docid]["category"], matches))
        return results, []

    @staticmethod
    def parse_proximity_query(query_str):
        """
        Parses a string like 'cotton WITHIN/3 shirt' into (term1, term2, k).
        Returns None if the string doesn't match the pattern.
        """
        m = re.match(r"^\s*(\S+)\s+WITHIN\s*/\s*(\d+)\s+(\S+)\s*$", query_str, re.IGNORECASE)
        if not m:
            return None
        term1, k, term2 = m.groups()
        return term1, term2, int(k)


if __name__ == "__main__":
    ps = PositionalSearcher()

    print("=== Phrase queries ===")
    for phrase in ["cotton shirt", "stretch denim", "zip closure", "high waist"]:
        results, oov = ps.phrase_search(phrase)
        print(f"\nPhrase: {phrase!r}")
        if oov:
            print(f"  out-of-vocabulary: {oov}")
        for docid, title, cat, positions in results[:5]:
            print(f"  {docid}  [{cat}]  {title}   positions={positions}")
        if not results and not oov:
            print("  No exact phrase matches.")

    print("\n=== Proximity queries ===")
    for q in ["cotton WITHIN/3 shirt", "stretch WITHIN/4 denim", "winter WITHIN/3 wear"]:
        term1, term2, k = ps.parse_proximity_query(q)
        results, oov = ps.proximity_search(term1, term2, k)
        print(f"\nQuery: {q!r}")
        if oov:
            print(f"  out-of-vocabulary: {oov}")
        for docid, title, cat, matches in results[:5]:
            print(f"  {docid}  [{cat}]  {title}   matches={matches[:3]}")
        if not results and not oov:
            print("  No matches within k.")
