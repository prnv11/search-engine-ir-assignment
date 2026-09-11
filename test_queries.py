"""
Part E - Testing

Runs the mandatory test set required by the assignment and writes a
human-readable report to output/test_report.txt:
  - >= 10 free-text queries
  - >= 5 exact phrase queries
  - >= 3 proximity queries with different k values
  - >= 1 query containing a term not in the corpus
  - discussion of >= 2 cases where positional info changes the result set/order
"""

import os
from vsm import VSMSearcher
from positional_search import PositionalSearcher

OUT_PATH = os.path.join(os.path.dirname(__file__), "output", "test_report.txt")

FREETEXT_QUERIES = [
    "cotton shirt",
    "festive kurta",
    "winter jacket",
    "stretch denim jeans",
    "breathable fabric t-shirt",
    "high waist leggings",
    "regular fit sweatshirt",
    "printed saree",
    "slim fit hoodie",
    "zip closure jacket",
    "xylophone spaceship",       # OOV test query (nonsense terms)
]

PHRASE_QUERIES = [
    "cotton shirt",
    "stretch denim",
    "festive wear",
    "winter wear",
    "regular fit",
    "breathable fabric",
    "zip closure",
    "high waist",
]

PROXIMITY_QUERIES = [
    ("cotton", "shirt", 3),
    ("stretch", "denim", 4),
    ("winter", "wear", 3),
    ("festive", "kurta", 4),
]


def run():
    vsm = VSMSearcher()
    ps = PositionalSearcher()

    lines = []
    lines.append("=" * 78)
    lines.append("PART E - TEST REPORT")
    lines.append("=" * 78)

    # ---------- Free-text queries ----------
    lines.append("\n\n### 1. FREE-TEXT QUERIES (VSM, lnc.ltc cosine) ###\n")
    for q in FREETEXT_QUERIES:
        results, oov = vsm.search(q, top_k=10)
        lines.append(f"Query: {q!r}")
        if oov:
            lines.append(f"  [out-of-vocabulary terms ignored: {oov}]")
        if not results:
            lines.append("  No results.")
        for docid, score, title, cat in results:
            lines.append(f"  {docid}  score={score:.4f}  [{cat}]  {title}")
        lines.append("")

    # ---------- Phrase queries ----------
    lines.append("\n### 2. EXACT PHRASE QUERIES (positional index) ###\n")
    for q in PHRASE_QUERIES:
        results, oov = ps.phrase_search(q)
        lines.append(f"Phrase: {q!r}")
        if oov:
            lines.append(f"  [out-of-vocabulary terms: {oov}]")
        if not results:
            lines.append("  No exact phrase matches.")
        for docid, title, cat, pos in results[:10]:
            lines.append(f"  {docid}  [{cat}]  {title}  positions={pos}")
        lines.append("")

    # ---------- Proximity queries ----------
    lines.append("\n### 3. PROXIMITY QUERIES (WITHIN/k, positional index) ###\n")
    for t1, t2, k in PROXIMITY_QUERIES:
        results, oov = ps.proximity_search(t1, t2, k)
        lines.append(f"Query: '{t1} WITHIN/{k} {t2}'")
        if oov:
            lines.append(f"  [out-of-vocabulary terms: {oov}]")
        if not results:
            lines.append("  No matches within k.")
        for docid, title, cat, matches in results[:10]:
            lines.append(f"  {docid}  [{cat}]  {title}  matches(pos1,pos2,dist)={matches[:3]}")
        lines.append("")

    # ---------- Discussion: where positional info changes results ----------
    lines.append("\n### 4. CASES WHERE POSITIONAL INFORMATION CHANGES THE RESULT ###\n")

    # Case 1: "zip closure" - free text vs phrase
    ft_results, _ = vsm.search("zip closure", top_k=10)
    ph_results, _ = ps.phrase_search("zip closure")
    lines.append("Case 1: Query 'zip closure'")
    lines.append(f"  Free-text VSM top result(s): {[r[0] for r in ft_results[:5]]}")
    lines.append(f"  Exact phrase result(s):      {[r[0] for r in ph_results]}")
    lines.append("  Explanation: The corpus contains 'zip fly' (Jeans) and 'button")
    lines.append("  closure' (Shirts) but never the exact bigram 'zip closure'.")
    lines.append("  Ordinary VSM still ranks jeans/shirts highly because 'zip' and")
    lines.append("  'closure' both appear SOMEWHERE in those documents (bag-of-words")
    lines.append("  co-occurrence is enough). The positional/phrase search correctly")
    lines.append("  returns ZERO documents, because it requires the two terms to be")
    lines.append("  adjacent in that exact order -- proving positional info is doing")
    lines.append("  real work beyond plain term co-occurrence.")

    # Case 2: "cotton shirt" phrase vs proximity vs free-text ordering
    ft2, _ = vsm.search("cotton shirt", top_k=10)
    ph2, _ = ps.phrase_search("cotton shirt")
    prox2, _ = ps.proximity_search("cotton", "shirt", 3)
    lines.append("\nCase 2: Query 'cotton shirt'")
    lines.append(f"  Free-text VSM top-5:  {[r[0] for r in ft2[:5]]}")
    lines.append(f"  Exact phrase matches: {[r[0] for r in ph2]}")
    lines.append(f"  Proximity (k=3) matches: {[r[0] for r in prox2]}")
    lines.append("  Explanation: VSM ranks ALL documents mentioning both 'cotton' and")
    lines.append("  'shirt' anywhere in the text (e.g. T-shirts where 'cotton' appears")
    lines.append("  in one sentence and 'shirt' several words later), so its result")
    lines.append("  set is larger and ordered purely by tf-idf weight, not adjacency.")
    lines.append("  The exact-phrase search is strictly more precise: it only returns")
    lines.append("  documents where 'cotton' is immediately followed by 'shirt', which")
    lines.append("  removes false positives that only share vocabulary, not meaning.")
    lines.append("  The proximity search (k=3) sits in between: it recovers documents")
    lines.append("  where the two words are close together but not strictly adjacent,")
    lines.append("  giving a superset of the phrase results and a subset of the VSM")
    lines.append("   'anywhere in doc' results -- demonstrating the three techniques")
    lines.append("  trade off precision vs recall differently.")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Test report written to {OUT_PATH}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    run()
