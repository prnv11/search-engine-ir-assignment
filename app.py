"""
Part D - Application

A minimal Flask app with two search modes:
  1. Free-text ranked search (Vector Space Model, Part B)
  2. Phrase / Proximity search using the positional index (Part C)

Run with:  python app.py
Then open: http://127.0.0.1:5000
"""

from flask import Flask, render_template, request

from vsm import VSMSearcher
from positional_search import PositionalSearcher

app = Flask(__name__)
vsm = VSMSearcher()
pos_searcher = PositionalSearcher()


@app.route("/", methods=["GET", "POST"])
def home():
    mode = request.form.get("mode", "freetext")
    proximity_k = request.form.get("k", "3")

    # Each tab now posts its own field name so Enter-to-submit can never
    # accidentally read a stale/wrong field from another tab.
    if mode == "phrase":
        query = request.form.get("phrase_query", "")
    elif mode == "proximity":
        query = request.form.get("prox_term1", "")
    else:
        query = request.form.get("query", "")

    results = None
    oov_terms = []
    error = None
    query_type_label = ""

    if request.method == "POST" and query.strip():
        if mode == "freetext":
            query_type_label = "Free-text (VSM, lnc.ltc cosine similarity)"
            ranked, oov_terms = vsm.search(query, top_k=10)
            results = [
                {"docid": d, "score": f"{s:.4f}", "title": t, "category": c, "positions": None}
                for d, s, t, c in ranked
            ]

        elif mode == "phrase":
            query_type_label = "Exact phrase search (positional index)"
            matches, oov_terms = pos_searcher.phrase_search(query)
            results = [
                {"docid": d, "score": None, "title": t, "category": c, "positions": pos}
                for d, t, c, pos in matches
            ]

        elif mode == "proximity":
            query_type_label = "Proximity search (positional index)"
            term2 = request.form.get("term2", "").strip()
            if not term2:
                error = "Please enter a second term for the proximity search."
            elif not proximity_k.strip().isdigit():
                error = f"'k' must be a positive whole number (got {proximity_k!r})."
            else:
                matches, oov_terms = pos_searcher.proximity_search(query, term2, int(proximity_k))
                results = [
                    {"docid": d, "score": None, "title": t, "category": c,
                     "positions": m[:3]} for d, t, c, m in matches
                ]

    return render_template(
        "index.html",
        mode=mode,
        query=query,
        proximity_k=proximity_k,
        term2=request.form.get("term2", ""),
        results=results,
        oov_terms=oov_terms,
        query_type_label=query_type_label,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
