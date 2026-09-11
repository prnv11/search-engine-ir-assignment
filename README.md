# Clothing Search Engine — README

A small IR system built over the supplied 100-document clothing corpus,
covering: inverted index + dictionary (Part A), Vector Space Model ranked
retrieval with lnc.ltc weighting (Part B), a positional index supporting
exact phrase and proximity search (Part C), a Flask web app (Part D), and
a test report (Part E).

## 1. Requirements

- Python 3.8+
- Flask (`pip install flask`)

No other third-party packages are used. Tokenization, stop-word removal,
and stemming (a from-scratch Porter Stemmer) are implemented directly in
this project so it runs fully offline.

## 2. Project structure

```
clothing_search_engine/
├── corpus_100.txt          # supplied corpus
├── stemmer.py               # Porter Stemmer (from scratch)
├── preprocess.py            # tokenize / normalize / stopwords / stem
├── indexer.py                # Part A + Part C: builds inverted & positional index
├── vsm.py                     # Part B: lnc.ltc cosine similarity search
├── positional_search.py       # Part C: phrase & proximity search
├── app.py                      # Part D: Flask web app
├── test_queries.py             # Part E: mandatory test suite -> report
├── templates/
│   └── index.html               # web UI
└── output/                       # generated on first run
    ├── dictionary_postings.txt    # Part A deliverable
    ├── positional_index.txt        # Part C deliverable
    ├── index_store.json             # internal cache used by app/vsm
    └── test_report.txt               # Part E deliverable
```

## 3. How to run (step by step, in VS Code)

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal in VS Code (`` Ctrl+` ``) and install Flask:
   ```
   pip install flask
   ```
3. Build the indexes (Part A + Part C). This reads `corpus_100.txt` and
   writes the dictionary/postings and positional index files into `output/`:
   ```
   python indexer.py
   ```
4. (Optional) Sanity-check Part B and Part C from the command line:
   ```
   python vsm.py
   python positional_search.py
   ```
5. Run the mandatory Part E test suite (writes `output/test_report.txt`):
   ```
   python test_queries.py
   ```
6. Launch the web app (Part D):
   ```
   python app.py
   ```
   Then open **http://127.0.0.1:5000** in your browser.
7. Take your screenshots here:
   - Free-text tab: try `cotton shirt`, `festive kurta`, `winter jacket`.
   - Exact Phrase tab: try `cotton shirt`, `stretch denim`, `zip closure`
     (this one deliberately returns *no* matches — good evidence the
     phrase search isn't just doing co-occurrence).
   - Proximity tab: try `cotton` WITHIN/`3` `shirt`.

## 4. Design decisions (for your report)

- **Stop-word list**: the standard ~179-word English list (same set NLTK
  ships), hardcoded in `preprocess.py` so there's no network/download
  dependency. Applied consistently to both documents and queries.
- **Stemming**: classic Porter algorithm, implemented from scratch in
  `stemmer.py`.
- **Indexed fields**: title + category + description text are concatenated
  and indexed together per document, since the sample queries (e.g.
  "festive wear") are meant to match anywhere in a product's description.
- **Position numbering**: positions are assigned over the *processed*
  token stream (after stopword removal + stemming), not the raw text.
  This keeps position indices consistent with the vocabulary used in Part
  A/B, but it does mean that if a stopword sits between two content
  words in the original text (e.g. "cotton **and** shirt"), those words
  become *adjacent* after stopword removal and so would satisfy a phrase
  query. This trade-off is discussed in `output/test_report.txt`.
- **lnc.ltc weighting**: implemented exactly as specified — document side
  uses log-tf only (no idf), query side uses log-tf × idf, both vectors
  cosine-normalized before the dot product.
- **Proximity semantics**: `t1 WITHIN/k t2` matches if any occurrence of
  `t1` and any occurrence of `t2` in the same document have
  `|pos1 - pos2| <= k`, in either order.

## 5. Packaging for submission

Zip the whole `clothing_search_engine/` folder (including the `output/`
files you generated and your screenshots) into a single ZIP, as required
by the assignment's Deliverables section.
