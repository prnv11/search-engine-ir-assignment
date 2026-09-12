# Clothing Search Engine

A small information retrieval system built over a 100-document clothing corpus, covering: inverted index and dictionary construction, Vector Space Model ranked retrieval with lnc.ltc weighting, a positional index supporting exact phrase and proximity search, a Flask web app, and a test report.

## Requirements

- Python 3.8+
- Flask (`pip install flask`)

No other third-party packages are used. Tokenization, stop-word removal, and stemming (a from-scratch Porter Stemmer) are implemented directly in this project so it runs fully offline.

## Project Structure

```
clothing_search_engine/
├── corpus_100.txt              # supplied corpus
├── stemmer.py                  # Porter Stemmer (from scratch)
├── preprocess.py                # tokenize / normalize / stopwords / stem
├── indexer.py                    # builds the inverted index and positional index
├── vsm.py                          # lnc.ltc cosine similarity search
├── positional_search.py            # phrase and proximity search
├── app.py                            # Flask web app
├── test_queries.py
├── test_queries.py                     # test suite -> report
├── screenshots/
│   └── 1_freetext_search.jpeg
│   └── 2_phrase_search.jpeg
│   └── 3_proximity_search.jpeg
├── templates/
│   └── index.html                        # web UI
└── output/                                 # generated on first run
    ├── dictionary_postings.txt              # inverted index and dictionary
    ├── positional_index.txt                   # positional index
    ├── index_store.json                         # internal cache used by app/vsm
    └── test_report.txt                            # test report
```

## How to Run

1. Open this folder in VS Code (`File > Open Folder...`).
2. Open a terminal in VS Code (`Ctrl+\``) and install Flask:
   ```
   pip install flask
   ```
3. Build the indexes. This reads `corpus_100.txt` and writes the dictionary/postings and positional index files into `output/`:
   ```
   python indexer.py
   ```
4. (Optional) Sanity-check ranked retrieval and phrase/proximity search from the command line:
   ```
   python vsm.py
   python positional_search.py
   ```
5. Run the test suite (writes `output/test_report.txt`):
   ```
   python test_queries.py
   ```
6. Launch the web app:
   ```
   python app.py
   ```
   Then open http://127.0.0.1:5000 in your browser.

### Example Queries

- **Free-text tab**: `cotton shirt`, `festive kurta`, `winter jacket`
- **Exact Phrase tab**: `cotton shirt`, `stretch denim`, `zip closure` (deliberately returns no matches — evidence phrase search isn't just doing co-occurrence)
- **Proximity tab**: `cotton WITHIN/3 shirt`

## Design Decisions

- **Stop-word list**: the standard ~179-word English list (same set NLTK ships), hardcoded in `preprocess.py` so there's no network/download dependency. Applied consistently to both documents and queries.
- **Stemming**: classic Porter algorithm, implemented from scratch in `stemmer.py`.
- **Indexed fields**: title + category + description text are concatenated and indexed together per document, since sample queries (e.g. "festive wear") are meant to match anywhere in a product's description.
- **Position numbering**: positions are assigned over the processed token stream (after stopword removal + stemming), not the raw text. This keeps position indices consistent with the vocabulary used elsewhere in the pipeline, but means that if a stopword sits between two content words in the original text (e.g. "cotton and shirt"), those words become adjacent after stopword removal and would satisfy a phrase query. This trade-off is discussed in `output/test_report.txt`.
- **lnc.ltc weighting**: implemented exactly as specified — document side uses log-tf only (no idf), query side uses log-tf × idf, both vectors cosine-normalized before the dot product.
- **Proximity semantics**: `t1 WITHIN/k t2` matches if any occurrence of `t1` and any occurrence of `t2` in the same document have `|pos1 - pos2| <= k`, in either order.


## Project Members

- Anuran Basu (2310110053)
- Pranav Talwar (2310110555)