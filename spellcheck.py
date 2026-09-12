"""
Novelty feature: "Did you mean ...?" suggestions for out-of-vocabulary
query terms.

Uses Levenshtein edit distance against the corpus vocabulary (the set of
terms actually present in the inverted/positional index, after the same
tokenize/stopword/stem pipeline used everywhere else). No external
dependencies, consistent with the rest of this project.
"""


def levenshtein(a, b):
    """Standard edit distance (insertions, deletions, substitutions)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(
                prev[j] + 1,          # deletion
                curr[j - 1] + 1,      # insertion
                prev[j - 1] + cost,   # substitution
            )
        prev = curr
    return prev[len(b)]


def suggest(term, vocabulary, max_distance=2):
    """
    Returns the closest vocabulary term to `term` within max_distance
    edits, or None if nothing in the vocabulary is close enough to be a
    useful suggestion. Ties are broken by picking the shorter/earlier
    term encountered (stable, deterministic given a sorted vocabulary).
    """
    best_term = None
    best_dist = max_distance + 1
    for v in vocabulary:
        d = levenshtein(term, v)
        if d < best_dist:
            best_dist = d
            best_term = v
    return best_term if best_dist <= max_distance else None
