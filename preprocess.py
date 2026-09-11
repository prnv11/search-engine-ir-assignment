"""
Part A helper: tokenization, case normalization, punctuation removal,
stop-word removal, and stemming.

Stop-word policy (documented, as the assignment asks):
    We use the standard 'long' English stop-word list (the same ~179-word
    list used by NLTK's `stopwords.words('english')`), applied AFTER
    lower-casing and punctuation stripping and BEFORE stemming. This list
    is fixed below so the project has no external download dependency.
"""

import re
from stemmer import PorterStemmer

_stemmer = PorterStemmer()

# Standard English stop-word list (kept identical across the whole project
# so preprocessing is 100% consistent between indexing and querying).
STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your
yours yourself yourselves he him his himself she she's her hers herself
it it's its itself they them their theirs themselves what which who whom
this that that'll these those am is are was were be been being have has
had having do does did doing a an the and but if or because as until
while of at by for with about against between into through during before
after above below to from up down in out on off over under again further
then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will
just don don't should should've now d ll m o re ve y ain aren aren't
couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan
shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn
wouldn't
""".split())

_token_re = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lower-case, strip punctuation, split into alphanumeric tokens."""
    text = text.lower()
    return _token_re.findall(text)


def preprocess(text, remove_stopwords=True, stem=True):
    """
    Full pipeline: tokenize -> normalize case -> remove punctuation
    (handled inside tokenize) -> remove stopwords -> stem.
    Returns a list of processed tokens, preserving original order/position.
    """
    tokens = tokenize(text)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if stem:
        tokens = [_stemmer.stem(t) for t in tokens]
    return tokens


def preprocess_query(text):
    """Same pipeline, used for queries so query/doc vocab matches exactly."""
    return preprocess(text)
