from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
import re
import string

# Reconstruct classes used in preprocessing.py so pickle can find them
# ==============================================================================

GER_PROFANITY = {
    "schlagen", "erschlagen", "töten", "ermorden", "verprügeln", "boxen",
    "kämpfen", "angreifen", "treten", "morden", "prügeln", "hauen",
}

_LEET_MAP = str.maketrans({
    "@": "a", "4": "a",
    "1": "i", "!": "i",
    "0": "o",
    "$": "s", "5": "s",
    "3": "e",
})

_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")


def _split_camel_case(token: str) -> str:
    token = re.sub(r"([a-z])([A-Z])", r"\1 \2", token)
    token = re.sub(r"([A-Za-z])(\d)", r"\1 \2", token)
    token = re.sub(r"(\d)([A-Za-z])", r"\1 \2", token)
    token = token.replace("_", " ")
    return token


def normalize_tweet(text: str) -> str:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""

    text = str(text)
    text = _URL_RE.sub(" HTTPURL ", text)
    text = _USER_RE.sub(" @USER ", text)

    def _hashtag_repl(m: re.Match) -> str:
        tag = m.group(1)
        tag = _split_camel_case(tag)
        return f" {tag} "

    text = _HASHTAG_RE.sub(_hashtag_repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_for_lexicon(text: str) -> str:
    t = normalize_tweet(text)
    t = t.lower().translate(_LEET_MAP)
    t = re.sub(r"(?<=\w)[^\w\s]+(?=\w)", "", t)
    return t


class ProfanityLexiconFeaturizer(BaseEstimator, TransformerMixin):
    """
    Outputs 3 numeric features:
      - profane_count
      - profane_ratio
      - has_profane
    """
    def __init__(self, lexicon=None, show_progress: bool = True):
        self.lexicon = lexicon or GER_PROFANITY
        # show_progress might be used in pickle state
        self.show_progress = show_progress

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
        out = np.zeros((len(s), 3), dtype=np.float32)
        
        for i, txt in enumerate(s):
            t = normalize_for_lexicon(txt)
            tokens = _WORD_RE.findall(t)
            if not tokens:
                continue
            hits = sum(1 for w in tokens if w in self.lexicon)
            out[i, 0] = hits
            out[i, 1] = hits / max(len(tokens), 1)
            out[i, 2] = 1.0 if hits > 0 else 0.0

        return out


_PUNCT_SET = set(string.punctuation)


def punctuation_length_rate(text: str) -> float:
    t = normalize_tweet(text)
    if not t:
        return 0.0
    punct = sum(1 for ch in t if ch in _PUNCT_SET)
    return punct / max(len(t), 1)


def punctuation_length_rate_transform(X):
    # Depending on how it was pickled, it might be a function or expected to be in module
    s = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(np.asarray(X).ravel())
    return s.apply(punctuation_length_rate).astype("float32").to_numpy().reshape(-1, 1)
