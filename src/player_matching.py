"""Fuzzy player name matching between squad rosters and goalscorer records."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache


def normalize_player_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace(" junior", " jr")
    return text.strip()


@lru_cache(maxsize=4096)
def _token_set(name: str) -> frozenset[str]:
    return frozenset(normalize_player_name(name).split())


def name_similarity(a: str, b: str) -> float:
    ta, tb = _token_set(a), _token_set(b)
    if not ta or not tb:
        return 0.0
    if ta == tb:
        return 1.0
    inter = len(ta & tb)
    union = len(ta | tb)
    jaccard = inter / union if union else 0.0
    la = max(ta, key=len) if ta else ""
    lb = max(tb, key=len) if tb else ""
    last_boost = 0.25 if la == lb and la else 0.0
    return min(1.0, jaccard + last_boost)


def best_name_match(query: str, candidates: list[str], *, threshold: float = 0.55) -> str | None:
    if not candidates:
        return None
    nq = normalize_player_name(query)
    best, score = None, 0.0
    for cand in candidates:
        if normalize_player_name(cand) == nq:
            return cand
        s = name_similarity(query, cand)
        if s > score:
            best, score = cand, s
    return best if score >= threshold else None
