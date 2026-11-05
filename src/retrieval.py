"""
Step 4 - Chunk approved policies (keeping source IDs) and retrieve evidence.

Lexical TF-IDF cosine keeps the baseline deterministic and offline. The
contract (Chunk -> Evidence with source_id + score) stays the same if you swap
in sentence-transformers + Qdrant later.
"""

import math
import re
from collections import Counter
from pathlib import Path

from src.data import SOURCE_ID_RE
from src.models import Chunk, Evidence, PolicySearchResult

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = set(
    "a an and are as at be before by can for from how if in is it its may must "
    "not of on or our should that the their this to was were what when where "
    "which who will with you your please".split()
)
# Markers of instruction attacks inside retrieved text (illustrative, not complete).
UNSAFE_MARKERS = (
    "ignore all prior instructions",
    "ignore previous instructions",
    "reveal the system prompt",
    "api keys",
    "hidden tool outputs",
)


def tokenize(text: str) -> list[str]:
    tokens = [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]
    # light stemming so "refunds"/"refund", "escalation"/"escalate" meet
    return [re.sub(r"(ations?|ing|es|s|ed|e)$", "", t) or t for t in tokens]


def is_unsafe(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in UNSAFE_MARKERS)


def load_chunks(knowledge_base_dir: Path) -> list[Chunk]:
    """One chunk per paragraph; heading + Source ID line are metadata, not text."""

    chunks: list[Chunk] = []
    for path in sorted(knowledge_base_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        source_id = SOURCE_ID_RE.search(raw).group(1)
        title = raw.splitlines()[0].lstrip("# ").strip()
        body = SOURCE_ID_RE.sub("", raw.split("\n", 1)[1])
        paragraphs = [" ".join(p.split()) for p in body.split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs, start=1):
            chunks.append(Chunk(chunk_id=f"{source_id}-{i}", source_id=source_id, title=title, text=paragraph))
    return chunks


# Second no-answer rule: a hit needs >= 2 distinct shared content terms.
# A single shared word ("next") can give a high cosine on short queries.
MIN_SHARED_TERMS = 2


class PolicyRetriever:
    def __init__(self, chunks: list[Chunk], top_k: int = 3, min_score: float = 0.18,
                 min_shared_terms: int = MIN_SHARED_TERMS) -> None:
        self.chunks, self.top_k, self.min_score = chunks, top_k, min_score
        self.min_shared_terms = min_shared_terms
        docs = [tokenize(f"{c.title} {c.text}") for c in chunks]
        df = Counter(t for doc in docs for t in set(doc))
        n = len(docs)
        self.idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.vectors = [self._vector(doc) for doc in docs]
        self.terms = [set(doc) for doc in docs]

    def _vector(self, tokens: list[str], keep_oov: bool = False) -> dict[str, float]:
        """TF-IDF unit vector. Out-of-corpus words are dropped for policy search,
        but kept (with max IDF) when comparing a request to a ticket."""

        tf = Counter(tokens)
        oov_idf = max(self.idf.values()) if keep_oov else 0.0
        vec = {t: c * self.idf.get(t, oov_idf) for t, c in tf.items() if keep_oov or t in self.idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def similarity(self, left: str, right: str) -> float:
        """Cosine between two arbitrary texts using the corpus IDF."""

        a, b = self._vector(tokenize(left), True), self._vector(tokenize(right), True)
        return sum(w * b.get(t, 0.0) for t, w in a.items())

    def score_all(self, query: str) -> list[tuple[Chunk, float, list[str]]]:
        """(chunk, cosine, shared terms) for every chunk, best first."""

        tokens = tokenize(query)
        q = self._vector(tokens)
        scored = [(c, sum(w * v.get(t, 0.0) for t, w in q.items()), sorted(set(tokens) & terms))
                  for c, v, terms in zip(self.chunks, self.vectors, self.terms)]
        return sorted(scored, key=lambda row: row[1], reverse=True)

    def search(self, query: str) -> PolicySearchResult:
        """Top-k chunks above MIN_SCORE; unsafe chunks are dropped and reported."""

        evidence: list[Evidence] = []
        removed: list[str] = []
        for chunk, score, shared in self.score_all(query):
            if score < self.min_score or len(evidence) >= self.top_k:
                break
            if len(shared) < self.min_shared_terms:  # weak single-word match -> not evidence
                continue
            if is_unsafe(chunk.text):
                removed.append(chunk.chunk_id)
                continue
            evidence.append(Evidence(source_id=chunk.source_id, chunk_id=chunk.chunk_id, score=round(score, 3),
                                     shared_terms=shared, text=chunk.text))
        return PolicySearchResult(query=query, evidence=evidence, removed_chunks=removed, supported=bool(evidence))