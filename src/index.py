"""Chunk CVs with their context and search them with hybrid (embeddings + BM25) retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .parse import CV, skills_in

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge expects this instruction on queries (not on passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@dataclass
class Chunk:
    cand: int            # index of the CV
    text: str            # the line as written in the CV
    section: str         # Summary, Experience, Education, Skills...
    context: str = ""    # "NLP Engineer at Shirakawa Legal Tech", so a bullet still knows where it happened
    role: int | None = None
    head: bool = False   # the role's own title line
    tokens: list = field(default_factory=list)

    @property
    def full(self) -> str:
        return f"{self.context}: {self.text}" if self.context else self.text


def chunk_cv(cv: CV, cand: int) -> list[Chunk]:
    out = []
    if cv.summary:
        out.append(Chunk(cand, cv.summary, "Summary", f"{cv.name}, {cv.title}"))
    for ri, r in enumerate(cv.roles):
        where = f"{r.title} at {r.company}" if r.company else r.title
        out.append(Chunk(cand, where + (f" ({r.dates})" if r.dates else ""), "Experience", "", ri, head=True))
        out += [Chunk(cand, b, "Experience", where, ri) for b in r.bullets]
    out += [Chunk(cand, e, "Education", "Education") for e in cv.education]
    out += [Chunk(cand, c, "Certifications", "Certification") for c in cv.certifications]
    out += [Chunk(cand, p, "Projects", "Project") for p in cv.projects]
    if cv.skills_listed:
        out.append(Chunk(cand, ", ".join(cv.skills_listed), "Skills", "Skills"))
    if cv.languages:
        out.append(Chunk(cand, "Speaks " + ", ".join(cv.languages), "Languages", "Languages"))
    return out


# ---------- BM25, the keyword half of the hybrid search ----------
_STOP = set("""a an the and or of to in on for with by at from as is are was were be been has have had who what which
that this these those i my me we our do does did can could any some all about into over under than then it its their
them they he she his her you your experience years year candidate candidates someone anyone""".split())
_IRREG = {"led": "lead", "leading": "lead", "leads": "lead", "built": "build", "building": "build", "builds": "build",
          "ran": "run", "running": "run", "wrote": "write", "written": "write", "mentored": "mentor",
          "mentoring": "mentor", "speaks": "speak", "spoken": "speak", "deployed": "deploy",
          "deploying": "deploy", "deployment": "deploy", "managed": "manage", "managing": "manage"}


def tokens(text: str) -> list[str]:
    out = []
    for w in re.split(r"[^a-z0-9+#]+", text.lower()):
        if not w or w in _STOP:
            continue
        w = _IRREG.get(w, w)
        if len(w) > 4:
            w = re.sub(r"(ing|ed|es|s)$", "", w)
        out.append(w)
    return out


def _bm25(query_tokens, docs, k1=1.2, b=0.75) -> np.ndarray:
    n = len(docs)
    if not n:
        return np.zeros(0)
    avg = sum(len(d) for d in docs) / n or 1
    counts = [Counter(d) for d in docs]
    scores = np.zeros(n)
    for t in set(query_tokens):
        df = sum(1 for c in counts if t in c)
        if not df:
            continue
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        for i, c in enumerate(counts):
            f = c.get(t, 0)
            if f:
                scores[i] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(docs[i]) / avg))
    top = scores.max()
    return scores / top if top > 0 else scores


@dataclass
class Hit:
    chunk: Chunk
    score: float
    semantic: float
    keyword: float


_EMBEDDER = None  # shared by every index; kept out of the index so per-visitor copies stay small


def set_embedder(embedder) -> None:
    global _EMBEDDER
    _EMBEDDER = embedder


class CVIndex:
    """Parsed CVs plus every chunk, with embeddings when the model is available.

    The index holds only data (CVs, chunks, vectors), so it can be copied per
    visitor; the embedding model is shared through set_embedder().
    """

    def __init__(self):
        self.cvs: list[CV] = []
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None

    @property
    def embedder(self):
        return _EMBEDDER

    @property
    def semantic(self) -> bool:
        return self.embedder is not None

    def add(self, cv: CV) -> int:
        cand = len(self.cvs)
        self.cvs.append(cv)
        new = chunk_cv(cv, cand)
        for c in new:
            c.tokens = tokens(c.full)
        self.chunks += new
        if self.embedder is not None:
            v = self.embedder.encode([c.full for c in new])
            self.vectors = v if self.vectors is None else np.vstack([self.vectors, v])
        return cand

    def search(self, query: str) -> list[Hit]:
        """Every chunk scored against the query, best first."""
        kw = _bm25(tokens(query), [c.tokens for c in self.chunks])
        if self.embedder is not None and self.vectors is not None:
            q = self.embedder.encode([query], query=True)[0]
            sem = self.vectors @ q
        else:
            sem = np.zeros(len(self.chunks))
        q_skills = set(skills_in(query))
        hits = []
        for i, c in enumerate(self.chunks):
            bonus = 0.08 if q_skills and q_skills & set(skills_in(c.text)) else 0.0
            score = (sem[i] + 0.2 * kw[i] + bonus) if self.semantic else (kw[i] + bonus)
            hits.append(Hit(c, float(score), float(sem[i]), float(kw[i])))
        return sorted(hits, key=lambda h: -h.score)

    def rank_candidates(self, query: str, per_cand: int = 3):
        """Candidates ordered by their best few chunks for this query."""
        hits = self.search(query)
        ranked = []
        for cand in range(len(self.cvs)):
            mine = [h for h in hits if h.chunk.cand == cand][:per_cand]
            if not mine:
                continue
            score = mine[0].score + 0.3 * (mine[1].score if len(mine) > 1 else 0)
            margin = 0.12 if self.semantic else 0.25
            ranked.append({"cand": cand, "score": score, "evidence": [h for h in mine if h.score >= mine[0].score - margin]})
        ranked.sort(key=lambda r: -r["score"])
        return ranked, hits

    def match_job(self, job_description: str):
        """Score every candidate against each requirement line of a job description."""
        lines = [l.strip() for l in job_description.split("\n") if l.strip()]
        bullets = [l for l in lines if re.match(r"^[-•*·]", l)]
        reqs = [re.sub(r"^[-•*·]\s*", "", l) for l in (bullets or lines[1:] or lines)]
        reqs = [r for r in reqs if len(r) > 8][:12]
        rows = []
        for req in reqs:
            years = re.search(r"(\d+)\+?\s*(?:years?|yrs)", req, re.I)
            cells = []
            hits = None if years else self.search(req)
            for cand, cv in enumerate(self.cvs):
                if years:
                    need = int(years.group(1))
                    cells.append({"value": min(1.0, cv.years / need) if need else 1.0, "kind": "years",
                                  "why": f"{cv.years} years of experience, from role dates"})
                    continue
                best = next(h for h in hits if h.chunk.cand == cand)
                if self.semantic:
                    value = (best.score - 0.58) / 0.32
                else:
                    value = best.score
                cells.append({"value": float(max(0.0, min(1.0, value))), "kind": "evidence", "why": best.chunk.text,
                              "section": best.chunk.section, "context": best.chunk.context, "raw": best.score})
            rows.append({"requirement": req, "cells": cells})
        totals = [sum(r["cells"][c]["value"] for r in rows) / len(rows) if rows else 0 for c in range(len(self.cvs))]
        order = sorted(range(len(self.cvs)), key=lambda c: -totals[c])
        return {"requirements": rows, "totals": totals, "order": order}


class Embedder:
    """bge-small sentence embeddings on CPU (it's small enough that a GPU isn't worth it)."""

    def __init__(self, model_name: str = EMBED_MODEL):
        from sentence_transformers import SentenceTransformer

        # device is fixed to CPU: on ZeroGPU hardware a GPU is reported but only usable inside @spaces.GPU calls
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts, query: bool = False) -> np.ndarray:
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        return np.asarray(self.model.encode(list(texts), normalize_embeddings=True, batch_size=32,
                                            show_progress_bar=False), dtype=np.float32)


def load_embedder():
    """The embedding model, or None (keyword search only) if it can't be loaded."""
    try:
        return Embedder()
    except Exception as exc:  # no network, or sentence-transformers missing
        print(f"[cv-parser] embeddings unavailable, using keyword search only: {exc}")
        return None
