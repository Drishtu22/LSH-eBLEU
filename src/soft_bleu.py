"""Soft n-gram BLEU with exact all-pairs matching or LSH candidate search.

Two code paths share the same building blocks:

* ``score`` is the real metric: it computes only the dot products it needs
  (all pairs for exact matching, colliding pairs plus hash projections for LSH).
* ``prepare`` / ``counts_exact`` / ``counts_lsh`` precompute every similarity
  and every hash bit of a segment once, so a whole grid of (delta, k, L)
  settings can be evaluated without recomputing embeddings. Similarities that
  the LSH path would not have computed are never read.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import exp, log

import numpy as np

from embeddings import DIM, tokenize


MAX_ORDER = 4
MAX_TABLES = 32
MAX_BITS = 16
EXACT_TOL = 1e-6
# Eq. (4) is a bag-of-words sum, so a reordered n-gram ("12 jeho" / "jeho 12")
# has cosine 1 with the original. Capping non-identical pairs just below 1
# keeps the property that delta = 1 accepts exact matches only (= BLEU).
NON_IDENTICAL_MAX = 1.0 - 10 * EXACT_TOL


def hyperplanes(seed=42):
    """Random normals r ~ N(0, I_d); config (k, L) uses planes[:L, :k]."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((MAX_TABLES, MAX_BITS, DIM)).astype(np.float32)


def ngram_ids(tokens, n, vocabulary):
    """Integer id per n-gram; equal ids mean identical token tuples."""
    return np.array(
        [vocabulary.setdefault(tuple(tokens[i:i + n]), len(vocabulary))
         for i in range(len(tokens) - n + 1)],
        dtype=np.int64,
    )


def ngram_vectors(word_vectors, alphas, n):
    """Unit SIF-weighted n-gram embeddings, eq. (4).

    Windows are summed slice by slice (not with a cumulative sum) so that the
    same n-gram always gets bit-identical vectors and therefore identical hash
    keys: exact matches can never be lost by LSH.
    """
    m = len(word_vectors) - n + 1
    if m <= 0:
        return np.zeros((0, DIM), dtype=np.float32)

    weighted = word_vectors * alphas[:, None]
    sums = weighted[0:m].copy()
    for i in range(1, n):
        sums += weighted[i:i + m]

    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    return sums / np.maximum(norms, 1e-12)


def hash_keys(bits, k, L):
    """Pack the first k bits of the first L tables into one integer per table."""
    weights = np.int64(1) << np.arange(k, dtype=np.int64)
    return bits[:, :L, :k].astype(np.int64) @ weights


def lsh_pairs(cand_keys, ref_keys):
    """All (candidate, reference) index pairs sharing a bucket in any table, eq. (11)."""
    pairs = set()
    for j in range(cand_keys.shape[1]):
        buckets = defaultdict(list)
        for r, key in enumerate(ref_keys[:, j].tolist()):
            buckets[key].append(r)
        for c, key in enumerate(cand_keys[:, j].tolist()):
            for r in buckets.get(key, ()):
                pairs.add((c, r))

    if not pairs:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty

    pair_array = np.array(sorted(pairs), dtype=np.int64)
    return pair_array[:, 0], pair_array[:, 1]


def greedy_credit(cand_idx, ref_idx, sims, delta):
    """Greedy one-to-one matching in descending similarity, eq. (12)-(13).

    Each candidate n-gram and each reference n-gram position is used at most
    once, which is BLEU's clipping by Count(g; y) generalised to soft matches.
    """
    keep = sims >= delta - EXACT_TOL
    cand_idx, ref_idx, sims = cand_idx[keep], ref_idx[keep], sims[keep]

    used_cand, used_ref = set(), set()
    credit = 0.0
    for p in np.argsort(-sims, kind="stable"):
        c, r = int(cand_idx[p]), int(ref_idx[p])
        if c in used_cand or r in used_ref:
            continue
        used_cand.add(c)
        used_ref.add(r)
        credit += min(float(sims[p]), 1.0)

    return credit


def bleu_from_counts(correct, total, cand_len, ref_len):
    """Sentence BLEU from (soft) matched counts, eqs. (2), (3), (14).

    Mirrors sacrebleu's sentence_bleu: exponential smoothing for zero
    precisions and effective order for sentences shorter than four tokens.
    """
    if not any(c > 0 for c in correct):
        return 0.0

    precisions = []
    smooth = 1.0
    for n in range(MAX_ORDER):
        if total[n] == 0:
            break
        if correct[n] <= 0:
            smooth *= 2.0
            precisions.append(1.0 / (smooth * total[n]))
        else:
            precisions.append(correct[n] / total[n])

    if not precisions:
        return 0.0

    if cand_len > ref_len:
        bp = 1.0
    else:
        bp = exp(1.0 - ref_len / cand_len) if cand_len > 0 else 0.0

    return 100.0 * bp * exp(sum(log(p) for p in precisions) / len(precisions))


def _embed_pair(mt, ref, embedder):
    cand_tokens = tokenize(mt, embedder.lang)
    ref_tokens = tokenize(ref, embedder.lang)
    cand_words, cand_alphas = embedder.word_vectors(cand_tokens)
    ref_words, ref_alphas = embedder.word_vectors(ref_tokens)

    vocabulary = {}
    orders = []
    for n in range(1, MAX_ORDER + 1):
        orders.append((
            ngram_ids(cand_tokens, n, vocabulary),
            ngram_ids(ref_tokens, n, vocabulary),
            ngram_vectors(cand_words, cand_alphas, n),
            ngram_vectors(ref_words, ref_alphas, n),
        ))

    return len(cand_tokens), len(ref_tokens), orders


def score(mt, ref, embedder, delta, k=None, L=None, planes=None):
    """LSH-eBLEU (k, L, planes given) or exact soft BLEU (k = L = None).

    Returns (score, similarity_computations, hash_projections).
    """
    cand_len, ref_len, orders = _embed_pair(mt, ref, embedder)
    use_lsh = k is not None

    if use_lsh:
        flat_planes = planes[:L, :k].reshape(L * k, DIM)

    correct, total = [], []
    similarity_computations = 0
    hash_projections = 0

    for cand_ids, ref_ids, cand_vecs, ref_vecs in orders:
        if use_lsh:
            cand_bits = (cand_vecs @ flat_planes.T >= 0).reshape(-1, L, k)
            ref_bits = (ref_vecs @ flat_planes.T >= 0).reshape(-1, L, k)
            hash_projections += (len(cand_vecs) + len(ref_vecs)) * L * k

            cand_idx, ref_idx = lsh_pairs(hash_keys(cand_bits, k, L), hash_keys(ref_bits, k, L))
            sims = np.einsum("ij,ij->i", cand_vecs[cand_idx], ref_vecs[ref_idx])
        else:
            cand_idx, ref_idx = np.divmod(np.arange(len(cand_vecs) * len(ref_vecs)), len(ref_vecs))
            sims = (cand_vecs @ ref_vecs.T).ravel()

        sims = np.minimum(sims, NON_IDENTICAL_MAX)
        sims[cand_ids[cand_idx] == ref_ids[ref_idx]] = 1.0
        similarity_computations += len(sims)

        correct.append(greedy_credit(cand_idx, ref_idx, sims, delta))
        total.append(len(cand_vecs))

    return (
        bleu_from_counts(correct, total, cand_len, ref_len),
        similarity_computations,
        hash_projections,
    )


@dataclass
class PreparedOrder:
    sims: np.ndarray        # (c_n, r_n) cosine matrix, exact matches set to 1
    cand_keys: np.ndarray   # (c_n, MAX_TABLES) MAX_BITS-bit keys; low k bits = k-bit key
    ref_keys: np.ndarray    # (r_n, MAX_TABLES)


@dataclass
class PreparedSegment:
    cand_len: int
    ref_len: int
    orders: list


def prepare(mt, ref, embedder, planes):
    cand_len, ref_len, orders = _embed_pair(mt, ref, embedder)
    flat_planes = planes.reshape(-1, DIM)

    prepared = []
    for cand_ids, ref_ids, cand_vecs, ref_vecs in orders:
        sims = np.minimum(cand_vecs @ ref_vecs.T, NON_IDENTICAL_MAX)
        sims[cand_ids[:, None] == ref_ids[None, :]] = 1.0
        cand_bits = (cand_vecs @ flat_planes.T >= 0).reshape(-1, MAX_TABLES, MAX_BITS)
        ref_bits = (ref_vecs @ flat_planes.T >= 0).reshape(-1, MAX_TABLES, MAX_BITS)
        prepared.append(PreparedOrder(
            sims=sims,
            cand_keys=hash_keys(cand_bits, MAX_BITS, MAX_TABLES).astype(np.int32),
            ref_keys=hash_keys(ref_bits, MAX_BITS, MAX_TABLES).astype(np.int32),
        ))

    return PreparedSegment(cand_len, ref_len, prepared)


def counts_exact(segment, delta):
    """(score, similarity_computations, hash_projections) with all-pairs matching."""
    correct, total, computations = [], [], 0
    for order in segment.orders:
        c, r = order.sims.shape
        cand_idx, ref_idx = np.divmod(np.arange(c * r), r)
        correct.append(greedy_credit(cand_idx, ref_idx, order.sims.ravel(), delta))
        total.append(c)
        computations += c * r

    return bleu_from_counts(correct, total, segment.cand_len, segment.ref_len), computations, 0


def counts_lsh(segment, delta, k, L):
    """(score, similarity_computations, hash_projections) with LSH candidates."""
    mask = (1 << k) - 1
    correct, total, computations, projections = [], [], 0, 0
    for order in segment.orders:
        cand_idx, ref_idx = lsh_pairs(
            order.cand_keys[:, :L] & mask, order.ref_keys[:, :L] & mask
        )
        correct.append(greedy_credit(cand_idx, ref_idx, order.sims[cand_idx, ref_idx], delta))
        total.append(order.sims.shape[0])
        computations += len(cand_idx)
        projections += sum(order.sims.shape) * L * k

    return (
        bleu_from_counts(correct, total, segment.cand_len, segment.ref_len),
        computations,
        projections,
    )
