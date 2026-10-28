"""Correctness checks for soft_bleu.py, run on the WMT19 dev set."""

import numpy as np
import pandas as pd

from baselines import sentence_bleu
from embeddings import Embedder, target_language
from soft_bleu import counts_exact, counts_lsh, hyperplanes, prepare, score


def main():
    data = pd.read_csv("data/processed/wmt19_dev.csv")
    planes = hyperplanes()
    embedders = {}

    def embedder_for(lp):
        lang = target_language(lp)
        if lang not in embedders:
            embedders[lang] = Embedder(lang)
        return embedders[lang]

    # 1. delta = 1 keeps only exact matches, so soft BLEU must equal sacrebleu's
    #    sentence BLEU (en-zh excluded: we tokenize Chinese with jieba, sacrebleu by character).
    non_zh = data[data["lp"] != "en-zh"].sample(500, random_state=0)
    worst_exact, worst_lsh = 0.0, 0.0
    for mt, ref, lp in zip(non_zh["mt"], non_zh["ref"], non_zh["lp"]):
        expected = sentence_bleu(mt, ref, lp)
        exact, _, _ = score(mt, ref, embedder_for(lp), delta=1.0)
        lsh, _, _ = score(mt, ref, embedder_for(lp), delta=1.0, k=16, L=1, planes=planes)
        worst_exact = max(worst_exact, abs(exact - expected))
        worst_lsh = max(worst_lsh, abs(lsh - expected))
    print(f"delta=1 vs sacrebleu BLEU, max abs diff: exact {worst_exact:.2e}, LSH {worst_lsh:.2e}")
    assert worst_exact < 1e-3 and worst_lsh < 1e-3

    # 2. The precomputed path used for tuning must agree with the real metric.
    sample = data.sample(300, random_state=1)
    worst = 0.0
    for mt, ref, lp in zip(sample["mt"], sample["ref"], sample["lp"]):
        embedder = embedder_for(lp)
        segment = prepare(mt, ref, embedder, planes)
        for delta, k, L in [(0.6, 8, 16), (0.8, 4, 2), (0.7, 12, 32)]:
            direct = score(mt, ref, embedder, delta, k=k, L=L, planes=planes)
            cached = counts_lsh(segment, delta, k, L)
            assert direct[1:] == cached[1:], (direct, cached)
            worst = max(worst, abs(direct[0] - cached[0]))
        direct = score(mt, ref, embedder, 0.7)
        cached = counts_exact(segment, 0.7)
        assert direct[1:] == cached[1:]
        worst = max(worst, abs(direct[0] - cached[0]))
    print(f"direct vs precomputed path, max abs diff: {worst:.2e}")
    assert worst < 1e-3

    # 3. The slide 2 example: soft matching must credit the synonyms.
    ref = "the car was parked outside the house"
    mt = "the automobile stood outside the home"
    en = Embedder("en")
    print("slide example | BLEU", round(sentence_bleu(mt, ref, "de-en"), 2),
          "| soft (delta=0.5)", round(score(mt, ref, en, 0.5)[0], 2))
    for a, b in [("car", "automobile"), ("house", "home"), ("parked", "stood")]:
        va, _ = en.word_vectors([a])
        vb, _ = en.word_vectors([b])
        print(f"  cos({a}, {b}) = {float(va[0] @ vb[0]):.3f}")

    # 4. Empirical collision rate vs eq. (10) for unigram pairs.
    k, L = 8, 16
    cos_values, collided = [], []
    for mt, ref, lp in zip(sample["mt"][:100], sample["ref"][:100], sample["lp"][:100]):
        order = prepare(mt, ref, embedder_for(lp), planes).orders[0]
        mask = (1 << k) - 1
        same = (order.cand_keys[:, None, :L] & mask) == (order.ref_keys[None, :, :L] & mask)
        cos_values.append(order.sims.ravel())
        collided.append(same.any(axis=2).ravel())
    cos_values = np.concatenate(cos_values)
    collided = np.concatenate(collided)
    print(f"collision rate, k={k}, L={L}: empirical vs theory (eq. 10)")
    for lo in np.arange(0.0, 1.0, 0.2):
        in_bin = (cos_values >= lo) & (cos_values < lo + 0.2)
        mid = np.clip(cos_values[in_bin], -1, 1)
        theory = 1 - (1 - (1 - np.arccos(mid) / np.pi) ** k) ** L
        print(f"  cos [{lo:.1f}, {lo + 0.2:.1f}): n={in_bin.sum():6d} "
              f"empirical {collided[in_bin].mean():.3f}  theory {theory.mean():.3f}")

    print("All checks passed.")


if __name__ == "__main__":
    main()
