import argparse
import gzip
import hashlib
import io
import os
import urllib.request

import jieba
import numpy as np
import pandas as pd
from sacrebleu.tokenizers.tokenizer_13a import Tokenizer13a


SPLIT_FILES = ["data/processed/wmt19_dev.csv", "data/processed/wmt20_test.csv"]
EMBEDDING_DIR = "data/embeddings"
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.{lang}.300.vec.gz"

# fastText .vec files are sorted by frequency; we read only this many lines
MAX_WORDS = 400_000
# Vocabulary size of the full cc.*.300 files, used for the Zipf estimate of P(w)
FULL_VOCAB = 2_000_000
DIM = 300
SIF_A = 1e-3

TARGET_LANGUAGES = ["en", "cs", "de", "ru", "zh"]

_tokenizer_13a = Tokenizer13a()
jieba.setLogLevel(60)


def tokenize(text, lang):
    """Same tokenization for BLEU, soft matching and LSH."""
    if lang == "zh":
        return [t for t in jieba.lcut(text) if t.strip()]
    return _tokenizer_13a(text).split()


def target_language(lp):
    return lp.split("-")[1]


def corpus_vocabulary(lang):
    vocab = set()
    for path in SPLIT_FILES:
        df = pd.read_csv(path)
        df = df[df["lp"].map(target_language) == lang]
        for text in pd.concat([df["mt"], df["ref"]]):
            for token in tokenize(text, lang):
                vocab.add(token)
                vocab.add(token.lower())
    return vocab


def download_vectors(lang):
    """Stream the top MAX_WORDS fastText vectors, keeping only corpus words."""
    vocab = corpus_vocabulary(lang)
    words, vectors, ranks = [], [], []

    response = urllib.request.urlopen(FASTTEXT_URL.format(lang=lang))
    stream = io.TextIOWrapper(gzip.GzipFile(fileobj=response), encoding="utf-8")
    next(stream)  # header: "<count> <dim>"

    for rank, line in enumerate(stream, start=1):
        if rank > MAX_WORDS:
            break
        word, rest = line.rstrip("\n").split(" ", 1)
        if word in vocab:
            words.append(word)
            vectors.append(np.array(rest.split(), dtype=np.float32))
            ranks.append(rank)

    response.close()

    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    np.savez(
        f"{EMBEDDING_DIR}/{lang}.npz",
        words=np.array(words),
        vectors=np.stack(vectors),
        ranks=np.array(ranks),
    )
    print(f"{lang}: kept {len(words)} of {len(vocab)} corpus word forms")


class Embedder:
    """Word vectors for one language plus SIF weights alpha = a / (a + P(w))."""

    def __init__(self, lang):
        data = np.load(f"{EMBEDDING_DIR}/{lang}.npz")
        vectors = data["vectors"]
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        # Zipf's law: P(rank) = 1 / (rank * H_N)
        harmonic = np.log(FULL_VOCAB) + 0.5772
        probabilities = 1.0 / (data["ranks"] * harmonic)

        self.lang = lang
        self.index = {w: i for i, w in enumerate(data["words"].tolist())}
        self.vectors = vectors
        self.alphas = SIF_A / (SIF_A + probabilities)
        self.oov_alpha = SIF_A / (SIF_A + 1.0 / (FULL_VOCAB * harmonic))
        self._oov_cache = {}

    def _oov_vector(self, word):
        # Deterministic random unit vector: an unknown word matches itself
        # exactly and is near-orthogonal to everything else.
        if word not in self._oov_cache:
            seed = int.from_bytes(hashlib.md5(word.encode("utf-8")).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
            self._oov_cache[word] = v / np.linalg.norm(v)
        return self._oov_cache[word]

    def word_vectors(self, tokens):
        """(len(tokens), DIM) unit vectors and (len(tokens),) SIF weights."""
        vectors = np.empty((len(tokens), DIM), dtype=np.float32)
        alphas = np.empty(len(tokens), dtype=np.float32)

        for i, token in enumerate(tokens):
            idx = self.index.get(token, self.index.get(token.lower()))
            if idx is None:
                vectors[i] = self._oov_vector(token)
                alphas[i] = self.oov_alpha
            else:
                vectors[i] = self.vectors[idx]
                alphas[i] = self.alphas[idx]

        return vectors, alphas


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("lang", choices=TARGET_LANGUAGES)
    args = parser.parse_args()

    download_vectors(args.lang)
