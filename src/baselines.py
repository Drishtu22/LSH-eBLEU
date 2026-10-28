import argparse

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF

from evaluation import correlations


SPLITS = {
    "dev": "data/processed/wmt19_dev.csv",
    "test": "data/processed/wmt20_test.csv",
}

# sacrebleu's default 13a tokenizer does not split Chinese, so en-zh needs "zh"
BLEU_13A = BLEU(tokenize="13a", effective_order=True)
BLEU_ZH = BLEU(tokenize="zh", effective_order=True)
CHRF_METRIC = CHRF()


def target_language(lp):
    return lp.split("-")[1]


def sentence_bleu(mt, ref, lp):
    metric = BLEU_ZH if target_language(lp) == "zh" else BLEU_13A
    return metric.sentence_score(mt, [ref]).score


def sentence_chrf(mt, ref):
    return CHRF_METRIC.sentence_score(mt, [ref]).score


def score_split(df):
    df = df.copy()
    df["bleu"] = [
        sentence_bleu(mt, ref, lp) for mt, ref, lp in zip(df["mt"], df["ref"], df["lp"])
    ]
    df["chrf"] = [sentence_chrf(mt, ref) for mt, ref in zip(df["mt"], df["ref"])]
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="dev")
    args = parser.parse_args()

    data = pd.read_csv(SPLITS[args.split])

    print(f"Scoring {len(data)} segments ({args.split})...")
    scored = score_split(data)

    table = correlations(scored, ["bleu", "chrf"])

    scored[["lp", "score", "bleu", "chrf"]].to_csv(
        f"results/baseline_scores_{args.split}.csv", index=False
    )
    table.to_csv(f"results/baseline_correlations_{args.split}.csv", index=False)

    print(
        table.pivot(index="lp", columns="metric", values=["pearson", "kendall"])
        .round(3)
        .to_string()
    )
