import pandas as pd
from scipy.stats import kendalltau, pearsonr


def correlations(df, metric_columns, human_column="score"):
    """Pearson and Kendall correlation with human scores, per language pair.

    Returns one row per (lp, metric) plus an "avg" row per metric that is the
    unweighted mean over language pairs.
    """
    rows = []

    for lp, group in df.groupby("lp"):
        for metric in metric_columns:
            rho, _ = pearsonr(group[metric], group[human_column])
            tau, _ = kendalltau(group[metric], group[human_column])
            rows.append({"lp": lp, "metric": metric, "pearson": rho, "kendall": tau})

    table = pd.DataFrame(rows)

    averages = (
        table.groupby("metric", sort=False)[["pearson", "kendall"]]
        .mean()
        .reset_index()
        .assign(lp="avg")
    )

    return pd.concat([table, averages], ignore_index=True)
