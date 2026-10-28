from datasets import load_dataset
import pandas as pd


LANGUAGE_PAIRS = [
    "de-en",
    "en-cs",
    "en-de",
    "en-ru",
    "en-zh",
    "ru-en",
    "zh-en",
]

SAMPLES_PER_PAIR = 560
RANDOM_SEED = 42


def load_full_dataset():
    print("Loading WMT Direct Assessment dataset...")

    dataset = load_dataset(
        "RicardoRei/wmt-da-human-evaluation",
        split="train"
    )

    print("Dataset loaded successfully.")
    return dataset


def create_split(dataset, year):
    rows = []

    for lp in LANGUAGE_PAIRS:

        subset = dataset.filter(
            lambda row: row["year"] == year and row["lp"] == lp
        )

        df = subset.to_pandas()

        # Keep only fields needed for this project
        df = df[
            ["lp", "src", "mt", "ref", "score"]
        ]

        # Remove incomplete examples
        df = df.dropna()

        # Fixed random sample for reproducibility
        df = df.sample(
            n=SAMPLES_PER_PAIR,
            random_state=RANDOM_SEED
        )

        rows.append(df)

        print(f"{year} | {lp} | selected {len(df)} examples")

    final_df = pd.concat(rows, ignore_index=True)

    return final_df


if __name__ == "__main__":

    dataset = load_full_dataset()

    print("\nCreating WMT19 development set...")
    wmt19 = create_split(dataset, 2019)

    print("\nCreating WMT20 test set...")
    wmt20 = create_split(dataset, 2020)

    wmt19.to_csv(
        "data/processed/wmt19_dev.csv",
        index=False
    )

    wmt20.to_csv(
        "data/processed/wmt20_test.csv",
        index=False
    )

    print("\n--------------------------------")
    print("WMT19 development rows:", len(wmt19))
    print("WMT20 test rows:", len(wmt20))

    print("\nWMT19 counts:")
    print(wmt19["lp"].value_counts().sort_index())

    print("\nWMT20 counts:")
    print(wmt20["lp"].value_counts().sort_index())

    print("\nProcessed datasets saved successfully.")
