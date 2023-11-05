import pandas as pd
import numpy as np
import os
import sys
import datasets


def main():
    # SET MANUAL RANDOM SEED
    np.random.seed(42)
    args = sys.argv[1:]

    print("Loading raw data from tsv to Pandas DataFrame")
    # load raw data from csv
    RAW_DATA_DIR = args[0]
    dataset = pd.read_csv(RAW_DATA_DIR + "filtered.tsv", sep="\t", index_col=0)

    print("Spliting data by task")
    # split data by task (toxic2nontoxic and nontoxic2toxic)
    detoxification_samples = dataset[dataset["ref_tox"] >= dataset["trn_tox"]]
    toxification_samples = dataset[dataset["ref_tox"] < dataset["trn_tox"]]

    SAVE_PATH = args[1]
    toxification_samples.to_csv(SAVE_PATH + "toxification_data.csv")
    detoxification_samples.to_csv(SAVE_PATH + "detoxification_data.csv")

    # rename DataFrame features to make it suitable to detoxication
    toxification_samples.columns = [
        "translation",
        "reference",
        "similarity",
        "lenght_diff",
        "trn_tox",
        "ref_tox",
    ]

    tox_dataset = datasets.Dataset.from_pandas(toxification_samples)
    detox_dataset = datasets.Dataset.from_pandas(detoxification_samples)

    print(
        "Merging splitted DataFrames into single HuggingFace Dataset with data for text detoxification task"
    )
    merged_dataset = datasets.concatenate_datasets([tox_dataset, detox_dataset])
    merged_dataset = merged_dataset.train_test_split(train_size=0.95)
    merged_dataset.save_to_disk(SAVE_PATH + "hf_dataset")
    print("Merged Dataset saved to: " + SAVE_PATH + "hf_dataset")


if __name__ == "__main__":
    main()
