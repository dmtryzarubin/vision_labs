from pathlib import Path
from typing import Union

import fire
from sklearn.model_selection import train_test_split

from src.preprocess.functional import open_json, save_json

RANDOM_STATE = 2023


def split(dataset_path: str, val_size: Union[float, int] = 500) -> None:
    """
    Splits dataset into train/val and test split

    :param dataset_path: Path to dataset, obtained by create_annotation script
    :param val_size: Fraction if float or num samples to be in val split, defaults to 500
    :return: Dict with splits as keys and saved paths
    """
    datadir = Path(dataset_path).parent
    data = open_json(dataset_path)
    test = [d for d in data if d["split"] == "test"]
    train = [d for d in data if d["split"] == "train"]
    train, val = train_test_split(train, test_size=val_size, random_state=RANDOM_STATE)
    paths = {
        "train": datadir / "train.json",
        "val": datadir / "val.json",
        "test": datadir / "test.json",
    }
    save_json(train, paths["train"])
    save_json(val, paths["val"])
    save_json(test, paths["test"])
    return paths


if __name__ == "__main__":
    fire.Fire(split)
